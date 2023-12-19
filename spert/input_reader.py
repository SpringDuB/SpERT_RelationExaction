import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer

from spert import util
from spert.entities import Dataset, EntityType, RelationType, Entity, Relation, Document, Token
from spert.util import build_text_to_tokens_func


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None, **kwargs):
        types = json.load(
            open(types_path, "r", encoding="utf-8"),
            object_pairs_hook=OrderedDict
        )  # entity + relation types 实体关系标签

        self._entity_types = OrderedDict()  # 名称到实体类别
        self._idx2entity_type = OrderedDict()  # id到实体类别
        self._relation_types = OrderedDict()  # 名称关系类别
        self._idx2relation_type = OrderedDict()  # id到关系类别

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')  # "空实体" 这个类别
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types 遍历所有实体元数据，构建所有实体的映射关系
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')  # "空关系" 这个类别
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types 遍历所有关系元数据，构建关系的映射原始信息
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size  # 词汇表大小

    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, logger)

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)  # 数据解析给定的文件数据，并将解析结果更新到dataset对象中
        self._datasets[dataset_label] = dataset  # 保持到当前内存对象中
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path, "r", encoding="utf-8"))  # 原始训练数据加载 --> 文本的列表
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)  # 解析当前文本中的实体和关系

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']  # 当前文本对应的实际token列表
        jrelations = doc['relations']  # 当前文本对应的实体列表
        jentities = doc['entities']  # 当前文本对应的关系列表

        # parse tokens 解析token --> 得到token的id以及token span的范围
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # parse entity mentions 解析实体 --> 将实体对应的多个token span组合到一起，并保存到dataset对象的_entities属性中
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # parse relations 解析关系 --> 将主体/客体实际对象以及关系类型组合到一起，并保存到dataset对象的_relations属性中
        relations = self._parse_relations(jrelations, entities, dataset)

        # create document 创建文本对象 --> 将token span、token id、entity、relation四个列表组成一个doc对象，并保存
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)

        return document

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []  # 保存实体对象

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]  # 基于实体type名称获取实体元数据
            start, end = jentity['start'], jentity['end']  # 实体在实际token序列中的范围

            # create entity mention
            tokens = doc_tokens[start:end]  # 获取实体对应的token列表/范围
            phrase = " ".join([t.phrase for t in tokens])  # 将token拼接成文本字符串
            entity = dataset.create_entity(entity_type, tokens, phrase)  # 创建实体对象
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]  # 根据关系type名称获取对应元数据

            head_idx = jrelation['head']  # 主体的下标
            tail_idx = jrelation['tail']  # 客体的下标

            # create relation
            head = entities[head_idx]  # 获取主体对应的实体对象
            tail = entities[tail_idx]  # 获取客体对应的实体对象

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations


class JsonPredictionInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, spacy_model: str = None,
                 max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, max_span_size=max_span_size, logger=logger)
        self._spacy_model = spacy_model

        self._nlp = build_text_to_tokens_func(spacy_model)  # 构造一个文本划分token的对象/方法

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path, "r", encoding="utf-8"))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset) -> Document:
        if type(document) == list:
            jtokens = document
        elif type(document) == dict:
            jtokens = document['tokens']
        else:
            jtokens = self._nlp(document)  # 文本分词

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # create document
        if dataset is None:
            document = Document(0, doc_tokens, [], [], doc_encoding)
        else:
            document = dataset.create_document(doc_tokens, [], [], doc_encoding)

        return document

    def parse_tokens(self, document):
        return self._parse_document(document, dataset=None)


def _parse_tokens(jtokens, dataset, tokenizer: BertTokenizer):
    doc_tokens = []  # 当前文档的实际token对象列表

    # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]  # 添加前缀 [CLS]对应的token id

    # parse tokens 针对每个token，提取对应的id
    for i, token_phrase in enumerate(jtokens):
        # 利用迁移过来的tokenizer解析器进行token id转换 NOTE: token_encoding长度可能是0、1、2、....
        token_encoding = tokenizer.encode(token_phrase.lower(), add_special_tokens=False)
        if not token_encoding:
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
        span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))  # 当前token对应的起始id和结束id

        if dataset is None:
            token = Token(0, i, span_start, span_end, token_phrase)
        else:
            token = dataset.create_token(i, span_start, span_end, token_phrase)

        doc_tokens.append(token)
        doc_encoding += token_encoding

    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]  # 结尾id添加

    return doc_tokens, doc_encoding
