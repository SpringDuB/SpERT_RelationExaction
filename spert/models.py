import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]  # 得到embedding的大小

    token_h = h.view(-1, emb_size)  # [N,T,E] --> [N*T,E]
    flat = x.contiguous().view(-1)  # [N,T] -> [N*T]

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]  # 提取所有特征为[CLS]对应的特征输出

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model 基础语言模型:将token转换为向量/每个token对应的上下文特征向量
        self.bert = BertModel(config)
        self.fusion_layers = min(config.num_hidden_layers, 3)
        self.dym_weight = nn.Parameter(torch.ones(self.fusion_layers, 1, 1, 1))
        nn.init.xavier_normal_(self.dym_weight)  # 参数初始化

        # layers 关系分类的分支 + 命名实体识别的分支
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)  # 关系决策输出
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)  # 实体决策输出
        self.size_embeddings = nn.Embedding(100, size_embedding)  # entity span的长度映射向量表
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token  # [CLS]对应的token id eg:101
        self._relation_types = relation_types  # 关系的类别数量 eg:5
        self._entity_types = entity_types  # 实体的类别数量 eg:5
        self._max_pairs = max_pairs  # 最大允许的实体关系组合对数目 eg:1000

        # weight initialization 参数初始化
        self.init_weights()

        if freeze_transformer:  # 是否冻结bert语言模型部分参数
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _fetch_bert_embedding(self, encodings, context_masks):
        # get contextualized token embeddings from last transformer layer 仅获取bert/基础语言模型最后一层的输出
        context_masks = context_masks.float()
        # h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']  # [N,T,E]

        outputs = self.bert(
            input_ids=encodings,
            attention_mask=context_masks,
            output_hidden_states=True  # 是否返回每一层的结果值
        )  # [N,T] --> [N,T,E]
        # 将list/tuple形式的tensor合并成一个tensor对象, [fusion_layers, N,T,E]
        hidden_stack = torch.stack(outputs.hidden_states[-self.fusion_layers:], dim=0)
        hidden_stack = hidden_stack * torch.softmax(self.dym_weight, dim=0)
        h = torch.sum(hidden_stack, dim=0)  # [fusion_layers, N,T,E] --> [N,T,E]

        return h

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        """
            E: 表示基础语言模型每个token输出的向量维度大小
            N批次大小，也就是序列样本数量
            T表示序列长度
            N1表示每个序列样本中存在多少个实体，也就是实体数目(正实体和负实体)
            N2表示每个序列样本中存在多少个实体关系对，也就是关系数目(正关系和负关系)
        @param encodings: [N,T]
        @param context_masks: [N,T]
        @param entity_masks: [N,N1,T] 每个实体对应的token位置信息
        @param entity_sizes: [N,N1] 每个实体对应的长度信息
        @param relations: [N,N2,2]
        @param rel_masks: [N,N2,T]
        """
        h = self._fetch_bert_embedding(encodings, context_masks)

        batch_size = encodings.shape[0]  # 获取批次大小

        # classify entities 获取实体的预测置信度以及每个实体span范围的特征信息(bert输出的)
        size_embeddings = self.size_embeddings(entity_sizes)  # [N,N1,size_embedding]
        # entity_clf: [N,N1,entity_type_num]; entity_spans_pool:[N,N1,E] 实体分类
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # classify relations 关系置信度的预测
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage 由于关系置信度的计算比较消耗内存，所以分段执行
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates [N,min(N2,_max_pairs),rel_types_num]
            chunk_rel_logits = self._classify_relations(
                entity_spans_pool,  # [N,N1,E] 每个实体span对应的特征向量
                size_embeddings,  # [N,N1,size_embedding] 每个实体span长度对应的特征向量
                relations,  # 候选的关系元数据，保存的是关系对应的两个实体下标 [N,N2,2]
                rel_masks,  # 候选的关系元数据，保存的是关系对中两个实体之间的mask信息 [N,N2,T]
                h_large,  # bert的原始输出，做了一个复制扩展的对象 [N,min(N2,_max_pairs),T,E]
                i
            )
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor,
                           support_entity_types=None):
        # get contextualized token embeddings from last transformer layer 获取bert结构的输出 [N, T, E]
        # context_masks = context_masks.float()
        # h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        h = self._fetch_bert_embedding(encodings, context_masks)

        batch_size = encodings.shape[0]  # 批次大小
        ctx_size = context_masks.shape[-1]  # 序列长度大小

        # classify entities 获取实体的预测置信度以及每个候选实体span范围的特征信息(bert输出的)
        size_embeddings = self.size_embeddings(entity_sizes)  # [N,N1,size_embedding]
        # entity_clf: 置信度 [N,N1,entity_type_num]; entity_spans_pool: 实体对应的特征向量 [N,N1,E]
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        # 基于命名实体识别预测出来的每个候选实体对应的置信度信息 --> 预测的实体id/序号 --> 将预测的实体id进行组合形成候选的实体关系对以及对应的上下文mask信息
        relations, rel_masks, rel_sample_masks = self._filter_spans(
            entity_clf,  # [N,N1,entity_type_num] 每个样本、每个实体属于各个实体类别的置信度
            entity_spans,  # [N,N1,2] 每个样本、每个实体对应的span片段的index起始位置
            entity_sample_masks,  # [N,N1] 每个样本对应的实体mask信息
            ctx_size,  # 序列长度T
            support_entity_types=support_entity_types
        )

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage 防止过大，进行分块计算
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates 调用全连接结构获取每个关系对应预测的关系类别置信度
            chunk_rel_logits = self._classify_relations(
                entity_spans_pool,  # [N,N1,E] 每个实体span对应的特征向量
                size_embeddings,  # [N,N1,size_embedding] 每个实体span长度对应的特征向量
                relations,  # 候选的关系元数据，保存的是关系对应的两个实体下标 [N,N2,2]
                rel_masks,  # 候选的关系元数据，保存的是关系对中两个实体之间的mask信息 [N,N2,T]
                h_large,  # bert的原始输出，做了一个复制扩展的对象 [N,min(N2,_max_pairs),T,E]
                i
            )
            # apply sigmoid sigmoid求概率值
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax 获取每个样本、每个实体对应的预测实体类别概率
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        # max pool entity candidate spans 提取bert输出的实体范围内的特征的最大值作为当前实体的最终特征
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)  # [N,N2,T,1] 将所有不是实体的位置设置为接近负无穷大的数，实体位置设置为0
        # [N,N2,T,E] 实体位置就是bert原始的输出值，其它位置为负无穷大的值
        # [N,N2,T,1] + [N,1,T,128].repeat(1,N2,1,1) --> [N,N2,T,1] + [N,N2,T,128]
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]  # 针对每个样本的每个实体获取T个token中的最大值 [N,N2,E]

        # get cls token as candidate context representation 通过[CLS]标记位获取整个bert输出的序列特征信息/上下文信息 [N,E]
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations isize_embeddingncluding context, max pooled span and size embedding
        # 合并三个最终的实体决策特征 [N,N2,E+E+size_embedding_dim]：基于[CLS]提取的序列上下文特征、实体token范围的bert输出特征、实体长度大小转换/embedding映射特征
        entity_repr = torch.cat([
            entity_ctx.unsqueeze(1).repeat(1, entity_masks.shape[1], 1), entity_spans_pool, size_embeddings],
            dim=2
        )
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates 获取得到每个实体范围span属于各个实体类别的置信度
        entity_clf = self.entity_classifier(entity_repr)  # [N,N2,num_entity]

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]  # 获取批次大小，也就是N

        # create chunks if necessary 分段
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations 获取关系实体对中的每个实体对应的实体特征
        entity_pairs = util.batch_index(entity_spans, relations)  # [N,N1,E] + pair_index --> [N,N2,2,E]
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)  # [N,N2,2,E] -> [N,N2,2*E]

        # get corresponding size embeddings 获取关系实体对中的每个实体对应的长度特征向量
        size_pair_embeddings = util.batch_index(size_embeddings, relations)  # [N,N1,SE] + pair_index --> [N,N2,2,SE]
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)  # [N,N2,2*SE]

        # relation context (context between entity candidate pair) 获取两个实体之间的span特征作为关系的context特征向量
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)  # [N,N2,T,1] 将context位置设为0，其它位置设为-1e30
        rel_ctx = m + h  # [N,N2,T,1] + [N,N2,T,E] --> [N,N2,T,E]
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]  # 在T个时刻中选择最大的特征向量
        # set the context vector of neighboring or adjacent entity candidates to zero 针对关系负样本来讲，不存在上下文的，所以max选出来的就是负无穷大，所以需要重置为0
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings 上下文特征 + 两个实体特征 + 两个实体长度特征
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size, support_entity_types=None):
        batch_size = entity_clf.shape[0]  # 批次样本大小
        # 获取每个候选实体在各个类别上的置信度最大的位置索引，也就是预测类别下标 + mask转换(填充对应位置也转换为0)
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)  # 获取预测实体类别不为0的候选span片段下标 --> 预测结果为实体的值
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()  # 获取所有预测实体对应的token索引范围
            non_zero_indices = non_zero_indices.tolist()  #

            # create relations and masks 创建候选关系实体组 -- 遍历所有实体与实体之间，组合候选关系集
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        if support_entity_types is not None:
                            # 1. 获取i1这个实体对应的实体类别
                            i1_type_id = entity_logits_max[i][i1].item()
                            # 2. 获取i2这个实体对应的实体类别
                            i2_type_id = entity_logits_max[i][i2].item()
                            # 3. 判断(i1,i2)这个类别组合是不是正常的关系实体组合
                            type_ids = (i1_type_id, i2_type_id)
                            # 4. 如果不是正常组合，直接过滤掉；如果是，那就正常执行
                            if type_ids not in support_entity_types:
                                continue  # 如果不在列表中，直接当前组合不考虑
                        rels.append((i1, i2))  # 关系对应的两个实体的下标
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))  # 构建关系对应的context-mask
                        sample_masks.append(1)

                #         # TODO: 测试，用于控制产生的候选关系组的数量
                #         if i2 > 5:
                #             break
                # # TODO: 测试，用于控制产生的候选关系组的数量
                # if i1 > 5:
                #     break

            if not rels:
                # case: no more than two spans classified as entities 针对当前样本没有实体正样本的预测
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack 填充合并数据
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)  # 训练的前向过程
        else:
            return self._forward_inference(*args, **kwargs)  # 推理的前向过程


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]


def load_model(args, tokenizer, input_reader):
    model_class = get_model(args.model_type)  # 获取class对象

    config = BertConfig.from_pretrained(args.model_path, cache_dir=args.cache_path)  # 构建bert config对象
    util.check_version(config, model_class, args.model_path)

    # noinspection SpellCheckingInspection
    config.spert_version = model_class.VERSION
    model = model_class.from_pretrained(args.model_path,  # 模型文件路径
                                        config=config,  # bert模型配置信息 BertConfig
                                        # SpERT model parameters
                                        cls_token=tokenizer.convert_tokens_to_ids('[CLS]'),  # cls token id
                                        relation_types=input_reader.relation_type_count - 1,  # 关系类别的数量(去除no relation)
                                        entity_types=input_reader.entity_type_count,  # 实体类别的数量
                                        max_pairs=args.max_pairs,
                                        prop_drop=args.prop_drop,
                                        size_embedding=args.size_embedding,
                                        freeze_transformer=args.freeze_transformer,
                                        cache_dir=args.cache_path)

    return model
