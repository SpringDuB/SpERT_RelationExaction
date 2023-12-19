import random

import torch

from spert import util


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    encodings = doc.encoding  # token转换的id，也就是常规所说的x输入
    token_count = len(doc.tokens)  # 实际token长度
    context_size = len(encodings)  # token id的长度

    # create tensors --> 针对x
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)  # 原始文本token id对象
    # masking of tokens 关于token输入的mask
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # ==== 针对实体属性进行构建 ====
    # positive entities 实体正样本的构建
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)  # 当前实体在encoding中的起始、结束下标
        pos_entity_types.append(e.entity_type.index)  # 当前实体类别id
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))  # 当前实体对应的mask向量/数组
        pos_entity_sizes.append(len(e.tokens))  # 当前实体的长度大小

    # negative entities 实体负样本的构建(遍历所有可能)
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):  # size就是负样本实体的长度大小
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span  # 候选token span片段对象
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # sample negative entities 随机抽取负样本实体对象
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]  # 针对负样本实体创建对应的mask
    neg_entity_types = [0] * len(neg_entity_spans)  # type类型全为0 -- 负样本实体对象

    # merge 正负样本合并
    entity_types = pos_entity_types + neg_entity_types  # list[int] 每个实体对应的类别id
    entity_masks = pos_entity_masks + neg_entity_masks  # list[tensor[bool]] 每个实体对应的位置mask信息
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)  # list[int] 每个实体的token实际大小

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)  # 实体数量的mask信息
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    # ==== 关系数据的构建 ====
    # positive relations 关系正样本构建
    # collect relations between entity pairs 按照主体-客体对进行数据区分(同一组实体可能对应多种关系的情况)
    entity_pair_relations = dict()
    for rel in doc.relations:
        pair = (rel.head_entity, rel.tail_entity)  # 构建从主体-客体的实体对
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # build positive relation samples 关系正样本构建
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair  # 主体、客体
        s1, s2 = head_entity.span, tail_entity.span  # 主体的encoding下标范围、客体的encoding下标范围
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))  # 主体和客体的实体下标
        pos_rel_spans.append((s1, s2))

        pair_rel_types = [r.relation_type.index for r in rels]  # 主体和客体之间的关系类别id列表
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]  # 将类别id进行multi onehot转换
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))  # 创建关系对应的mask对象(两个实体之间的所有位置设1)

    # negative relations 关系负样本的构建 -- 遍历所有的正样本实体，来构建负关系对象
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []
    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    # sample negative relations 负样本关系抽样
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    # noinspection PyTypeChecker
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [(0,) * (rel_type_count - 1)] * len(neg_rel_spans)

    # 关系正负样本合并
    rels = pos_rels + neg_rels  # list(tuple(int,int)) 每个关系对应的主体&客体对应在entity_types中的下标
    rel_types = pos_rel_types + neg_rel_types  # list[int] 每个关系对应的关系类别id(做了multi onehot操作)
    rel_masks = pos_rel_masks + neg_rel_masks  # list(tensor(bool)) 每个关系对应的两个实体之间的位置为true，其它位置为false的一个mask信息

    assert len(rels) == len(rel_masks) == len(rel_types)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count - 1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    """
        T: 处理后的token序列长度
        N1: 表示实体正样本和负样本的总数量
        N2: 表示关系正样本和负样本的总数量
        REL_N: 表示关系的类别数量
        encodings: 原始的token进行wordsprice处理之后的对应id列表，也就是理解成模型的原始输入x，eg: [101, 2342, ...., 234, 102], shape: [T]
        context_masks: 恒定为1/True的一个列表, shape: [T]
        entity_masks: 表示实体正样本和负样本的mask列表，每个实体一个mask列表；
            如果某个token属于当前实体，那么该token对应位置为1/True，否则为False, shape: [N1,T]
        entity_sizes: 每个实体(正样本和负样本)的长度，shape: [N1]
        entity_types：每个实体(正样本和负样本)的类别id，shape: [N1]
        entity_sample_masks: 全为1, shape: [N1]
        rels: 给定实体对的索引下标,下标取值范围:[0,N1)，shape: [N2,2]
        rel_masks：表示任意两个实体之间的关系mask列表，
            每个实体对存在一个mask；其中两个实体之间的token对应位置为1/True，其它位置为False，shape:[N2,T]
        rel_types: 给定两个实体对之间的关系类别id进行multi onehot之后的值， eg: [0,1,0,0,1]表示这两个实体之间的关系存在两种:关系2和关系5，shape:[N2,REL_N]
        rel_sample_masks：全为1，shape: [N2]
    """
    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding  # x, token id列表
    token_count = len(doc.tokens)  # 实际token的数量
    context_size = len(encodings)  # 处理后的token数量，也就是encodings的长度

    # create tensors
    # token indices x
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens x mask 样本对应的mask信息，在组成成batch的时候会进行填充
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # create entity candidates 实体候选集的构建
    entity_spans = []  # 保存的是每个实体的起始下标和结束下标
    entity_masks = []  # 每个候选实体在每个token位置的mask值
    entity_sizes = []  # 每个候选实体的长度大小

    # 遍历每个位置，获取对应的entity span，长度从1到max_span_size
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span  # 起始下标(包含)，结尾下标(不包含)
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))  # 对应范围为1/True，其它位置为0/False
            entity_sizes.append(size)  # span片段的长度大小

    # entities 将实体候选集转换为tensor对象
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(
        # encodings： 原始的token进行wordsprice处理之后的对应id列表，也就是理解成模型的原始输入x，eg: [101, 2342, ...., 234, 102], shape: [T]
        encodings=encodings,
        # context_masks：恒定为1/True的一个列表, shape: [T]
        context_masks=context_masks,
        # entity_masks: 表示实体的mask列表，每个实体一个mask列表；
        #             如果某个token属于当前实体，那么该token对应位置为1/True，否则为False, shape: [N1,T]
        entity_masks=entity_masks,
        # entity_sizes: 每个实体的长度，shape: [N1]
        entity_sizes=entity_sizes,
        # entity_spans: 每个实体对应的起始和结束的token index，[N1,2]
        entity_spans=entity_spans,
        # entity_sample_masks: 全为1, shape: [N1]
        entity_sample_masks=entity_sample_masks
    )


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    # 仅考虑s1和s2一个在前，一个在后的情况，不考虑两个实体交叉的情况; 标注两个实体之间的范围
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack(samples)

    return padded_batch
