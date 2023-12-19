import json
from typing import Tuple

import torch

from spert import util
from spert.input_reader import BaseInputReader


def convert_predictions(batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                        batch_rels: torch.tensor, batch: dict, rel_filter_threshold: float,
                        input_reader: BaseInputReader, no_overlapping: bool = False):
    # get maximum activation (index of predicted entity type)
    batch_entity_types = batch_entity_clf.argmax(dim=-1)  # 选择预测概率最大的类别作为每个实体对应的预测实体类别id
    # apply entity sample mask
    batch_entity_types *= batch['entity_sample_masks'].long()  # 将所有填充实体对应类别重置为0

    # apply threshold to relations 选择关系预测中概率低于阈值的概率，重置为0
    batch_rel_clf[batch_rel_clf < rel_filter_threshold] = 0

    batch_pred_entities = []
    batch_pred_relations = []

    for i in range(batch_rel_clf.shape[0]):  # 针对每个样本进行遍历
        # get model predictions for sample
        entity_types = batch_entity_types[i]  # 获取当前样本所有实体的预测类别id
        entity_spans = batch['entity_spans'][i]  # 获取当前样本每个实体的token的范围
        entity_clf = batch_entity_clf[i]  # 获取当前样本所有实体的预测概率值
        rel_clf = batch_rel_clf[i]  # 获取当前样本所有关系组的预测概率值
        rels = batch_rels[i]  # 获取当前样本所有关系的实体组序号id

        # convert predicted entities 提取出预测的实体列表，包括每个实体对应的(起始下标，结束下标，类别对象，预测概率值)
        sample_pred_entities = _convert_pred_entities(entity_types, entity_spans,
                                                      entity_clf, input_reader)

        # convert predicted relations 提取组合关系列表，包括每个关系对应的()
        sample_pred_relations = _convert_pred_relations(rel_clf, rels,
                                                        entity_types, entity_spans, input_reader)

        if no_overlapping:
            sample_pred_entities, sample_pred_relations = remove_overlapping(sample_pred_entities,
                                                                             sample_pred_relations)

        batch_pred_entities.append(sample_pred_entities)
        batch_pred_relations.append(sample_pred_relations)

    return batch_pred_entities, batch_pred_relations


def _convert_pred_entities(entity_types: torch.tensor, entity_spans: torch.tensor,
                           entity_scores: torch.tensor, input_reader: BaseInputReader):
    # get entities that are not classified as 'None'
    valid_entity_indices = entity_types.nonzero().view(-1)  # 获取预测类别中所有非0类别对应的下标
    pred_entity_types = entity_types[valid_entity_indices]  # 得到每个候选实体对应的预测实体类别id
    pred_entity_spans = entity_spans[valid_entity_indices]  # 得到每个候选实体对应的span范围
    pred_entity_scores = torch.gather(entity_scores[valid_entity_indices], 1,
                                      pred_entity_types.unsqueeze(1)).view(-1)

    # convert to tuples (start, end, type, score) # 针对每个实体进行处理
    converted_preds = []
    for i in range(pred_entity_types.shape[0]):
        label_idx = pred_entity_types[i].item()  # 类别的标签id
        entity_type = input_reader.get_entity_type(label_idx)  # 类别对象

        start, end = pred_entity_spans[i].tolist()  # 实体对应的范围
        score = pred_entity_scores[i].item()  # 预测对应的概率

        converted_pred = (start, end, entity_type, score)
        converted_preds.append(converted_pred)

    return converted_preds


def _convert_pred_relations(rel_clf: torch.tensor, rels: torch.tensor,
                            entity_types: torch.tensor, entity_spans: torch.tensor, input_reader: BaseInputReader):
    rel_class_count = rel_clf.shape[1]  # 关系的类别数量:rel_types_num
    rel_clf = rel_clf.view(-1)  # [N2,rel_types_num] --> [N2*rel_types_num]

    # get predicted relation labels and corresponding entity pairs
    rel_nonzero = rel_clf.nonzero().view(-1)  # 获取所有关系实体对所属关系类别概率中大于0的索引下标
    pred_rel_scores = rel_clf[rel_nonzero]

    pred_rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1) 获取所有非0位置对应的预测类别id
    valid_rel_indices = rel_nonzero // rel_class_count  # 获取关系实体实际的索引id
    valid_rels = rels[valid_rel_indices]  # 提取出最终的预测关系实体对

    # get masks of entities in relation 提取关系实体对中，两个实体的索引范围 [N2,2,2]
    pred_rel_entity_spans = entity_spans[valid_rels].long()

    # get predicted entity types 提取关系实体对中，主体和客体两个实体的预测类别id [N2,2]
    pred_rel_entity_types = torch.zeros([valid_rels.shape[0], 2])
    if valid_rels.shape[0] != 0:
        pred_rel_entity_types = torch.stack([entity_types[valid_rels[j]] for j in range(valid_rels.shape[0])])

    # convert to tuples ((head start, head end, head type), (tail start, tail end, tail type), rel type, score))
    converted_rels = []
    check = set()

    for i in range(pred_rel_types.shape[0]):
        label_idx = pred_rel_types[i].item()  # 获取当前关系的预测类别id
        pred_rel_type = input_reader.get_relation_type(label_idx)  # 关系类别id转换为具体的关系对象
        pred_head_type_idx, pred_tail_type_idx = pred_rel_entity_types[i][0].item(), pred_rel_entity_types[i][1].item()
        pred_head_type = input_reader.get_entity_type(pred_head_type_idx)  # 预测关系中主体对应的实体类别对象
        pred_tail_type = input_reader.get_entity_type(pred_tail_type_idx)  # 预测关系中客体对应的实体类别对象
        score = pred_rel_scores[i].item()  # 预测关系概率值

        spans = pred_rel_entity_spans[i]  # 主体和客体的span范围
        head_start, head_end = spans[0].tolist()
        tail_start, tail_end = spans[1].tolist()

        converted_rel = ((head_start, head_end, pred_head_type),
                         (tail_start, tail_end, pred_tail_type), pred_rel_type)
        converted_rel = _adjust_rel(converted_rel)

        if converted_rel not in check:
            check.add(converted_rel)
            converted_rels.append(tuple(list(converted_rel) + [score]))

    return converted_rels


def remove_overlapping(entities, relations):
    non_overlapping_entities = []
    non_overlapping_relations = []

    for entity in entities:
        if not _is_overlapping(entity, entities):
            non_overlapping_entities.append(entity)

    for rel in relations:
        e1, e2 = rel[0], rel[1]
        if not _check_overlap(e1, e2):
            non_overlapping_relations.append(rel)

    return non_overlapping_entities, non_overlapping_relations


def _is_overlapping(e1, entities):
    for e2 in entities:
        if _check_overlap(e1, e2):
            return True

    return False


def _check_overlap(e1, e2):
    if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
        return False
    else:
        return True


def _adjust_rel(rel: Tuple):
    adjusted_rel = rel
    if rel[-1].symmetric:
        head, tail = rel[:2]
        if tail[0] < head[0]:
            adjusted_rel = tail, head, rel[-1]

    return adjusted_rel


def parse_predictions(documents, pred_entities, pred_relations):
    predictions = []
    documents = documents[:len(pred_entities)]  # 本来没有这行代码，这里加是因为代码中进行了break的一些操作导致的

    for i, doc in enumerate(documents):
        tokens = doc.tokens
        sample_pred_entities = pred_entities[i]
        sample_pred_relations = pred_relations[i]

        # convert entities
        converted_entities = []
        for entity in sample_pred_entities:
            entity_span = entity[:2]
            span_tokens = util.get_span_tokens(tokens, entity_span)
            entity_type = entity[2].identifier
            converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
            converted_entities.append(converted_entity)
        converted_entities = sorted(converted_entities, key=lambda e: e['start'])

        # convert relations
        converted_relations = []
        for relation in sample_pred_relations:
            head, tail = relation[:2]
            head_span, head_type = head[:2], head[2].identifier
            tail_span, tail_type = tail[:2], tail[2].identifier
            head_span_tokens = util.get_span_tokens(tokens, head_span)
            tail_span_tokens = util.get_span_tokens(tokens, tail_span)
            relation_type = relation[2].identifier

            converted_head = dict(type=head_type, start=head_span_tokens[0].index,
                                  end=head_span_tokens[-1].index + 1)
            converted_tail = dict(type=tail_type, start=tail_span_tokens[0].index,
                                  end=tail_span_tokens[-1].index + 1)

            head_idx = converted_entities.index(converted_head)
            tail_idx = converted_entities.index(converted_tail)

            converted_relation = dict(type=relation_type, head=head_idx, tail=tail_idx)
            converted_relations.append(converted_relation)
        converted_relations = sorted(converted_relations, key=lambda r: r['head'])

        doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=converted_entities,
                               relations=converted_relations)
        predictions.append(doc_predictions)
    return predictions


def store_predictions(documents, pred_entities, pred_relations, store_path):
    predictions = parse_predictions(documents, pred_entities, pred_relations)
    # store as json
    with open(store_path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)
    return predictions
