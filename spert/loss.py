from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion  # 关系的损失计算方法
        self._entity_criterion = entity_criterion  # 实体的损失计算方法
        self._model = model  # 模型对象
        self._optimizer = optimizer  # 优化器
        self._scheduler = scheduler  # 学习率变化方法
        self._max_grad_norm = max_grad_norm  # 最大梯度

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss 计算实体识别的损失
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])  # [N,N1,e_num] -> [N*N1,e_num]
        entity_types = entity_types.view(-1)  # [N,N1] --> [N*N1]
        entity_sample_masks = entity_sample_masks.view(-1).float()  # [N,N1] --> [N*N1] 实际值为1，填充位置为0

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss 关系识别的损失
        rel_sample_masks = rel_sample_masks.view(-1).float()  # [N,N2] --> [N*N2]
        rel_count = rel_sample_masks.sum()  # 获取总的关系实体对的数量(除了填充之外的)

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])  # [N,N2,rel_num] -> [N*N2,rel_num]
            rel_types = rel_types.view(-1, rel_types.shape[-1])  # [N,N2,rel_num] -> [N*N2,rel_num]

            rel_loss = self._rel_criterion(rel_logits, rel_types)  # sigmoid交叉熵损失 [N*N2,rel_num]
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss 损失联合
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        # 参数更新
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()  # 参数更新
        self._scheduler.step()  # 学习率更新
        self._model.zero_grad()  # 梯度重置为0
        return train_loss.item()
