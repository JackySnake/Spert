from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1]) # entity_logits: [batch_size*entity_num, entity_type]
        entity_types = entity_types.view(-1) # entity_types: [batch_size*entity_num] 标准答案
        entity_sample_masks = entity_sample_masks.view(-1).float() # bool-> float

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()
        # entity_loss * entity_sample_masks 相乘 排除掉mask的entity，因为不同的batch可能实体取样数目不一样
        # 此时的entity_loss是实体平均的loss

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float() # bool->float
        rel_count = rel_sample_masks.sum() # 求和

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step() # 更新参数
        self._scheduler.step() # 更新学习率
        self._model.zero_grad() # 置0
        return train_loss.item()
