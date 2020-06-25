import random

import torch

from spert import util

# 对一个句子，创建对应的样本
def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    encodings = doc.encoding # 句子的encoding，即token的id
    token_count = len(doc.tokens) # 句子长度，是指未分词的句子长度
    context_size = len(encodings) # 分词后的encoding长度

    # 正样本
    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span) # 实体span的起始，encoding后的位置
        pos_entity_types.append(e.entity_type.index) # 实体的类型
        pos_entity_masks.append(create_entity_mask(*e.span, context_size)) # 获得一个和句子长度一致的bool tensor,实体span对应的位置为1
        pos_entity_sizes.append(len(e.tokens)) #实体长度，原始的jtokens长度

    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for rel in doc.relations:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span #s1和s2代表头尾实体的span的encoding位置
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2))) # 实体正样本中，取得头尾实体对应的索引index
        pos_rel_spans.append((s1, s2)) # 头尾实体的encoding起始
        pos_rel_types.append(rel.relation_type) # 关系类型
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

    # 实体的负样本
    # negative entities 
    # 取所有可能的span
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1): # 1~（10）+1，应该是用来计算负样本的span的长度
        for i in range(0, (token_count - size) + 1): # 从句子第一个字符开始，按照分词前的jtokens取，-size是为了防止out of index
            span = doc.tokens[i:i + size].span # 逐个取span
            if span not in pos_entity_spans: # 如果不包括在正样本中，则放入负样本list里，并记录对应的span长度
                neg_entity_spans.append(span) # encoding的起始
                neg_entity_sizes.append(size) # 长度（原始token)

    # sample negative entities 
    # 采样
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count)) #随机采样
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], []) #实体的位置，长度

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans] # 求实体的mask
    neg_entity_types = [0] * len(neg_entity_spans) # 实体类型

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans): # 实体正样本，头实体
        for i2, s2 in enumerate(pos_entity_spans): # 实体正样本，尾实体
            rev = (s2, s1) # 取反
            # 如果反关系rev (s2,s1)是正样本，且对应的关系是对称的，则rev_symmetric = rev
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric # 对称

            # do not add as negative relation sample:
            # neg. relations from an entity to itself (自指向)
            # entity pairs that are related according to gt 
            # entity pairs whose reverse exists as a symmetric relation in gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [0] * len(neg_rel_spans)

    # merge （合并正负样本作为整体的样本）
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long) # 句子encoding

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool) # 与encoding长度相等的mask

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long) # tensor,1维，长度=实体样本数,记录实体类型
        entity_masks = torch.stack(entity_masks) # tensor,2维，样本数*句子长度
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long) # tensor,1维，长度=实体样本数，记录实体长度（原始token)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool) # tensor,1维，长度=实体样本数，全为true
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)
    rel_types_onehot = rel_types_onehot[:, 1:]  # all zeros for 'none' relation

    # 输出是一个dict
    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types_onehot,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
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

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks)

#span的start
#span的end
#原句分词后的长度
def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1] #取两个实体之间文本的start
    end = s2[0] if s1[1] < s2[0] else s1[0] # end
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch): #list,list每个元素为Dataset里__getitem__()返回的样本
    padded_batch = dict()
    keys = batch[0].keys() # sample的dict

    for key in keys:
        samples = [s[key] for s in batch] # 逐个key构建list，对整个batch，list长度为batch中样本数

        if not batch[0][key].shape: # 如果对应的key没有shape
            padded_batch[key] = torch.stack(samples) # 堆叠
        else: # 有shape，是个tensor，需要padding
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
