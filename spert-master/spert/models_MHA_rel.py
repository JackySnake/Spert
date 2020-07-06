import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util

import numpy as np
import math

# h 隐变量表示
# x encoding 索引
def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1] #隐藏表示的长度

    token_h = h.view(-1, emb_size) # view,重整tensor的维度 [batch_size*sentence_length, embedding_size]
    flat = x.contiguous().view(-1) # contiguous()，返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor；维度变换之后，保持tensor连续
                                   # 直接将encoidngs展平，1维，长度为[batch size*句子长度]
                                   # 保证了flat.shape[0] == token_h.shape[0]

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :] # 按顺序取得某个token的表示, 如[CLS] （这里类似的前提都是建立在各个张量都是按序一一对应的前提下）

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1' 
    # config: 模型设置
    # cls_token
    # relation_types
    # entity_types
    # size_embedding: embedding大小
    # prop_drop: dropout
    # freeze_transformer:是否冻结transformer
    # max_pairs: 最大的实体对数目
    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers 

        # 建模entity
        # modify on imp start
        self.lstm_layer = nn.LSTM(input_size=config.hidden_size,
                                  hidden_size=config.hidden_size // 2,
                                  num_layers=2,
                                  #dropout=0.5,
                                  bidirectional=True)
        # self.lstm_layer = nn.LSTM(input_size=config.hidden_size,
        #                           hidden_size=config.hidden_size // 2,
        #                           num_layers=1,
        #                           #dropout=0.5,
        #                           bidirectional=True)                                  
        # modify on imp end                                  

        # multiheadAtt 关系抽取
        self.multiheadAttlayer = MultiHeadedAttention(12, config.hidden_size, 0.01) # 关系抽取for multiHeadAtt add

        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types) # 关系分类
        # self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types) # entity分类
        self.entity_classifier = nn.Linear(config.hidden_size + size_embedding, entity_types) # entity分类,不使用[CLS]全局表示
        self.size_embeddings = nn.Embedding(100, size_embedding) # width embedding
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    #
    # encodings batch size*句子长度
    # context_masks
    # entity_masks
    # entity_sizes
    # relations
    # rel_masks
    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0] # 预训练的输出表示

        batch_size = encodings.shape[0] # batch_size大小

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings) # batchsize*span数*句子长度

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1) # [batchsize, relation_sample_num, 句子长度, embedding_size]
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans, 
                                                                    entity_sample_masks, ctx_size)
        
        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        # entity_masks shape: batch大小*大小候选span数*句子长, span 对应的 句子长度 的向量，表示了实体在encoding中的位置
        # max pool entity candidate spans
        # mod on imp start
        batch_size = entity_masks.shape[0] # batch 大小
        span_num = entity_masks.shape[1] # span的长度
        embedding_size = h.shape[-1] # 隐蔽表示embedding 大小

        #### 句子长度的LSTM输入    
        # # entity_masks.unsqueeze(-1) 扩展维度, batch大小*大小候选span数*句子长*1（扩展维度）
        # # entity_masks.unsqueeze(-1) == 0 如果相等，对应的元素设为true。在这里，相当于是为了将entity_masks.unsqueeze(-1)的0，1互换, 即0代表span的位置
        # # (entity_masks.unsqueeze(-1) == 0).float() * (-1e30) 相乘，span位置元素为0，其他位置为-1e30
        # m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # # m是转换后的向量
        # # h.unsqueeze(1) 扩展维度，batch大小*1（扩展维度）*句子长度*embedding大小
        # # h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1) 在第二个维度方向复制，batch大小*大小候选的span数*句子长度*embedding_size
        # entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # # 全句表示，实体，句子长度*batch大小*大小候选的span数*embedding_size
        # entity_whole_sentence = entity_spans_pool.permute(2, 0, 1, 3) #
        # # 句子长度 *  (batch_size * span_num) * embedding_size
        # entity_whole_sentence = entity_whole_sentence.reshape([-1, batch_size * span_num, embedding_size])

        #### span长度的LSTM输入，变长处理
        # 获得regions的表示，list[batch_size*span_num, length, embedding_size]
        regions = []
        for batch_i in range(0, batch_size): # batch
            sent= []
            for span_i in range(0, span_num): # span
                span_repr = []
                for token_i in range(0, entity_masks.shape[-1]): # sent
                    if  entity_masks[batch_i][span_i][token_i] == True:
                        span_repr.append(h[batch_i][token_i]) # 获取对应region的h表示
                if len(span_repr) != 0:
                    regions.append(torch.stack(span_repr,dim=0))
                else:
                    regions.append(torch.zeros(1,embedding_size))

        pack_regions = torch.nn.utils.rnn.pack_sequence(regions, enforce_sorted=False) # span regions 的压缩，保证LSTM处理变长序列

        self.lstm_layer.flatten_parameters()

        # output, (final_hidden_state, final_cell_state) = self.lstm_layer(entity_whole_sentence)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pack_regions)
        #
        final_hidden_state = final_hidden_state.reshape([final_hidden_state.shape[0], batch_size, span_num, -1]) # 从句子级别的batch_size,reshap, [layer*biredirect, batch_size, span_num, embedding_size]

        # concat [h-2, h-1]
        entity_spans_pool = torch.cat([final_hidden_state[-2], final_hidden_state[-1]], dim=2)

        # mod on imp end
        # get cls token as candidate context representation
        # entity_ctx = get_token(h, encodings, self._cls_token) #不使用[CLS全局表示]

        # create candidate representations including context, max pooled span and size embedding
        # entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
        #                          entity_spans_pool, size_embeddings], dim=2)
        entity_repr = torch.cat([entity_spans_pool, size_embeddings], dim=2) #不使用[CLS全局表示]
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]
        embedding_size = entity_spans.shape[2] # 关系抽取for multiHeadAtt add

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        # entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)  ## 关系抽取for multiHeadAtt del

        # 关系抽取for multiHeadAtt add start
        entity_pairs_whole_sentence = entity_pairs.reshape([batch_size, -1, embedding_size]) # 头尾embedding展开 [batchsize,relation_sample_num*2, embedding_size]
        h = h.reshape([batch_size, -1, embedding_size]) # 按照样本数展开[batchsize, relation_sample,embedding_size]
        whole_sentences = self.multiheadAttlayer.forward(entity_pairs_whole_sentence, h, h)
        whole_sentences = whole_sentences.reshape([batch_size, -1, embedding_size]).reshape(batch_size, entity_pairs.shape[1], -1)

        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # 关系抽取for multiHeadAtt add end

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # 关系抽取for multiHeadAtt del start
        # # relation context (context between entity candidate pair)
        # # mask non entity candidate tokens
        # m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        # rel_ctx = m + h
        # # max pooling
        # rel_ctx = rel_ctx.max(dim=2)[0]
        # # set the context vector of neighboring or adjacent entity candidates to zero
        # rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0
        # 关系抽取for multiHeadAtt del end

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        # rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = torch.cat([rel_ctx, entity_pairs], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

# 关系抽取for multiHeadAtt add start
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x).squeeze(1)  # (batch, time1, d_model)
# 关系抽取for multiHeadAtt add end
# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
