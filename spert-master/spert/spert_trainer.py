import argparse
import math
import os

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from spert import models
from spert import sampling
from spert import util
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss
from tqdm import tqdm
from spert.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # 分词器
        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)
        # 预测结果
        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        # types_path: task的三元组的schema
        # self._tokenizer: 分词器
        # args.neg_entity_count: 实体负样本数目
        # args.neg_relation_count: 关系负样本数目
        # args.max_span_size: span最大长度
        # self._logger 
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path}) # 读取数据集，此时还未进行负采样, 保存到input_reader.datasets
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label) # 获取训练集
        train_sample_count = train_dataset.document_count # 训练集样本数
        updates_epoch = train_sample_count // args.train_batch_size # 计算每个epoch 迭代的更新参数的 次数
        updates_total = updates_epoch * args.epochs # 计算总的更新参数次数

        # # 分三个训练阶段，entity，relation，joint
        # milestone = args.epochs // 3
        # phase1_epoch = 0 # entity 
        # phase2_epoch = milestone # relation
        # phase3_epoch = milestone * 2 # joint
        # # updates_epoch = train_sample_count // args.train_batch_size # 计算每个epoch 迭代的更新参数的 次数
        # updates_total_phase1 = updates_epoch * (phase2_epoch-phase1_epoch)
        # updates_total_phase2 = updates_epoch * (phase3_epoch - phase2_epoch)
        # updates_total_phase3 = updates_epoch * (args.epochs - phase2_epoch)

        # 分两个训练阶段，entity，relation
        milestone = args.epochs // 2
        phase1_epoch = 0 # entity 
        phase2_epoch = milestone # relation
        # updates_epoch = train_sample_count // args.train_batch_size # 计算每个epoch 迭代的更新参数的 次数
        updates_total_phase1 = updates_epoch * (phase2_epoch-phase1_epoch)
        updates_total_phase2 = updates_epoch * (args.epochs - phase2_epoch)

        validation_dataset = input_reader.get_dataset(valid_label) # 验证集

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        #到这里读入的都是正样本，应该还没做负样本抽样
        # create model
        model_class = models.get_model(self.args.model_type) #获取模型

        # load model
        # 这里可能才是真正的实例化
        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)

        config.spert_version = model_class.VERSION
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # parallelize model
        # 并行化
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # 优化器
        # 设置了不更新偏差bias
        # 创建了AdamW优化器
        # create optimizer
        # optimizer_params = self._get_optimizer_params(model)
        # optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        
        # 模型不同部分使用不同的学习率
        # optimizer_params_diff_part = self._get_optimizer_params_on_diff_part_with_lr(model, entity_part_lr=0.001, relation_part_lr = 0.001)
        
        # optimizer_params_diff_part = self._get_optimizer_params_with_condition(model, entity_part_lr=0.001, relation_part_lr = 0.001)
        # optimizer = AdamW(optimizer_params_diff_part, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        
        # # phase 1 only entity
        # optimizer_params_diff_part_phase1 = self._get_optimizer_params_with_condition(model, entity_part_lr=0.001, relation_part_lr = 0, 
        #                                                                                 entity_bool = True, relaiton_bool = False)
        # optimizer_phase1 = AdamW(optimizer_params_diff_part_phase1, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # #phase 2 only rel
        # optimizer_params_diff_part_phase2 = self._get_optimizer_params_with_condition(model, entity_part_lr=0, relation_part_lr = 0.001,
        #                                                                                 entity_bool = False, relaiton_bool = True)
        # optimizer_phase2 = AdamW(optimizer_params_diff_part_phase2, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # #phase 3 joint entity and rel
        # optimizer_params_diff_part_phase3 = self._get_optimizer_params_with_condition(model, entity_part_lr=0.001, relation_part_lr = 0.001, 
        #                                                                                 entity_bool = True, relaiton_bool = True)
        # optimizer_phase3 = AdamW(optimizer_params_diff_part_phase3, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)

        # phase 1 only entity
        optimizer_params_diff_part_phase1 = self._get_optimizer_params_with_condition(model, entity_part_lr=0.001, relation_part_lr = 0, 
                                                                                        entity_bool = True, relaiton_bool = False)
        optimizer_phase1 = AdamW(optimizer_params_diff_part_phase1, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        #phase 2 only rel
        optimizer_params_diff_part_phase2 = self._get_optimizer_params_with_condition(model, entity_part_lr=args.lr, relation_part_lr = 0.001,
                                                                                        entity_bool = True, relaiton_bool = True)
        optimizer_phase2 = AdamW(optimizer_params_diff_part_phase2, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)

        # create scheduler
        # torch.optim.lr_scheduler接口,是一种学习率调整策略，其中提供了基于多种epoch数目调整学习率的方法,本方法使用线性的衰减策略。
        # 用一个Schedule把原始Optimizer装饰上，然后再输入一些相关参数，然后用这个Schedule做step()。
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
        #                                                          num_warmup_steps=args.lr_warmup * updates_total,
        #                                                          num_training_steps=updates_total)
        # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
        #                                                          num_warmup_steps=args.lr_warmup * updates_total,
        #                                                          num_training_steps=updates_total)
        
        # scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
        #                                                          num_warmup_steps=args.lr_warmup * updates_total,
        #                                                          num_training_steps=updates_total,
        #                                                          num_cycles = 2)
        
        # scheduler in three phase
        # scheduler_phase1 = transformers.get_cosine_schedule_with_warmup(optimizer_phase1,
        #                                                     num_warmup_steps=args.lr_warmup * updates_total_phase1,
        #                                                     num_training_steps=updates_total_phase1)
        # scheduler_phase2 = transformers.get_cosine_schedule_with_warmup(optimizer_phase2,
        #                                                     num_warmup_steps=args.lr_warmup * updates_total_phase2,
        #                                                     num_training_steps=updates_total_phase2)
        # scheduler_phase3 = transformers.get_cosine_schedule_with_warmup(optimizer_phase3,
        #                                                     num_warmup_steps=args.lr_warmup * updates_total_phase3,
        #                                                     num_training_steps=updates_total_phase3)

        scheduler_phase1 = transformers.get_cosine_schedule_with_warmup(optimizer_phase1,
                                                            num_warmup_steps=args.lr_warmup * updates_total_phase1,
                                                            num_training_steps=updates_total_phase1)
        scheduler_phase2 = transformers.get_cosine_schedule_with_warmup(optimizer_phase2,
                                                            num_warmup_steps=args.lr_warmup * updates_total_phase2,
                                                            num_training_steps=updates_total_phase2)

        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm) # 初始化Loss
        optimizer = optimizer_phase1
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler_phase1, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch 
            # 按阶段设置学习率,通过更改优化器的方式
            if epoch == phase1_epoch: # phase 1
                # optimizer = optimizer_params_diff_part_phase1
                pass
            elif epoch == phase2_epoch: # phase 1
                optimizer = optimizer_phase2
                compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler_phase2, args.max_grad_norm)
            # elif epoch == phase3_epoch: # phase 1
            #     optimizer = optimizer_phase3
            #     compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler_phase3, args.max_grad_norm)
            # 训练每一轮
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # add for eval in every X epoch 
            if (epoch +1) % 5 == 0 and epoch +1 != args.epochs:
                extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
                global_iteration = epoch+1 * updates_epoch
                self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                                optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                                include_iteration=False, name= str(epoch+1)+'_model')
                
                # self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

                self._logger.info("Logged in: %s" % self._log_path)
                self._logger.info("Saved in: %s" % self._save_path)
                self._summary_writer.close()
            # add for eval in every X epoch end
            
            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)

        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        # DataLoader加载数据，设定batchsize大小，并打乱数据，drop_last为true丢弃最后一个不足一个batch的数据，读取数据的进程num_workers,collate_fn是拼接多个样本为一个batch的方法
        # 负样本抽样在这里做的，spert.entities.Dataset的get_item
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size # 计算每个epoch的batch数目
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            # forward step
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'])
            # entity_logits:[batch_size, entity_num, entity_type], rel_logits:[batch_size,relation_num,relation_type]
            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks']) # 计算loss,更新参数，更新学习率

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               evaluate=True)
                entity_clf, rel_clf, rels = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()

    #分别保存了需要优化的参数
    #返回optimizer_params，数组长度为2
    #optimizer_params[0],dict,长度2，保存no_decay以外的参数
    #optimizer_params[1],dict,长度2，保存no_decay的参数
    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], #any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    #按预训练模型，实体分类，关系分类三个部分，分别保存了需要优化的参数
    #返回optimizer_params，数组长度为2
    #optimizer_params[0],dict,长度2，保存no_decay以外的参数
    #optimizer_params[1],dict,长度2，保存no_decay的参数
    def _get_optimizer_params_on_diff_part_with_lr(self, model, entity_part_lr, relation_part_lr):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        entity_part = ['lstm_layer', 'entity_global_map','entity_classifier','size_embeddings']
        relation_part = ['entity_head_linear', 'entity_tail_linear','rel_classifier']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' in n], # bert part, weight_decay
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0}, # bert part, no_weight_decay
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in entity_part) and not any(nd in n for nd in no_decay)], 'lr':entity_part_lr}, # entity_part, weight_decay
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in entity_part) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':entity_part_lr}, # entity_part, no_weight_decay
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in relation_part) and not any(nd in n for nd in no_decay)], 'lr':relation_part_lr}, # relation_part, weight_decay
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in relation_part) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':relation_part_lr} # relation_part, no_weight_decay
            ] # bert part, no weight_decay
    
    #按预训练模型，实体分类，关系分类三个部分，按照条件，分别设置要优化的参数和学习率
    #返回optimizer_params，数组长度为2
    #optimizer_params[0],dict,长度2，保存no_decay以外的参数
    #optimizer_params[1],dict,长度2，保存no_decay的参数
    def _get_optimizer_params_with_condition(self, model, entity_part_lr, relation_part_lr, entity_bool=True, relaiton_bool=True):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        entity_part = ['lstm_layer', 'entity_global_map','entity_classifier','size_embeddings']
        relation_part = ['entity_head_linear', 'entity_tail_linear','rel_classifier']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' in n], # bert part, weight_decay
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0}] # bert part, no_weight_decay
        
        if entity_bool: 
            # entity_part, weight_decay
            optimizer_params.append({'params': [p for n, p in param_optimizer if any(nd in n for nd in entity_part) and not any(nd in n for nd in no_decay)], 'lr':entity_part_lr})
            # entity_part, no_weight_decay
            optimizer_params.append({'params': [p for n, p in param_optimizer if any(nd in n for nd in entity_part) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':entity_part_lr})
        # else:
        #     pass

        if relaiton_bool:
            # relation_part, weight_decay
            optimizer_params.append({'params': [p for n, p in param_optimizer if any(nd in n for nd in relation_part) and not any(nd in n for nd in no_decay)], 'lr':relation_part_lr})
            # relation_part, no_weight_decay
            optimizer_params.append({'params': [p for n, p in param_optimizer if any(nd in n for nd in relation_part) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':relation_part_lr})
        # else:
        #     pass
        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
