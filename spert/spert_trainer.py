import argparse
import math
import os
from typing import Type

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from spert import models, prediction
from spert import sampling
from spert import util
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss
from tqdm import tqdm

from spert.models import load_model
from spert.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


# noinspection DuplicatedCode,PyTypeChecker
class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding 恢复token的解析器
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args  # 获取当前运行参数
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files 日志输出到文件中(构建)
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets 加载数据集（使用给定的数据加载器）
        input_reader = input_reader_cls(
            types_path,  # 实体和关系的元数据文件路径
            self._tokenizer,  # 文本token转换器
            args.neg_entity_count,  # 负实体样本数量，默认为100
            args.neg_relation_count,  # 负关系样本数量，默认为100
            args.max_span_size,  # 实体的token最大长度，默认为10
            self._logger  # 日志对象
        )
        train_dataset = input_reader.read(train_path, train_label)  # 数据对象构建
        validation_dataset = input_reader.read(valid_path, valid_label)  # 数据对象构建
        self._log_datasets(input_reader)  # 数据日志输出

        train_sample_count = train_dataset.document_count  # 总训练文本数量
        updates_epoch = train_sample_count // args.train_batch_size  # 一个epoch有多少个批次更新参数
        updates_total = updates_epoch * args.epochs  # 总的参数更新次数

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # load model 加载模型
        model = self._load_model(input_reader)
        self._logger.info(f"Model: \n{model}")

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)
        model.to(self._device)

        # create optimizer 优化器参数获取
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(
            optimizer_params, lr=args.lr, weight_decay=args.weight_decay,
            correct_bias=False, no_deprecation_warning=True
        )
        # create scheduler 学习率采用warmup方式变化：学习率先变大，再变小
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.lr_warmup * updates_total,
            num_training_steps=updates_total
        )
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # 关系分类采用sigmoid损失
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # 实体识别采用交叉熵损失/softmax损失
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)  # 这行代码执行很满

        # train 开始训练
        for epoch in range(args.epochs):
            # train epoch 当前epoch进行训练
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets 当参数final_eval为True的时候，表示仅最后一个epoch进行评估，否则每个epoch后均进行一次评估
            if not args.final_eval or (epoch == args.epochs - 1):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

        # save final model 保存
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self._args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model', safe_serialization=False)

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets 数据的构造
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        # load model 模型加载
        model = self._load_model(input_reader)
        model.to(self._device)

        # evaluate 评估模型效果
        self._eval(model, test_dataset, input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def predict(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args

        # read datasets 数据加载器
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size,
                                        spacy_model=args.spacy_model)
        dataset = input_reader.read(dataset_path, 'dataset')

        # 加载模型
        model = self._load_model(input_reader)
        model.to(self._device)

        # 推理预测
        predictions = self._predict(model, dataset, input_reader)

        # TODO: 結果已經保存到磁盤了，根據需要可以將磁盤的結果重新加載進來，然後保存到數據庫中
        print(predictions)

    def _load_model(self, input_reader):
        # 模型构造代码公用化处理
        return load_model(self._args, self._tokenizer, input_reader)

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader 创建Dataloader对象
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(
            dataset,  # 数据集
            batch_size=self._args.train_batch_size,  # 批次大小
            shuffle=True,
            drop_last=True,
            num_workers=self._args.sampling_processes,  # 加载数据的多进程数量
            collate_fn=sampling.collate_fn_padding  # 聚合函数
        )

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size  # 总批次大小
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)  # 设置数据的device信息

            # forward step 模型的前向执行，获取模型预测实体置信度和关系置信度
            # N: 表示批次样本数量；N1：表示候选的实体数量；N2：表示关系的候选数量；T表示序列长度；
            # entity_logits: [N,N1,num_entity] 预测每个候选实体对应各个类别的置信度
            # rel_logits: [N,N2,num_rel] 预测每个候选关系对应各个类别的置信度
            entity_logits, rel_logits = model(
                encodings=batch['encodings'],  # token id序列对象, [batch_size, seq_len] [N,T]
                context_masks=batch['context_masks'],  # 真实token位置为True，填充位置为False, [batch_size, seq_len] [N,T]
                # 实体位置的token为True，其它所有位置为False, [batch_size, entity_num, seq_len] [N,N1,T]
                entity_masks=batch['entity_masks'],
                # 每个实体的长度大小，[batch_size, entity_num] [N,N1]
                entity_sizes=batch['entity_sizes'],
                # 每个关系对应的实体索引下标，下标取值范围:[0,N1), [batch_size, rel_num, 2] [N,N2,2]
                relations=batch['rels'],
                # 每个实体关系对中实体之间的范围为True，其它范围为False, [batch_size, rel_num, seq_len], [N,N2,T]
                rel_masks=batch['rel_masks']
            )

            # compute loss and optimize parameters 计算损失，并且更新参数
            batch_loss = compute_loss.compute(
                entity_logits=entity_logits,  # 预测的各个实体的置信度[N,N1,entity_type_num]
                rel_logits=rel_logits,  # 预测的各个关系的置信度[N,N2,rel_type_num]
                rel_types=batch['rel_types'],  # 原始值，关系类别multi onehot值，[N,N2,rel_type_num]
                entity_types=batch['entity_types'],  # 原始值，实体类别id值 [N,N1]
                entity_sample_masks=batch['entity_sample_masks'],  # 填充位置为False，其它位置均为True, [N,N1]
                rel_sample_masks=batch['rel_sample_masks']  # 填充位置为False，其它位置均为True, [N,N2]
            )

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self._args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)
        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator 构建一个评估器对象
        predictions_path = os.path.join(self._log_path, f'predictions_{dataset.label}_epoch_{epoch}.json')
        examples_path = os.path.join(self._log_path, f'examples_%s_{dataset.label}_epoch_{epoch}.html')
        evaluator = Evaluator(
            dataset,  # 数据对象
            input_reader,  # 数据加载器对象
            self._tokenizer,  # token解析对象
            self._args.rel_filter_threshold,  # 关系预测阈值，默认0.4
            self._args.no_overlapping,  # 是否支持实体关系重叠，默认False
            predictions_path,  # 评估结果保存路径
            examples_path,  # 数据保存路径
            self._args.example_count  # 保存的数量
        )

        # create data loader 测试/验证数据集的创建
        dataset.switch_mode(Dataset.EVAL_MODE)  # 设置为eval，会影响dataset的数据获取逻辑
        data_loader = DataLoader(
            dataset,  # Dataset对象
            batch_size=self._args.eval_batch_size,  # 批次大小
            shuffle=False,
            drop_last=False,
            num_workers=self._args.sampling_processes,  # 数据加载的线程数目
            collate_fn=sampling.collate_fn_padding  # 聚合函数
        )

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)  # 总迭代批次
            batch_iteration = 0
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(
                    encodings=batch['encodings'],  # token id序列对象, [batch_size, seq_len] [N,T]
                    context_masks=batch['context_masks'],  # 真实token位置为True，填充位置为False, [batch_size, seq_len] [N,T]
                    # 实体位置的token为True，其它所有位置为False, [batch_size, entity_num, seq_len] [N,N1,T]
                    entity_masks=batch['entity_masks'],
                    # 每个实体的长度大小，[batch_size, entity_num] [N,N1]
                    entity_sizes=batch['entity_sizes'],
                    entity_spans=batch['entity_spans'],  # 给定每个实体的范围索引(开始索引和结束索引)
                    entity_sample_masks=batch['entity_sample_masks'],  # 填充位置为False，其它位置均为True, [N,N1]
                    inference=True,  # 给定模型推理采用训练还是预测的逻辑
                    # support_entity_types=[(3, 2), (3, 3), (2, 1), (3, 1)]  # TODO: 基于数据进行整理
                    support_entity_types=None  # TODO: 基于数据进行整理
                )
                # entity_clf: [N,N1,entity_types_num] 每个样本每个实体预测属于各个实体类别的概率值(softmax概率)
                # rel_clf: [N,N2,rel_types_num] 每个样本每个关系实体对预测属于各个关系类别的概率值(sigmoid概率)
                # rels: [N, N2, 2] 每个样本每个关系实体对中的具体下实体索引标
                entity_clf, rel_clf, rels = result

                # evaluate batch 构造预测的实体和关系数据
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

                # batch_iteration += 1
                # if batch_iteration > 1:
                #     # TODO: NOTE: 临时代码，主要用于快速看后续代码逻辑
                #     break

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()  # 计算评估指标
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self._args.store_predictions and not self._args.no_overlapping:
            evaluator.store_predictions()

        if self._args.store_examples:
            evaluator.store_examples()

    def _predict(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader):
        # create data loader 数据loader的创建
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self._args.eval_batch_size,
            shuffle=False, drop_last=False,
            num_workers=self._args.sampling_processes,
            collate_fn=sampling.collate_fn_padding
        )

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(
                    encodings=batch['encodings'], context_masks=batch['context_masks'],
                    entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                    entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                    inference=True
                )
                entity_clf, rel_clf, rels = result

                # convert predictions  将模型预测的结果(实体/关系属于各个类别的置信度信息)转换为用户可感知/好理解的结果数据
                predictions = prediction.convert_predictions(entity_clf, rel_clf, rels,
                                                             batch, self._args.rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities, batch_pred_relations = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)

        return prediction.store_predictions(dataset.documents, pred_entities, pred_relations,
                                            self._args.predictions_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self._args.weight_decay
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self._args.train_batch_size
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
