#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DKT模型训练脚本 - Concept层面（138个KC）
基于 trainDKT.py，针对 concept 级别的知识追踪进行优化
"""
import os
import time
from argparse import Namespace, ArgumentParser

import numpy as np
from mindspore import load_param_into_net, load_checkpoint, save_checkpoint
from mindspore.dataset import GeneratorDataset
from mindspore.nn import Adam, BCELoss, PolynomialDecayLR
from mindspore import context
from tqdm import tqdm

from KTScripts.BackModels import nll_loss
from KTScripts.DataLoader import KTDataset, RecDataset, RetrievalDataset
from KTScripts.PredictModel import ModelWithLoss, ModelWithLossMask, ModelWithOptimizer
from KTScripts.utils import set_random_seed, load_model, evaluate_utils


def get_concept_level_options(parser: ArgumentParser):
    """
    获取concept级别训练的配置选项
    """
    model = ['DKT', 'Transformer', 'CoKT', 'GRU4Rec']
    parser.add_argument('-m', '--model', type=str, choices=model, default='DKT', 
                       help="Model to use (推荐使用DKT)")
    parser.add_argument('-d', '--dataset', type=str, default='assist09_kcs', 
                       help="Dataset to use (concept级别数据)")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--load_model', action='store_true', 
                       help="是否加载已有模型继续训练")
    parser.add_argument('--without_label', action='store_true')
    parser.add_argument('-c', '--cuda', type=int, default=0, 
                       help="GPU设备ID，-1表示使用CPU")
    parser.add_argument('-e', "--num_epochs", type=int, default=20, 
                       help="训练轮数（concept级别建议20轮）")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="初始学习率")
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--valid_step', type=int, default=100)
    parser.add_argument("--postfix", type=str, default='', 
                       help="保存模型文件名的后缀")
    parser.add_argument("--rand_seed", type=int, default=42, 
                       help="随机种子")
    
    # Concept级别的模型配置
    parser.add_argument('--embed_size', type=int, default=128, 
                       help="Embedding维度（138个KC建议128）")
    parser.add_argument('--hidden_size', type=int, default=128, 
                       help="LSTM隐藏层维度")
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128, 
                       help="批次大小（concept级别建议128）")
    parser.add_argument('--dropout', type=float, default=0.3, 
                       help="Dropout比例（concept级别建议0.3）")
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--l2_reg', type=float, default=1e-5, 
                       help="L2正则化系数（concept级别建议小一些）")
    parser.add_argument('--pre_hidden_sizes', type=int, nargs='+', 
                       default=[256, 64, 16])
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--forRec', action='store_true')
    
    args = parser.parse_args().__dict__
    args = Namespace(**args)
    
    # 生成实验名称
    args.exp_name = f'{args.model}_{args.dataset}_concept138'
    if args.without_label:
        args.exp_name += '_without'
    if args.postfix != '':
        args.exp_name += f'_{args.postfix}'
    
    # 设置运行环境
    if args.cuda >= 0:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=args.cuda)
        print(f"使用GPU: {args.cuda}")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        print("使用CPU")
    
    return args


def main(args: Namespace):
    print("\n" + "="*60)
    print(f"DKT训练 - Concept级别（138个KC）")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"Embedding维度: {args.embed_size}")
    print(f"Hidden维度: {args.hidden_size}")
    print(f"Batch大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"Dropout: {args.dropout}")
    print(f"随机种子: {args.rand_seed}")
    print("="*60 + "\n")
    
    set_random_seed(args.rand_seed)
    
    # 加载数据集
    dataset = RecDataset if args.forRec else (RetrievalDataset if args.retrieval else KTDataset)
    dataset = dataset(os.path.join(args.data_dir, args.dataset))
    
    # 获取特征数量（应该是138）和用户数量
    args.feat_nums, args.user_nums = dataset.feats_num, dataset.users_num
    print(f"✅ 数据集加载成功:")
    print(f"   - KC数量: {args.feat_nums}")
    print(f"   - 学生数量: {args.user_nums}")
    
    # 验证KC数量
    if args.feat_nums != 138:
        print(f"⚠️  警告: 期望138个KC，实际获得{args.feat_nums}个")
    
    # 创建数据加载器
    if args.retrieval:
        dataset = GeneratorDataset(source=dataset,
                                   column_names=['intra_x', 'inter_his', 'inter_r', 'y', 'mask', 'inter_len'],
                                   shuffle=False, num_parallel_workers=8, python_multiprocessing=False)
        dataset = dataset.batch(args.batch_size, num_parallel_workers=1)
        train_data, test_data = dataset.split([0.8, 0.2], randomize=False)
    else:
        dataset = GeneratorDataset(source=dataset, column_names=['x', 'y', 'mask'], shuffle=True)
        dataset = dataset.batch(args.batch_size, num_parallel_workers=1)
        train_data, test_data = dataset.split([0.8, 0.2], randomize=True)
    
    if args.forRec:
        args.output_size = args.feat_nums
    
    train_total = train_data.get_dataset_size()
    test_total = test_data.get_dataset_size()
    print(f"   - 训练批次数: {train_total}")
    print(f"   - 测试批次数: {test_total}\n")
    
    # 创建模型
    model = load_model(args)
    model_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 加载预训练模型（如果指定）
    if args.load_model:
        if os.path.exists(f'{model_path}.ckpt'):
            load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
            print(f"✅ 加载模型: {model_path}.ckpt\n")
        else:
            print(f"⚠️  模型文件不存在: {model_path}.ckpt")
            print(f"   从头开始训练...\n")
    
    # 优化器
    polynomial_decay_lr = PolynomialDecayLR(learning_rate=args.lr,
                                            end_learning_rate=args.min_lr,
                                            decay_steps=train_total // 10 + 1,
                                            power=0.5,
                                            update_decay_steps=True)
    optimizer = Adam(model.trainable_params(), learning_rate=polynomial_decay_lr, weight_decay=args.l2_reg)
    
    # 损失函数
    if args.forRec:
        model_with_loss = ModelWithLossMask(model, nll_loss)
    else:
        model_with_loss = ModelWithLoss(model, BCELoss(reduction='mean'))
    
    model_train = ModelWithOptimizer(model_with_loss, optimizer, args.forRec)
    
    best_val_auc = 0
    best_epoch = 0
    
    print('-' * 60)
    print("开始训练")
    print('-' * 60)
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print('='*60)
        
        # 训练阶段
        avg_time = 0
        epoch_loss = 0
        epoch_acc = 0
        epoch_auc = 0
        
        model_train.set_train()
        for i, data in enumerate(tqdm(train_data.create_tuple_iterator(), 
                                       total=train_total, 
                                       desc=f"训练 Epoch {epoch+1}")):
            t0 = time.perf_counter()
            loss, output_data = model_train(*data)
            loss = loss.asnumpy()
            acc, auc = evaluate_utils(*output_data)
            avg_time += time.perf_counter() - t0
            
            epoch_loss += loss
            epoch_acc += acc
            epoch_auc += auc
            
            # 每50个batch打印一次
            if (i + 1) % 50 == 0 or i == 0:
                print(f'  batch {i+1}/{train_total} | '
                      f'avg_time: {avg_time/(i+1):.4f}s | '
                      f'loss: {loss:.4f} | '
                      f'acc: {acc:.4f} | '
                      f'auc: {auc:.4f}')
        
        # 打印训练epoch统计
        print(f"\n训练统计:")
        print(f"  平均loss: {epoch_loss/train_total:.4f}")
        print(f"  平均acc: {epoch_acc/train_total:.4f}")
        print(f"  平均auc: {epoch_auc/train_total:.4f}")
        
        # 验证阶段
        print(f"\n{'-'*60}")
        print("开始验证")
        print('-'*60)
        
        val_eval = [[], []]
        loss_total, data_total = 0, 0
        model_with_loss.set_train(False)
        
        for data in tqdm(test_data.create_tuple_iterator(), 
                        total=test_total, 
                        desc=f"验证 Epoch {epoch+1}"):
            loss, output_data = model_with_loss.output(*data)
            val_eval[0].append(output_data[0].asnumpy())
            val_eval[1].append(output_data[1].asnumpy())
            loss_total += loss.asnumpy() * len(data[0])
            data_total += len(data[0])
        
        val_eval = [np.concatenate(_) for _ in val_eval]
        acc, auc = evaluate_utils(*val_eval)
        
        print(f"\n验证结果:")
        print(f"  loss: {loss_total/data_total:.4f}")
        print(f"  acc: {acc:.4f}")
        print(f"  auc: {auc:.4f}")
        
        # 保存最佳模型
        if auc >= best_val_auc:
            best_val_auc = auc
            best_epoch = epoch + 1
            save_checkpoint(model, f"{model_path}.ckpt")
            print(f"\n✅ 新的最佳模型已保存! (AUC: {best_val_auc:.4f})")
        
        print(f"\n当前最佳 AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    
    # 测试阶段
    print(f"\n{'='*60}")
    print("最终测试")
    print('='*60)
    
    # 加载最佳模型
    load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
    print(f"加载最佳模型 (Epoch {best_epoch})")
    
    val_eval = [[], []]
    loss_total, data_total = 0, 0
    model_with_loss.set_train(False)
    
    for data in tqdm(test_data.create_tuple_iterator(), 
                    total=test_total, 
                    desc="测试"):
        loss, output_data = model_with_loss.output(*data)
        val_eval[0].append(output_data[0].asnumpy())
        val_eval[1].append(output_data[1].asnumpy())
        loss_total += loss.asnumpy() * len(data[0])
        data_total += len(data[0])
    
    val_eval = [np.concatenate(_) for _ in val_eval]
    acc, auc = evaluate_utils(*val_eval)
    
    print(f"\n最终测试结果:")
    print(f"  loss: {loss_total/data_total:.4f}")
    print(f"  acc: {acc:.4f}")
    print(f"  auc: {auc:.4f}")
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"模型保存路径: {model_path}.ckpt")
    print(f"最佳验证AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"最终测试AUC: {auc:.4f}")
    print('='*60 + "\n")


if __name__ == '__main__':
    parser = ArgumentParser("DKT Training - Concept Level (138 KCs)")
    args_ = get_concept_level_options(parser)
    main(args_)

