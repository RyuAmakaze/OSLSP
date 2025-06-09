# Copyright Yu Yamaoka. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from config import (
    DAY_PROP as DayProp,
    CLASS_DIVER as ClassDiver,
    GAUSSIAN_SIGMA,
    RESIZE,
    GLOBAL_CROP_SIZE,
    LOCAL_CROP_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    HORIZONTAL_FLIP_PROB,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION,
    COLOR_JITTER_HUE,
    COLOR_JITTER_PROB,
    RANDOM_GRAYSCALE_PROB,
    GAUSSIAN_BLUR1_PROB,
    GAUSSIAN_BLUR2_PROB,
    GAUSSIAN_BLUR_LOCAL_PROB,
    SOLARIZATION_PROB,
    MLP_HIDDEN_DIM_1,
    MLP_HIDDEN_DIM_2,
    VALID_LOG_FREQ,
)

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--sim_weight', type=float, default=1.0, help="feature loss parameter")
    parser.add_argument('--class_weight', type=float, default=1.0, help="head loss parameter")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    #OS-LLP
    parser.add_argument('--bag_num', type=int, default=100, help="""bag size for each days. theory : N in paper." """)
    parser.add_argument('--class_num', type=int, default=4, help="""day class number. 0day, 3day, 5day, 7day is 4 class." """)
    parser.add_argument('--osllp_bins', type=int, default=6, help="""depend on combo of feature" 4C2""")
    parser.add_argument('--backbone_outdim', type=int, default=768, help="""Number of fatures""")
    parser.add_argument('--test_data_path', default="", help='/path/to/test_image/', type=str)
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============    
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")
    print("dataloader length", len(data_loader))

    #valid用のデータloader
    val_transform = transforms.Compose([
        transforms.Resize(RESIZE), 
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])
    if(args.test_data_path!=""):
        dataset_val = datasets.ImageFolder(args.test_data_path, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # ============ building Backbone_model and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():        
        #backbone_model
        Backbone_model = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = Backbone_model.embed_dim     
    else:
        print(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    """non head by yamaoka"""
    
    #student = utils.MultiCropWrapper(Backbone_model, DINOHead(
    #    embed_dim,
    #    args.out_dim,
    #    use_bn=args.use_bn_in_head,
    #    norm_last_layer=args.norm_last_layer,
    #))

    # Ianna: MLP as head
    student = utils.MultiCropWrapper(Backbone_model, MLP(embed_dim, args.class_num))
    
    # move networks to gpu
    Backbone_model, student = Backbone_model.cuda(), student.cuda()
    
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        print(f"student")
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    print(f"student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    lsp_loss = lspLoss(
        args.backbone_outdim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    # Ianna: Finetuning only attention layers #
    for name, param in student.named_parameters():
        if "attn" not in name:
            param.requires_grad = False
    
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        for name, param in student.named_parameters():
            if param.requires_grad:
                print(name)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        lsp_loss=lsp_loss,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, Backbone_model,  lsp_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
        
        # ============ valid model yama add ============
        if (epoch % args.saveckp_freq == 0 or epoch == args.epochs - 1)and(args.test_data_path!=""):
            test_stats = validate_network(val_loader, student, args.n_last_blocks, args.avgpool_patchtokens)
            #print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            #best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'lsp_loss': lsp_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

################################################################################
# Constants for day proportions and class divergence are provided by config.py
################################################################################

#class同士のsimを返す
def Class_Divergence(classA, classB):
    return ClassDiver[classA][classB]     

#クラス同士のsimを考慮した分布(ヒストグラム)をdatalen長で返す
def DayCombo_to_FeatNumData(dayA, dayB, datalen):
    data = torch.zeros(datalen)
    data_itr = 0
    for clsA, propA in enumerate(DayProp[dayA]):
        for clsB, propB in enumerate(DayProp[dayB]):
            add_data_num = int(propA * propB * datalen)
            if(data_itr + add_data_num > datalen):#datalen以上に配列アクセスしないような処理
                add_data_num = datalen - data_itr
            data[data_itr:data_itr+add_data_num] = Class_Divergence(clsA, clsB)
            data_itr += add_data_num     
    return data

#正規化されたpとqのjs情報量を返す
def jensen_shannon(p, q):
    # ジェンセン・シャノン情報量を計算する関数
    m = (0.5 * (p + q)).log()
    return 0.5 * (F.kl_div(m, p) + F.kl_div(m, q))

def custom_histc(true_distribution, bins, range_min=0, range_max=1):
    # データをTensorに変換
    true_distribution_tensor = torch.tensor(true_distribution)
    
    # バケットの幅を計算
    bin_width = (range_max - range_min) / bins
    
    # バケットの境界を生成
    bin_edges = torch.linspace(range_min, range_max, bins + 1)
    
    # データをバケットに割り当て
    bin_indices = ((true_distribution_tensor - range_min) / bin_width).floor().clamp(0, bins - 1).long()
    
    # バケットごとの頻度を計算
    hist = torch.zeros(bins, dtype=torch.int64)
    for index in bin_indices:
        hist[index] += 1

    return hist

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x
    
#Define yamaoka add end
##############################################################################################################
import itertools
import random
def train_one_epoch(student, Backbone_model, lsp_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    student.train()

    ############################################################################################################
    #同じ日付クラスから2枚をglobal, 引数の画像をlocalにして学習
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomApply(
            [transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
                hue=COLOR_JITTER_HUE,
            )],
            p=COLOR_JITTER_PROB,
        ),
        transforms.RandomGrayscale(p=RANDOM_GRAYSCALE_PROB),
    ])
    transform_global = transforms.Compose([
        transforms.Resize(RESIZE), 
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])
    transform_local = transforms.Compose([
        transforms.Resize(RESIZE), 
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ]) 
    dataset = data_loader.dataset  
    print(dataset.imgs[0]) 
    random.shuffle(dataset.imgs)# dataset.imgsをシャッフルする
    
    #https://chat.openai.com/share/6ef3af61-9b25-4461-9943-0649d324575b
    # データセットからNの枚数を計算する.
    N = len(dataset.imgs)
    class_arrays = [[] for _ in range(args.class_num)]# クラスごとにImageを分割するための空の配列を作成
    
    # クラスごとにImageを分割
    for img_path, img_class in dataset.imgs:
        class_arrays[img_class].append(img_path)
        
    # args.bag_numごとに分割されたBagを作成
    bags = [[] for _ in range(args.class_num)]
    for cls_id, class_array in enumerate(class_arrays): 
        print("class arrays length", len(class_array))
        for j in range(0, len(class_array)//args.bag_num):
            start_index = j * args.bag_num
            bags[cls_id].append(class_array[start_index:start_index+args.bag_num])
        print(cls_id, "class:bags len is ", len(bags[cls_id]))

    #bagsのshapeが正方行列ではない
    #bagsの2番めの次元のminで正方行列を作り、iterationを回す.
    min_bag_num = min(len(hoge) for hoge in bags)
    print("min_bag_num:", min_bag_num)

    #https://chat.openai.com/share/bf470c3c-1f97-49ad-aa54-710503acf67c
    indexes = list(itertools.product(range(args.class_num), range(min_bag_num)))# 全てのインデックスの組み合わせを生成
    random.shuffle(indexes)# インデックスの組み合わせをランダムにシャッフル
    print("len indexes(combo)", len(indexes))#奇数だとうまくいかない？
    
    #Start one epoch
    for it, index in enumerate(range(0, len(indexes)- (len(indexes)%2), 2)):#bagの組み合わせを取るために２つずつ
        print("epoch :", epoch, "bags iteration ; ", it, "bags shape", len(bags), len(bags[0]))
        it = len(data_loader) * epoch + it  # global training iteration
        combination = indexes[index:index+2]#鉄で2にしておく、1だとlist outしてしまう
        item1_day_class = combination[0][0]#int
        item1_bag_num = combination[0][1]#int
        item2_day_class = combination[1][0]#int
        item2_bag_num = combination[1][1]#int
        #print(combination)#XXX list out of --index!
        
        item1_bag = bags[item1_day_class][item1_bag_num]
        item2_bag = bags[item2_day_class][item2_bag_num]
        
        # teacher bagとBackbone_model bag内の画像を一度にGPUに乗っける.
        item1_images = []
        item2_images = []
        for i in range(args.bag_num):
            #print(item1_bag[i])
            #print(item2_bag[i])
            item1_images.append(transform_global(dataset.loader(item1_bag[i])).unsqueeze(0))#Transformを明示的に
            item2_images.append(transform_global(dataset.loader(item2_bag[i])).unsqueeze(0))#Transformを明示的に
        item1_images = [im.cuda(non_blocking=True) for im in item1_images]
        item2_images  = [im.cuda(non_blocking=True) for im in item2_images]
        
        InfeClass = []#クラス推論のllp loss計算用
        cos_sim_outputs = torch.zeros(args.bag_num)
        for i in range(args.bag_num):#2枚ずつcos simの値を計算
            images = [item1_images[i], item2_images[i]]
            item1_output = student.module.backbone(images[0])  # only the 2 global views pass through the teacher -> 1に変更 yamaoka chenge
            item2_output = student.module.backbone(images[1]).detach()
            #if i == 0:
            #    print(images[0].shape)
            #    print(item1_output,item2_output)
            #print(item1_output.shape,item2_output.shape) #torch.Size([1,768])
            cos_sim = lsp_loss(item1_output, item2_output, epoch)#[1,768]
            if torch.any(torch.isnan(cos_sim)):#issue25 https://github.com/RyuAmakaze/DINO_Day/issues/25
                print("issue25:cos sim is nan")
                print(item1_images[i], item2_images[i])
                print("teacher output max:" , torch.max(item1_output))
                print("Backbone_model output max:" , torch.max(item2_output))
                cos_sim = 0
            cos_sim_outputs[i] = cos_sim  
            
            #クラス推論
            class_output = student.module.head(item1_output.detach()) #class_output.shape is args.out_dim
            #print("class_output:", class_output)#[1,4]
            InfeClass.append(F.softmax(class_output, dim=1))
        
        print("cos_sim_outputs:", cos_sim_outputs)
        #np.save(loss_outputs, os.path.join(args.output_dir, str(epoch)+"_"+str(teacher_bag_num)+"_"+str(Backbone_model_bag_num)+"loss.npy")) 

        # ============ backbone loss ... ============
        #推論側の分布
        Infe_Distribution = cos_sim_outputs#F.softmax(cos_sim_outputs, dim=0)
        #Infe_Histgram = custom_histc(Infe_Distribution, bins=args.osllp_bins, range_min=0, range_max=1)
        Infe_Histgram = GaussianHistogram(bins=args.osllp_bins, min=0, max=1, sigma=GAUSSIAN_SIGMA)(Infe_Distribution)
        Infe_Histgram = Infe_Histgram / Infe_Histgram.sum()
        print("Infe_Histgram:", Infe_Histgram)

        print("item1_day_class", item1_day_class, "item2_day_class", item2_day_class)
        #真分布の用意
        True_Distribution = DayCombo_to_FeatNumData(item1_day_class, item2_day_class, args.bag_num)
        print("True_Distribution", True_Distribution)
        #True_Histgram = custom_histc(True_Distribution, bins=args.osllp_bins, range_min=0, range_max=1)#cos simの値は0から1
        True_Histgram = GaussianHistogram(bins=args.osllp_bins, min=0, max=1, sigma=GAUSSIAN_SIGMA)(True_Distribution)
        True_Histgram = True_Histgram / True_Histgram.sum()
        print("True_Histgram:", True_Histgram)
        
        #損失関数候補いくつか
        loss1 = F.cross_entropy(Infe_Histgram.unsqueeze(0), True_Histgram.unsqueeze(0))
        loss2 = F.kl_div(Infe_Histgram.log(), True_Histgram, reduction="batchmean")
        loss3 = jensen_shannon(Infe_Histgram, True_Histgram)
        
        # ============ head loss ... ============
        #クラス分類によるLLPロスの計算
        print(len(InfeClass), torch.cat(InfeClass, dim=0).shape)
        InfeClass_Distribution = torch.cat(InfeClass, dim=0).mean(dim=0)#推論側の分布
        TrueClass_distribution = torch.tensor(DayProp[item1_day_class]).to(InfeClass_Distribution.device)
        print(InfeClass_Distribution.shape)
        print(TrueClass_distribution.shape)
        head_loss = F.kl_div(InfeClass_Distribution.log(), TrueClass_distribution)
        print("head loss. day class", item1_day_class,InfeClass_Distribution,TrueClass_distribution)
        
        # ============ total loss ... ============
        loss = args.sim_weight*loss2.cuda() + args.class_weight*head_loss.cuda()
        print("cross_entropy:", loss1.item(), "|kl_div:", loss2.item(), "|js_div:", loss3.item(), "head_loss:", head_loss.item())
        print("loss:", loss)
        print("****************************************************************")
####################################################################################################################  
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Backbone_model update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            #fp16_scaler.scale(loss2.cuda()).backward(retain_graph=True)
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class lspLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, Backbone_model_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.Backbone_model_temp = Backbone_model_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, Backbone_model_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and Backbone_model networks.
        Args:
            Backbone_model : torch.Size([4, 65536])
            teacher : torch.Size([2, 65536])
        return:
            cos_simに変更
        """
        Backbone_model_out = Backbone_model_output / self.Backbone_model_temp#何か知らんけど
        #print("Backbone_model_temp", self.Backbone_model_temp)
        #Backbone_model_out = Backbone_model_out.chunk(self.ncrops)
        #Backbone_model_out = Backbone_model_output

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        #teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = (teacher_output - self.center) / (temp * teacher_output.norm(dim=1, keepdim=True))#何か知らんけど
        #teacher_out = teacher_out.detach().chunk(2)
        #teacher_out = teacher_output / teacher_output.norm(dim=1, keepdim=True)
        
        total_loss = 0
        n_loss_terms = 0
        
        Backbone_model_norm = Backbone_model_out / Backbone_model_out.norm(dim=1, keepdim=True)
        #print(teacher_out.shape, Backbone_model_norm.shape)
        cos_sim = F.cosine_similarity(Backbone_model_norm, teacher_out, dim = 1)
        cos_sim = 0.5 * (cos_sim + 1)# 0から1の範囲にスケーリング
        #print("Cos_sim", cos_sim)
        total_loss = cos_sim
        """
        for iq, q in enumerate(teacher_out):
            for v in range(len(Backbone_model_out)):
                if v == iq:
                    # we skip cases where Backbone_model and teacher operate on the same view
                    continue
                
                #loss = torch.sum(-q * F.log_softmax(Backbone_model_out[v], dim=-1), dim=-1)
                #total_loss += loss.mean()
                
                #cos類似度に変更 yamaoka
                #Backbone_model_log_softmax = F.softmax(Backbone_model_out[v], dim=-1)
                #print(Backbone_model_out[v].shape)#torch.Size([65536])
                #print(q.shape)#torch.Size([65536])
                Backbone_model_norm = Backbone_model_out[v] / Backbone_model_out[v].norm(dim=0, keepdim=True)
                #Backbone_model_log_softmax_flat = torch.flatten(Backbone_model_log)
                #q_flat = torch.flatten(q)
                #cos_sim = torch.dot(Backbone_model_log, q) / (torch.linalg.norm(Backbone_model_log) * torch.linalg.norm(q))
                #cos_sim = Backbone_model_log @ q.t()
                cos_sim = F.cosine_similarity(Backbone_model_norm, q, dim = 0)
                cos_sim = 0.5 * (cos_sim + 1)# 0から1の範囲にスケーリング
                total_loss += cos_sim
                ###change end
                n_loss_terms += 1
        total_loss /= n_loss_terms
        """
        #self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=COLOR_JITTER_BRIGHTNESS,
                    contrast=COLOR_JITTER_CONTRAST,
                    saturation=COLOR_JITTER_SATURATION,
                    hue=COLOR_JITTER_HUE,
                )],
                p=COLOR_JITTER_PROB,
            ),
            transforms.RandomGrayscale(p=RANDOM_GRAYSCALE_PROB),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(GLOBAL_CROP_SIZE, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(GAUSSIAN_BLUR1_PROB),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(GLOBAL_CROP_SIZE, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(GAUSSIAN_BLUR2_PROB),
            utils.Solarization(SOLARIZATION_PROB),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(LOCAL_CROP_SIZE, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=GAUSSIAN_BLUR_LOCAL_PROB),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
#ianna add
class MLP(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__() # python3では引数省略可能
        self.fc1 = nn.Linear(dim, MLP_HIDDEN_DIM_1, bias=True)
        self.fc2 = nn.Linear(MLP_HIDDEN_DIM_1, MLP_HIDDEN_DIM_2, bias=True)
        self.fc3 = nn.Linear(MLP_HIDDEN_DIM_2, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y

#yamaoka add
def create_array_with_one(N, i):
    # 長さNの配列を0で初期化
    array = torch.zeros(N, dtype=torch.float32)
    # 特定のインデックスiを1に設定
    if 0 <= i < N:
        array[i] = 1.0
    return array

@torch.no_grad()
def validate_network(val_loader, model, n, avgpool):
    CLASS_CHANGE_FLAG = True
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, VALID_LOG_FREQ, header):
        """TESTは以下の配列で入っている
            0: 'blue', -> 0
            1: 'orange', -> 3
            2: 'pink', -> 2
            3: 'red', -> 1
            4: 'yellow', -> 4
        """
        if(CLASS_CHANGE_FLAG):
            Test2ColorClass = [0,3,2,1,4]
            #Trainに合わせたLabelにする
            #print(target)
            #print("validate_network start. target unique:", torch.unique(target))
            assert len(Test2ColorClass)>max(torch.unique(target)), "Test classの変換に失敗"+str(torch.unique(target))
            for i, tar in enumerate(target):
                target[i] = Test2ColorClass[tar]#ラベル合わせ(trainとTestでフォルダ構造が異なるため)
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward(特徴量抽出)
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.module.backbone.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        #print(output.shape)     
        output = model.module.head(output)#[512, 1536]->[512, 4]
        output = F.softmax(output, dim=1) #[512, 4]各クラスの確信度
        max_index = torch.argmax(output, dim=1)
        target_onehot = create_array_with_one(len(Test2ColorClass), target.item()).unsqueeze(0).cuda(non_blocking=True)
        
        #print(output.shape, output)
        #print(target_onehot.shape, target_onehot)
        loss = nn.CrossEntropyLoss()(output, target_onehot)
        
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        
        #print('valid loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':    
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    assert len(DayProp[0])==args.class_num, "class num is NOT same."
    train_dino(args)
