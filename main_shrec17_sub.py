import sys
import torch

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datasets.data import SH17_subclass_SingleView, SH17_subclass_MultiView

from torchvision.models import (
    AlexNet_Weights, VGG11_BN_Weights, VGG11_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights)

from models.vsformer import BaseImageClassifier, VSFormer

from utils import init, get_logger, cuda_seed_setup, dist_setup, dist_cleanup
from utils import get_optimizer, get_lr_scheduler, get_loss_fn, freeze_model, Trainer
from parser import args


def entry(rank, num_devices):

    dist_setup(rank)

    cuda_seed_setup()

    assert args.batch_size % args.world_size == args.test_batch_size % args.world_size == 0, \
        'Argument `batch_size` and `test_batch_size` should be divisible by `world_size`'

    logger = get_logger()
    logger.write(str(args), rank=rank)
    msg = f'{num_devices} GPUs are available and {args.world_size} of them are used. Ready for DDP training ...'
    logger.write(msg, rank=rank)

    # ------ stage one: train base model for image feature extraction
    if args.stage_one:
        # --- 1. prepare data 
        sv_train_set = SH17_subclass_SingleView(label_file=args.train_label, version=args.shrec_version, 
                                                num_classes=args.num_obj_classes)
        sv_samples_per_gpu = args.base_model_batch_size // args.world_size
        sv_train_sampler = DistributedSampler(sv_train_set, num_replicas=args.world_size, rank=rank)
        sv_train_loader = DataLoader(sv_train_set, sampler=sv_train_sampler, batch_size=sv_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(sv_train_loader): {len(sv_train_loader)}', rank=rank)

        sv_test_set = SH17_subclass_SingleView(label_file=args.test_label, version=args.shrec_version, 
                                                num_classes=args.num_obj_classes)
        sv_test_samples_per_gpu = args.base_model_test_batch_size // args.world_size
        sv_test_sampler = DistributedSampler(sv_test_set, num_replicas=args.world_size, rank=rank)
        sv_test_loader = DataLoader(sv_test_set, sampler=sv_test_sampler, batch_size=sv_test_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(sv_test_loader): {len(sv_test_loader)}', rank=rank)

        # --- 2. construct model
        if 'alexnet' in args.base_model_name: weights = AlexNet_Weights.DEFAULT
        elif 'vgg11' in args.base_model_name: weights = VGG11_Weights.DEFAULT
        elif 'resnet18' in args.base_model_name: weights = ResNet18_Weights.DEFAULT 
        elif 'resnet34' in args.base_model_name: weights = ResNet34_Weights.DEFAULT
        elif 'resnet50' in args.base_model_name: weights = ResNet50_Weights.DEFAULT
        else: weights = None
        sv_classifier = BaseImageClassifier(model_name=args.base_model_name, base_feature_dim=args.base_feature_dim, 
                                            num_channels=args.num_channels, num_classes=args.num_obj_classes, 
                                            pretrained=args.base_pretrain, weights=weights).to(rank)
        sv_classifier_ddp = DDP(sv_classifier, device_ids=[rank], find_unused_parameters=False)

        # --- 3. create optimizer
        sv_optimizer = get_optimizer(sv_classifier_ddp.parameters(), args.base_model_optim, args.base_model_lr, weight_decay=0.001, momentum=0.9)
        logger.write(f'Using {args.base_model_optim} optimizer ...', rank=rank)

        # --- 4. define lr scheduler
        sv_lr_scheduler = get_lr_scheduler(sv_optimizer, args.base_model_lr_scheduler, args.base_model_epochs)
        logger.write(f'Using {args.base_model_lr_scheduler} learning rate scheduler ...', rank=rank)
        
        # --- 5. define loss function
        sv_criterion = get_loss_fn(rank)
        logger.write(f'Using CrossEntropyLoss with label smoothing = {args.label_smoothing} ...', rank=rank)

        # --- 6. construct trainer and start training
        sv_trainer = Trainer(sv_classifier_ddp, sv_train_loader, sv_train_sampler, sv_test_loader, sv_test_sampler,
                                    sv_optimizer, sv_lr_scheduler, sv_criterion, type='sv')
        sv_trainer.train(rank, logger, args)

    # ------ stage two: train vsformer for aggregating multi-view information
    if args.stage_two:
        # --- 1. prepare data
        mv_train_set = SH17_subclass_MultiView(label_file=args.train_label, version=args.shrec_version, 
                    num_views=args.num_views, total_num_views=args.total_num_views, num_classes=args.num_obj_classes)
        mv_samples_per_gpu = args.batch_size // args.world_size
        mv_train_sampler = DistributedSampler(mv_train_set, num_replicas=args.world_size, rank=rank)
        mv_train_loader = DataLoader(mv_train_set, sampler=mv_train_sampler, batch_size=mv_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(mv_train_loader): {len(mv_train_loader)}', rank=rank)
        
        mv_test_set = SH17_subclass_MultiView(label_file=args.test_label, version=args.shrec_version, 
                    num_views=args.num_views, total_num_views=args.total_num_views, num_classes=args.num_obj_classes)
        mv_test_samples_per_gpu = args.test_batch_size // args.world_size
        mv_test_sampler = DistributedSampler(mv_test_set, num_replicas=args.world_size, rank=rank)
        mv_test_loader = DataLoader(mv_test_set, sampler=mv_test_sampler, batch_size=mv_test_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(mv_test_loader): {len(mv_test_loader)}', rank=rank)

        # --- 2. construct model
        #   --- 2.1 load base_model_weights 
        sv_classifier = BaseImageClassifier(model_name=args.base_model_name, base_feature_dim=args.base_feature_dim,  
                                            num_channels=args.num_channels, num_classes=args.num_obj_classes).to(rank)
        if args.resume: #'You should specify the pretrained `base_model_weights`'
            logger.write(f'Loading pretrained weights of {args.base_model_name} on {args.dataset} ...', rank=rank)
            map_location = torch.device('cuda:%d' % rank)
            sv_pretrained = torch.load(args.base_model_weights, map_location=map_location)
            sv_classifier.load_state_dict(sv_pretrained, strict=True)
        else:
            logger.write(f'Constructing VSFormer without pretrained `sv_classifier.feature_extractor` ...', rank=rank)

        if args.freeze_base_model:
            freeze_model(sv_classifier.parameters())

        mv_classifier = VSFormer(sv_classifier.feature_extractor, base_model_name=args.base_model_name,
                                            base_feature_dim=args.base_feature_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                            num_channels=args.num_channels, widening_factor=args.mlp_widen_factor, 
                                            max_dpr=args.max_dpr, atten_drop=args.atten_drop, mlp_drop=args.mlp_drop, 
                                            num_views=args.num_views, clshead_layers=args.clshead_layers, num_classes=args.num_obj_classes, 
                                            use_cls_token=args.use_cls_token, use_max_pool=args.use_max_pool, 
                                            use_mean_pool=args.use_mean_pool, use_pos_embs=args.use_pos_embs).to(rank)
        mv_classifier_ddp = DDP(mv_classifier, device_ids=[rank], find_unused_parameters=False)

        # --- 3. create optimizer
        mv_optimizer = get_optimizer(mv_classifier_ddp.parameters(), args.optim, args.lr)
        logger.write(f'Using {args.optim} optimizer ...', rank=rank)

        # --- 4. define lr scheduler
        mv_lr_scheduler = get_lr_scheduler(mv_optimizer, args.lr_scheduler, args.epochs)
        logger.write(f'Using {args.lr_scheduler} learning rate scheduler ...', rank=rank)
        
        # --- 5. define loss function
        mv_criterion = get_loss_fn(rank)
        logger.write(f'Using CrossEntropyLoss with label smoothing = {args.label_smoothing} ...', rank=rank)
        
        # --- 6. construct trainer and start training
        mv_trainer = Trainer(mv_classifier_ddp, mv_train_loader, mv_train_sampler, mv_test_loader, mv_test_sampler,
                                mv_optimizer, mv_lr_scheduler, mv_criterion, type='mv')
        mv_trainer.train(rank, logger, args)

    dist_cleanup()


if '__main__' == __name__:
    init()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            mp.spawn(entry, args=(num_devices,), nprocs=args.world_size)
        else:
            sys.exit('Only one GPU is available, the process will be much slower. Exit')
    else:
        sys.exit('CUDA is unavailable! Exit')
