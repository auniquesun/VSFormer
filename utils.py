import os
import time
import wandb
import shutil
import logging
from datetime import datetime
import numpy as np

import torch
import torch.distributed as dist
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from parser import args


def init():
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists(os.path.join('runs', args.task)):
        os.makedirs(os.path.join('runs', args.task))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name)):
        os.makedirs(os.path.join('runs', args.task, args.proj_name))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name, args.exp_name)):
        os.makedirs(os.path.join('runs', args.task, args.proj_name, args.exp_name))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files')):
        os.makedirs(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'weights')):
        os.makedirs(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'weights'))

    shutil.copy(args.main_program, os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    shutil.copy(f'models/{args.model_name}', os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    shutil.copy('utils.py', os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    shutil.copy(args.shell_name, os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    

def cuda_seed_setup():
    # If you are working with a multi-GPU model, `torch.cuda.manual_seed()` is insufficient 
    # to get determinism. To seed all GPUs, use manual_seed_all().
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def dist_setup(rank):
    # initialization for distributed training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)


def dist_cleanup():
    dist.destroy_process_group()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_pos = 0
        self.num_neg = 0
        self.total = 0

    def update(self, num_pos, num_neg, n=1):
        self.num_pos += num_pos
        self.num_neg += num_neg
        self.total += n

    def pos_count(self, pred, label):
        # torch.eq(a,b): Computes element-wise equality
        results = torch.eq(pred, label)
        
        return results.sum()


class Logger(object):
    def __init__(self, logger_name='Test', log_level=logging.INFO, log_path='runs', log_file='test.log'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_path, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def write(self, msg, rank=-1):
        if rank == 0:
            self.logger.info(msg)


def get_logger():
    logger_name = args.proj_name
    log_path = os.path.join('runs', args.task, args.proj_name, args.exp_name)
    log_file = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    return Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)
    

class Trainer(object):
    def __init__(self, model, train_loader, train_sampler, test_loader, test_sampler, optimizer, lr_scheduler, criterion, type='mv'):
        
        self.model = model
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.test_loader = test_loader
        self.test_sampler = test_sampler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.type = type

    def train(self, rank, logger, args):
        if rank == 0:
            os.environ["WANDB_BASE_URL"] = args.wb_url
            wandb.login(key=args.wb_key)
            wandb.init(project=args.proj_name, name=args.exp_name)

        logger.write('Start DDP training on %s ...' % args.dataset, rank=rank)

        test_best_inst_epoch = 0
        test_best_class_epoch = 0
        test_best_inst_acc = .0
        test_best_class_acc = .0
        total_train_interval = .0
        total_test_interval = .0
        total_train_samples = 0
        total_test_samples = 0

        epochs = args.epochs if self.type == 'mv' else args.base_model_epochs
        for epoch in range(epochs):
            # ------ Train
            self.model.train()

            self.train_sampler.set_epoch(epoch)
            self.test_sampler.set_epoch(epoch)

            # average losses across all scanned batches within an epoch
            train_loss = AverageMeter()
            acc_meter = AccuracyMeter()
            train_interval = .0
            train_samples = 0

            for i, data in enumerate(self.train_loader):
                # data: (class_id, imgs_within_a_batch, imgs_path_within_a_batch)
                self.optimizer.zero_grad()

                if self.type == 'mv':
                    # B: batch_size, V: num_views
                    B, V, C, H, W = data[1].size()
                    in_data = data[1].view(-1, C, H, W).to(rank)
                elif self.type == 'sv':
                    # B: batch_size
                    B, C, H, W = data[1].size()
                    in_data = data[1].to(rank)
                target = data[0].to(rank).long()

                start = time.time()
                out_data = self.model(in_data)
                loss = self.criterion(out_data, target)
                loss.backward()
                self.optimizer.step()
                train_interval += time.time() - start
                train_samples += B
                train_loss.update(loss, n=B)

                pred = out_data.argmax(dim=1)
                pos = acc_meter.pos_count(pred, target)
                acc_meter.update(pos, B-pos, n=B)
                
                if i % args.print_freq == 0:
                    logger.write(f'Epoch: {epoch}/{epochs}, Batch: {i}/{len(self.train_loader)}, '
                                f'Loss : {train_loss.avg.item()}, Accuracy: {acc_meter.num_pos.item()/acc_meter.total} ', rank=rank)
            total_train_interval += train_interval
            total_train_samples += train_samples
            
            # ------ Test
            with torch.no_grad():
                train_acc = acc_meter.num_pos.item() / acc_meter.total
                logger.write('Start testing on %s ...' % args.dataset, rank=rank)
                
                test_loss, test_inst_acc, test_class_acc, test_interval, test_samples = self.test(rank, args)
                total_test_interval += test_interval
                total_test_samples += test_samples

                logger.write('Got test instance accuracy on [%s]: %f' % (args.dataset, test_inst_acc), rank=rank)
                logger.write('Got test class accuracy on [%s]: %f' % (args.dataset, test_class_acc), rank=rank)

                if rank == 0:
                    if test_inst_acc >= test_best_inst_acc:
                        test_best_inst_acc = test_inst_acc
                        test_best_inst_epoch = epoch
                        logger.write(f'Find new best Instance Accuracy: <{test_best_inst_acc}> at epoch [{test_best_inst_epoch}] !', rank=rank)
                        logger.write('Saving best model ...', rank=rank)
                        save_dict = self.model.module.state_dict()
                        save_path = os.path.join('runs', args.task, args.proj_name, args.exp_name, 'weights', f'{self.type}_model_best.pth')
                        torch.save(save_dict, save_path)
                    if test_class_acc >= test_best_class_acc:
                        test_best_class_acc = test_class_acc
                        test_best_class_epoch = epoch

                    wandb_log = dict()
                    if args.lr_scheduler == 'coswarm':
                        wandb_log['learning_rate'] = self.lr_scheduler.get_lr()[0]
                    else:
                        wandb_log['learning_rate'] = self.lr_scheduler.get_last_lr()[0]
                    wandb_log['train_loss'] = train_loss.avg.item()
                    wandb_log['train_acc'] = train_acc
                    # average time consuming of each training sample in current epoch
                    wandb_log['train_interval'] = train_interval / train_samples
                    # average time consuming of each training sample in all past epochs
                    wandb_log['total_train_interval'] = total_train_interval / total_train_samples
                    wandb_log['test_loss'] = test_loss
                    wandb_log['test_inst_acc'] = test_inst_acc
                    wandb_log['test_class_acc'] = test_class_acc
                    wandb_log['test_best_inst_acc'] = test_best_inst_acc
                    wandb_log['test_best_class_acc'] = test_best_class_acc
                    wandb_log['test_best_inst_epoch'] = test_best_inst_epoch
                    wandb_log['test_best_class_epoch'] = test_best_class_epoch
                    # average time consuming of each test sample in current epoch
                    wandb_log['test_interval'] = test_interval / test_samples
                    # average time consuming of each test sample in all past epochs
                    wandb_log['total_test_interval'] = total_test_interval / total_test_samples
                    wandb.log(wandb_log)

            # adjust learning rate after every epoch
            self.lr_scheduler.step()

        if rank == 0:
            logger.write(f'Final best Instance Accuracy on [{args.dataset}]: <{test_best_inst_acc}> at epoch [{test_best_inst_epoch}] !', rank=rank)
            logger.write(f'Final best Class Accuracy on [{args.dataset}]: <{test_best_class_acc}> at epoch [{test_best_class_epoch}] !', rank=rank)
            logger.write(f'End of DDP training on [{args.dataset}] ...', rank=rank)
            wandb.finish()

    def test(self, rank, args):
        self.model.eval()
 
        test_loss = AverageMeter()
        acc_meter = AccuracyMeter()
        test_interval = .0
        test_samples = 0
        # correct_pred_class = torch.zeros(args.num_obj_classes, device=f'cuda:{rank}')
        # all_pred_class = torch.zeros(args.num_obj_classes, device=f'cuda:{rank}')
        correct_pred_class = np.zeros(args.num_obj_classes)
        all_pred_class = np.zeros(args.num_obj_classes)
        for data in self.test_loader:

            if self.type == 'mv':
                B, V, C, H, W = data[1].size()
                in_data = data[1].view(-1, C, H, W).to(rank)
            elif self.type == 'sv':
                B, C, H, W = data[1].size()
                in_data = data[1].to(rank)
            target = data[0].to(rank).long()
            
            start = time.time()
            out_data = self.model(in_data)
            test_interval += time.time() - start
            test_samples += B
            
            loss = self.criterion(out_data, target)
            test_loss.update(loss, n=B)

            pred = out_data.argmax(dim=1)
            pos = acc_meter.pos_count(pred, target)
            acc_meter.update(pos, B-pos, n=B)

            # compute accuracy for each class
            results = pred == target
            for i in range(results.size()[0]):
                idx = target.cpu().numpy().astype('int')[i]
                if bool(results[i].cpu().numpy()):
                    correct_pred_class[idx] += 1
                all_pred_class[idx] += 1
            
        test_loss = test_loss.avg.item()
        test_inst_acc = acc_meter.num_pos.item() / acc_meter.total
        # 检查是否除数为 0
        # self.check_division(all_pred_class)
        test_class_acc = np.mean(correct_pred_class / all_pred_class)

        return test_loss, test_inst_acc, test_class_acc, test_interval, test_samples

    @staticmethod
    def check_division(all_pred_class):
        if np.any(all_pred_class==0, axis=0):
            print('Warning: there are elements equivalent to 0 in `all_pred_class`')
            print('all_pred_class:', all_pred_class)


def get_optimizer(parameters, opt, lr, weight_decay=0, momentum=0):
    if opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr)

    return optimizer


def get_lr_scheduler(optimizer, lr_sche, epochs):
    if lr_sche == 'cos':
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=epochs)
    elif lr_sche == 'coswarm':
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.step_size,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warm_epochs,
            gamma=args.gamma)
    elif lr_sche == 'plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=args.factor,
            patience=args.patience)
    elif lr_sche == 'step':
        lr_scheduler = StepLR(
            optimizer, 
            step_size=args.step_size,
            gamma=args.gamma)

    return lr_scheduler


def get_loss_fn(rank):
    return CrossEntropyLoss(label_smoothing=args.label_smoothing).to(rank)


def freeze_model(parameters):
    for p in parameters:
        p.requires_grad = False