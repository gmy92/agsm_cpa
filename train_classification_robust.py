"""
Robust RISurConv classification training script with partial consistency regularization
and ScanNet-specific data augmentation
"""

import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from data_utils.ScanObjectNNLoader import ScanObjectNN

# Add the project root to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='risurconv_cls_robust',
                        help='model name [default: risurconv_cls_robust]')
    parser.add_argument('--batch_size', type=int, default=16,  # 保持与原版一致
                        help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=450, type=int,  # 恢复原版 epoch 数量
                        help='Epoch to run [default: 450]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='weight decay [default: 2e-4]')
    parser.add_argument('--npoint', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--use_cpu', action='store_true', default=False,
                        help='Whether to use CPU for training [default: False]')
    # 添加 ScanObjectNN 需要的参数
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    parser.add_argument('--data_type', type=str, default='hardest',
                        help='Data type [default: hardest]')
    parser.add_argument('--data_path', type=str, default='../../data/scanobjectnn/main_split/',
                        help='data path')

    return parser.parse_args()


def random_point_dropout(points, max_ratio=0.3):
    """Random point dropout to simulate occlusion / incomplete scanning"""
    # points: [B, N, C]
    B, N, C = points.shape
    drop_ratio = np.random.rand(B) * max_ratio  # Different ratio for each sample
    for b in range(B):
        keepN = int(N * (1 - drop_ratio[b]))
        idx = np.random.permutation(N)[:keepN]
        # Pad remaining points with duplicates
        if keepN < N:
            pad_idx = np.random.randint(0, keepN, (N-keepN,))
            idx = np.concatenate([idx, pad_idx])
        points[b] = points[b][idx, :]
    return points


def random_block_erasing(points, p=0.5):
    """Random block erasing to simulate scanning occlusion"""
    if np.random.rand() > p:
        return points
    B, N, C = points.shape
    xyz = points[..., :3]
    min_xyz = np.min(xyz, axis=1, keepdims=True)  # [B, 1, 3]
    max_xyz = np.max(xyz, axis=1, keepdims=True)  # [B, 1, 3]
    center = min_xyz + (max_xyz - min_xyz) * np.random.rand(B, 1, 3)
    size = (max_xyz - min_xyz) * (0.2 + 0.3 * np.random.rand(B, 1, 3))
    mask = np.all((xyz > center - size/2) & (xyz < center + size/2), axis=-1)  # [B, N]
    # For masked points, do dropout
    for b in range(B):
        idx_keep = np.where(~mask[b])[0]
        if len(idx_keep) < N // 4:
            continue
        # Pad with random samples from kept points
        pad_idx = np.random.choice(idx_keep, N, replace=True)
        points[b] = points[b][pad_idx, :]
    return points


def weak_augment(points_np):
    """Weak augmentation: basic transformations"""
    import provider
    points = points_np.copy()
    # Basic transformations
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    return points


def strong_augment(points_np):
    """Strong augmentation: simplified version to avoid over-regularization"""
    import provider
    points = points_np.copy()
    # Random point dropout (reduced intensity)
    points = random_point_dropout(points, max_ratio=0.2)
    # Basic transformations only (removed block erasing for now)
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    # Reduced jitter
    jitter = np.random.normal(loc=0.0, scale=0.01, size=points[:, :, 0:3].shape)
    points[:, :, 0:3] += jitter.astype(points.dtype)
    return points


def test(model, loader, args, num_class=15):
    """
    Test function to calculate both instance and class accuracy
    """
    model = model.eval()
    class_acc = np.zeros((num_class, 3))  # [total_count, correct_count, accuracy]
    
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points = points.data.numpy()
        points = torch.Tensor(points)
        # Don't transpose - the model expects [B, N, C] format
        # points = points.transpose(2, 1)
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        
        for i in range(len(target)):
            class_acc[target[i], 0] += 1  # total count
            if target[i] == pred_choice[i]:
                class_acc[pred_choice[i], 1] += 1  # correct count
    
    # Calculate class accuracy
    for i in range(num_class):
        if class_acc[i, 0] > 0:
            class_acc[i, 2] = class_acc[i, 1] / class_acc[i, 0]  # accuracy
    
    instance_average = sum(class_acc[:, 1]) / sum(class_acc[:, 0])  # overall accuracy
    class_average = sum(class_acc[:, 2]) / num_class  # mean class accuracy
    
    return instance_average, class_average


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create experiment directory
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = os.path.join(BASE_DIR, 'log', 'classification_robust')
    if args.log_dir is None:
        exp_dir = os.path.join(exp_dir, timestr)
    else:
        exp_dir = os.path.join(exp_dir, args.log_dir)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (exp_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # Data loading - follow the original approach
    log_string('Load dataset ...')
    if args.data_type == 'OBJ_NOBG':
        data_path = '../../data/scanobjectnn/main_split_nobg/'
    elif args.data_type == 'hardest' or 'OBJ_BG': 
        data_path = '../../data/scanobjectnn/main_split/'
    else:
        raise NotImplementedError()

    TRAIN_DATASET = ScanObjectNN(
        root=data_path,
        args=args,
        split='train'
    )
    TEST_DATASET = ScanObjectNN(
        root=data_path,
        args=args,
        split='test'
    )
    
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10
    )

    # Model loading
    num_class = 15
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, 1)  # Removed normal_channel parameter
    criterion = model.get_loss()
    
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.decay_rate
        )
    else:
        raise NotImplementedError
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.lr_decay
    )

    # Start training
    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    checkpoint_dir = exp_dir
    checkpoints_dir = os.path.join(checkpoint_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    '''TRANING'''
    logger.info('Start training...')
    lambda_consistency = 0.25  # Weight for consistency loss (reduced for partial consistency)
    consistency_batch_ratio = 0.2  # Only apply consistency loss to 20% of batches

    # Training loop
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # 恢复原版的学习率 warm-up 机制
        if epoch == 300:  # Warm up, a popular transformer training scheme
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
        
        # Training phase
        classifier.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0
        consistency_loss_sum = 0
        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # Determine if we should apply consistency loss for this batch
            apply_consistency = np.random.rand() < consistency_batch_ratio
            
            # Prepare weak and strong augmented versions for consistency training
            points_weak = points.data.numpy()
            points_strong = points.data.numpy()
            
            # Apply weak augmentation
            points_weak = weak_augment(points_weak)
            
            # Apply strong augmentation
            points_strong = strong_augment(points_strong)
            
            # Convert to tensors
            points_weak = torch.Tensor(points_weak)
            points_strong = torch.Tensor(points_strong)

            # Don't transpose - the model expects [B, N, C] format
            # points_weak = points_weak.transpose(2, 1)
            # points_strong = points_strong.transpose(2, 1)
            
            if not args.use_cpu:
                points_weak, points_strong, target = points_weak.cuda(), points_strong.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass with weak augmentation
            pred_weak, _ = classifier(points_weak)
            
            # Standard classification loss
            classification_loss = F.nll_loss(pred_weak, target.long())
            
            total_loss = classification_loss
            
            # Apply consistency loss only for selected batches
            if apply_consistency:
                # Forward pass with strong augmentation
                pred_strong, _ = classifier(points_strong)
                
                # Consistency loss using KL divergence
                consistency_loss = F.kl_div(
                    F.log_softmax(pred_strong, dim=1),
                    F.softmax(pred_weak.detach(), dim=1),
                    reduction='batchmean'
                )
                
                total_loss = classification_loss + lambda_consistency * consistency_loss
                consistency_loss_sum += consistency_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss_sum += total_loss.item()
            pred_choice = pred_weak.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            train_correct += correct.item()
            train_total += points_weak.size()[0]
            global_step += 1
            
        train_instance_acc = train_correct / train_total
        train_loss = train_loss_sum / len(trainDataLoader)
        avg_consistency_loss = consistency_loss_sum / len(trainDataLoader)
        
        log_string('Train Instance Accuracy: %f, Train Loss: %f, Avg Consistency Loss: %f' % 
                  (train_instance_acc, train_loss, avg_consistency_loss))
        log_string('lr: %f' % optimizer.param_groups[0]['lr'])
        
        # Evaluation phase
        with torch.no_grad():
            test_instance_acc, test_class_acc = test(classifier.eval(), testDataLoader, args, num_class=15)
            
            if (test_class_acc >= best_class_acc):
                best_class_acc = test_class_acc
            
            log_string('Test Instance Accuracy: %f, Test Class Accuracy: %f' % (test_instance_acc, test_class_acc))
            log_string('Best Instance Accuracy: %f, Best Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            
            # Save best model
            if test_instance_acc >= best_instance_acc:
                best_instance_acc = test_instance_acc
                best_epoch = epoch + 1
                
                # Save checkpoint
                savepath = os.path.join(checkpoints_dir, 'best_model.pth')
                log_string('Saving best model at epoch %d...' % best_epoch)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_instance_acc,
                    'class_acc': test_class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            
            log_string('Best Instance Accuracy: %f at epoch %d' % (best_instance_acc, best_epoch))
        
        scheduler.step()
        global_epoch += 1

    # Save final model
    savepath = os.path.join(checkpoints_dir, 'last_model.pth')
    log_string('Saving last model...')
    state = {
        'epoch': args.epoch,
        'instance_acc': best_instance_acc,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)
    
    log_string('Final Best Instance Accuracy: %f at epoch %d' % (best_instance_acc, best_epoch))


if __name__ == '__main__':
    args = parse_args()
    main(args)