"""
A. ModelNet40 Clean Data Training (HNR + FC-Loss Strategy)
    - Trains using Strong/Weak Augmentation (FC-Loss).
    - Tests on CLEAN ModelNet40 data (No TTA).
"""
import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# --- 核心增强函数和 TTA Helpers (用于训练和测试) ---
def random_block_erasing(points, p=0.7, max_size=0.5):
    if np.random.rand() > p: return points
    B, N, C = points.shape
    xyz = points[..., :3]; min_xyz = np.min(xyz, axis=1, keepdims=True); max_xyz = np.max(xyz, axis=1, keepdims=True)
    center = min_xyz + (max_xyz - min_xyz) * np.random.rand(B, 1, 3)
    size = (max_xyz - min_xyz) * (0.2 + max_size * np.random.rand(B, 1, 3))
    mask = np.all((xyz > center - size/2) & (xyz < center + size/2), axis=-1)
    for b in range(B):
        idx_keep = np.where(~mask[b])[0]
        if len(idx_keep) < N // 4: continue
        pad_idx = np.random.choice(idx_keep, N, replace=True)
        points[b] = points[b][pad_idx, :]
    return points

def rotate_point_cloud_with_normal(points):
    points[:, :, 0:6] = provider.rotate_point_cloud_with_normal(points[:, :, 0:6])
    return points

def strong_augment(points_np):
    points = points_np.copy()
    points = rotate_point_cloud_with_normal(points)
    points = random_block_erasing(points, p=0.7, max_size=0.5) 
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3], sigma=0.01, clip=0.03)
    return points

def weak_augment(points_np):
    points = points_np.copy()
    points = rotate_point_cloud_with_normal(points)
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    return points

def add_gaussian_noise_to_points(points_np, sigma=0.01):
    points = points_np.copy()
    points[:, :, 0:3] = provider.add_gaussian_noise(points[:, :, 0:3], sigma=sigma)
    return points

def apply_random_dropout(points_np, ratio=0.3):
    points = points_np.copy()
    points = provider.random_point_dropout(points, max_dropout_ratio=ratio) 
    return points
# ---------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='risurconv_cls_robust', help='model name [default: risurconv_cls_robust]')
    parser.add_argument('--num_category', default=40, type=int, help='training on ModelNet40')
    parser.add_argument('--epoch', default=450, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=5e-4, help='weight decay [SOTA config]')
    parser.add_argument('--step_size', type=int, default=30, help='Decay step [SOTA config]')
    parser.add_argument('--lr_decay', type=float, default=0.85, help='Decay rate [SOTA config]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', type=bool, default=True, help='use normals')
    parser.add_argument('--process_data', type=bool, default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    return parser.parse_args()

# --- 修复后的函数格式 ---
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inplace_relu(m): 
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1: 
        m.inplace = True
# -------------------------

# --- TEST FUNCTION (Clean Data) ---
def test(model, loader, args, num_class=40):
    classifier = model.eval()
    new_class_acc = np.zeros((num_class, 3))
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # Test on clean data
        points = torch.Tensor(points.data.numpy()) 
        if not args.use_cpu: points, target = points.cuda(), target.cuda()
        
        pred, _ = classifier(points)
        if len(pred.shape) == 3: pred = pred.mean(dim=1)
        pred_choice = pred.data.max(1)[1]

        for i in range(len(target)): new_class_acc[target[i],0]+=1
        for i in range(len(target)):
            if target[i]==pred_choice[i]: new_class_acc[pred_choice[i],1]+=1
        
    new_class_acc[:, 2] = new_class_acc[:, 1] / new_class_acc[:, 0]
    in_average=sum(new_class_acc[:, 1])/sum(new_class_acc[:, 0])
    cla_average=sum(new_class_acc[:, 2])/len(new_class_acc[:, 2])
    return in_average, cla_average
# ----------------------------------


def main(args):
    def log_string(str): logger.info(str); print(str)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --- Directory Setup ---
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/classification_modelnet40/').joinpath(args.log_dir or timestr)
    checkpoints_dir = exp_dir.joinpath('checkpoints/'); log_dir = exp_dir.joinpath('logs/')
    exp_dir.mkdir(exist_ok=True, parents=True); checkpoints_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Model"); logger.setLevel(logging.INFO)
    
    # --- Data Loading ---
    data_path = '../../data/modelnet40_preprocessed/' # 确认路径
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # --- Model Loading ---
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils_robust.py' % args.model.split('_')[0], str(exp_dir))
    
    classifier = model.get_model(num_class, 1, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    if not args.use_cpu: classifier = classifier.cuda(); criterion = criterion.cuda()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    
    # --- 修复 UnboundLocalError: 确保变量初始化 ---
    start_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    global_epoch = 0
    
    '''TRANING (HNR + FC-Loss)'''
    logger.info('Start training...')
    lambda_consistency = 0.25 # Consistency Weight 

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train_loss_sum = 0; train_correct = 0; train_total = 0

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            # 1. Generate weak and strong augmented versions (FC-Loss)
            points_np = points.data.numpy()
            points_weak = weak_augment(points_np)
            points_strong = strong_augment(points_np)
            
            points_weak = torch.Tensor(points_weak)
            points_strong = torch.Tensor(points_strong)

            if not args.use_cpu:
                points_weak, points_strong, target = points_weak.cuda(), points_strong.cuda(), target.cuda()
            
            # 2. Classification Loss (Weak View)
            pred_weak, _ = classifier(points_weak)
            classification_loss = F.nll_loss(pred_weak, target.long())
            
            # 3. Consistency Loss (Strong View vs. Detached Weak View)
            pred_strong, _ = classifier(points_strong)
            consistency_loss = F.kl_div(F.log_softmax(pred_strong, dim=1), 
                                        F.softmax(pred_weak.detach(), dim=1),
                                        reduction='batchmean')
            
            total_loss = classification_loss + lambda_consistency * consistency_loss
            
            total_loss.backward(); optimizer.step()
            
            # Statistics
            train_loss_sum += total_loss.item()
            pred_choice = pred_weak.data.max(1)[1]
            train_correct += pred_choice.eq(target.long().data).cpu().sum().item()
            train_total += points_weak.size(0)

        train_instance_acc = train_correct / train_total
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        
        if epoch == 300: 
            for param_group in optimizer.param_groups: param_group["lr"] = args.learning_rate
        
        scheduler.step()

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args, num_class=num_class)
            
            # --- 修复后的日志输出和保存逻辑 ---
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            
            if (class_acc >= best_class_acc): best_class_acc = class_acc
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, savepath)
            
            log_string('Best Instance Accuracy: %f at epoch %d' % (best_instance_acc, best_epoch))
        
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)