"""
train_partseg_fcreg.py
MODIFIED: ShapeNet Part Segmentation with Feature Consistency Regularization (FC-Reg).
用于验证 HNR + FC-Loss 机制对细粒度特征的提升效果。
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
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# --- 辅助变量和函数 (来自原始 train_partseg.py) ---
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1: m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda): return new_y.cuda()
    return new_y
    
# --- 核心增强函数 (用于 FC-Loss) ---
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
# ----------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='risurconv_part_seg_hnr', help='model name') # MODIFIED
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=250, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=5e-4, help='weight decay') # MODIFIED
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', type=bool, default=True, help='use normals')
    parser.add_argument('--step_size', type=int, default=30, help='decay step for lr decay') # MODIFIED
    parser.add_argument('--lr_decay', type=float, default=0.8, help='decay rate for lr decay') # MODIFIED
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/partseg/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('partseg_fcreg') # MODIFIED: 更改日志目录名称
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO); file_handler.setFormatter(formatter)
    logger.addHandler(file_handler); log_string('PARAMETER ...'); log_string(args)

    root = '../../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/' # 确认数据路径

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils_robust.py' % args.model.split('_')[0], str(exp_dir)) # Copy Robust Utils
    shutil.copy('./train_partseg_fcreg.py', str(exp_dir)) # Copy current script name

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    
    # --- Optimizer, Scheduler setup ---
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    
    # --- Initialization ---
    start_epoch = 0
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    
    # --- Load Checkpoint ---
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        
    
    # --- Main Training Loop ---
    lambda_consistency = 0.25 # Consistency Weight 

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # Adjust LR and BN momentum
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01: momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum)).train()

        '''learning one epoch (FC-Loss REGULARIZATION)'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            # 1. Generate weak and strong augmented versions
            points_np = points.data.numpy()
            points_weak = weak_augment(points_np)
            points_strong = strong_augment(points_np)
            
            points_weak, points_strong = torch.Tensor(points_weak), torch.Tensor(points_strong)
            points_weak, points_strong = points_weak.float().cuda(), points_strong.float().cuda()
            label, target = label.long().cuda(), target.long().cuda()

            # 2. Forward pass with weak augmentation (Teacher view)
            seg_pred_weak, trans_feat = classifier(points_weak, to_categorical(label, num_classes))
            seg_pred_weak_view = seg_pred_weak.contiguous().view(-1, num_part)
            target_flat = target.view(-1, 1)[:, 0]
            
            classification_loss = F.nll_loss(seg_pred_weak_view, target_flat)
            
            # 3. Consistency Loss (Student view)
            seg_pred_strong, _ = classifier(points_strong, to_categorical(label, num_classes))
            seg_pred_strong_view = seg_pred_strong.contiguous().view(-1, num_part)

            consistency_loss = F.kl_div(
                F.log_softmax(seg_pred_strong_view, dim=1), # Student
                F.softmax(seg_pred_weak_view.detach(), dim=1), # Detached Teacher
                reduction='mean'
            )
            
            total_loss = classification_loss + lambda_consistency * consistency_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Statistics (using weak prediction for training acc)
            pred_choice = seg_pred_weak_view.data.max(1)[1]
            correct = pred_choice.eq(target_flat.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0; total_seen = 0; total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]; shape_ious = {cat: [] for cat in seg_classes.keys()}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]: seg_label_to_cat[label] = cat
            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target); total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))
                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]; segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]: all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
            
            for cat in sorted(shape_ious.keys()): log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg IOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        
        # --- Saving Logic ---
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {'epoch': epoch, 'train_acc': train_instance_acc, 'test_acc': test_metrics['accuracy'],
                     'class_avg_iou': test_metrics['class_avg_iou'], 'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                     'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, savepath)
            log_string('Saving model....')
        
        if test_metrics['accuracy'] > best_acc: best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou: best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou: best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg IOU is: %.5f' % best_inctance_avg_iou)
        
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)