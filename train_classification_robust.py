"""
Robust RISurConv classification training script with partial consistency regularization
and ScanObjectNN-specific data augmentation

✅ Modified to support:
- G-SNR gating ablation: --gating_layers (0/2/5)
- EMA sensitivity: --ema_decay
- Gate strength sensitivity: --gate_alpha
- CP-SCL sensitivity: --lambda_consistency, --consistency_batch_ratio
- Paper-consistent strong aug: optional block erasing --use_block_erasing
✅ Also fixes two silent bugs:
- data_type branching (was always true)
- class_acc counting (was using pred index instead of target)
"""

import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.ScanObjectNNLoader import ScanObjectNN
import provider  # keep your provider import

# Add the project root to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='risurconv_cls_robust',
                        help='model name [default: risurconv_cls_robust]')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=450, type=int,
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
                        help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--use_cpu', action='store_true', default=False,
                        help='Whether to use CPU for training [default: False]')

    # ScanObjectNN params
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--use_uniform_sample', type=bool, default=True,
                        help='use uniform sampling')
    parser.add_argument('--data_type', type=str, default='hardest',
                        help='Data type [default: hardest]')
    parser.add_argument('--data_path', type=str, default='../../data/scanobjectnn/main_split/',
                        help='data path')

    # ===== Robust gating / EMA ablation =====
    parser.add_argument('--gating_layers', type=int, default=2,
                        help='Number of layers using G-SNR gating (0/2/5)')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                        help='EMA decay for G-SNR variance normalization')
    parser.add_argument('--gate_alpha', type=float, default=0.2,
                        help='Residual gate strength alpha')
    parser.add_argument('--use_sigmoid_gate', action='store_true',
                        help='Use sigmoid gate instead of 1/(1+var)')

    # ===== Consistency regularization =====
    parser.add_argument('--lambda_consistency', type=float, default=0.25,
                        help='Weight for CP-SCL consistency loss (0 disables)')
    parser.add_argument('--consistency_batch_ratio', type=float, default=0.2,
                        help='Fraction of batches applying consistency loss')

    # ===== Strong augmentation control =====
    parser.add_argument('--use_block_erasing', action='store_true',
                        help='Enable block erasing in strong augmentation')
    parser.add_argument('--block_erasing_prob', type=float, default=0.5,
                        help='Probability of applying block erasing')
    parser.add_argument('--seed', type=int, default=0,
                    help='Random seed (0 means no fixed seed)')


    return parser.parse_args()


def random_point_dropout(points, max_ratio=0.3):
    """Random point dropout to simulate occlusion / incomplete scanning.
    points: [B, N, C] (numpy)
    """
    B, N, C = points.shape
    drop_ratio = np.random.rand(B) * max_ratio
    for b in range(B):
        keepN = int(N * (1 - drop_ratio[b]))
        keepN = max(keepN, 1)
        idx = np.random.permutation(N)[:keepN]
        if keepN < N:
            pad_idx = np.random.randint(0, keepN, (N - keepN,))
            idx = np.concatenate([idx, pad_idx])
        points[b] = points[b][idx, :]
    return points


def random_block_erasing(points, p=0.5):
    """Random block erasing to simulate scanning occlusion.
    points: [B, N, C] (numpy)
    """
    if np.random.rand() > p:
        return points
    B, N, C = points.shape
    xyz = points[..., :3]
    min_xyz = np.min(xyz, axis=1, keepdims=True)
    max_xyz = np.max(xyz, axis=1, keepdims=True)
    center = min_xyz + (max_xyz - min_xyz) * np.random.rand(B, 1, 3)
    size = (max_xyz - min_xyz) * (0.2 + 0.3 * np.random.rand(B, 1, 3))

    mask = np.all((xyz > center - size / 2) & (xyz < center + size / 2), axis=-1)  # [B, N]

    for b in range(B):
        idx_keep = np.where(~mask[b])[0]
        if len(idx_keep) < N // 4:
            continue
        pad_idx = np.random.choice(idx_keep, N, replace=True)
        points[b] = points[b][pad_idx, :]
    return points


def weak_augment(points_np):
    """Weak augmentation: basic transformations (numpy in/out)."""
    points = points_np.copy()
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    return points


def strong_augment(points_np, args):
    """Strong augmentation: dropout + optional block erasing + jitter (numpy in/out)."""
    points = points_np.copy()
    points = random_point_dropout(points, max_ratio=0.2)
    if args.use_block_erasing:
        points = random_block_erasing(points, p=args.block_erasing_prob)

    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

    jitter = np.random.normal(loc=0.0, scale=0.01, size=points[:, :, 0:3].shape)
    points[:, :, 0:3] += jitter.astype(points.dtype)
    return points


@torch.no_grad()
def test(model, loader, args, num_class=15):
    """Test function to calculate both instance and class accuracy."""
    model = model.eval()
    class_acc = np.zeros((num_class, 3), dtype=np.float64)  # [total_count, correct_count, accuracy]

    for _, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points = torch.tensor(points.numpy(), dtype=torch.float32)
        if not args.use_cpu:
            points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)

        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]

        for i in range(len(target)):
            cls = int(target[i].item())
            class_acc[cls, 0] += 1
            if pred_choice[i] == target[i]:
                class_acc[cls, 1] += 1  # ✅ correct count should be on target class

    for i in range(num_class):
        if class_acc[i, 0] > 0:
            class_acc[i, 2] = class_acc[i, 1] / class_acc[i, 0]

    instance_average = class_acc[:, 1].sum() / max(class_acc[:, 0].sum(), 1.0)
    class_average = class_acc[:, 2].mean()
    return float(instance_average), float(class_average)


def main(args):
    def log_string(s):
        logger.info(s)
        print(s)
    
    if args.seed > 0:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = os.path.join(BASE_DIR, 'log', 'classification_robust')
    exp_dir = os.path.join(exp_dir, timestr if args.log_dir is None else args.log_dir)
    os.makedirs(exp_dir, exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (exp_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(str(args))

    # Data loading
    log_string('Load dataset ...')
    if args.data_type == 'OBJ_NOBG':
        data_path = '../../data/scanobjectnn/main_split_nobg/'
    elif args.data_type in ['hardest', 'OBJ_BG']:  # ✅ fix: was always true
        data_path = '../../data/scanobjectnn/main_split/'
    else:
        raise NotImplementedError(f'Unknown data_type: {args.data_type}')

    TRAIN_DATASET = ScanObjectNN(root=data_path, args=args, split='train')
    TEST_DATASET = ScanObjectNN(root=data_path, args=args, split='test')

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
        num_workers=10, drop_last=True, pin_memory=not args.use_cpu
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False,
        num_workers=10, pin_memory=not args.use_cpu
    )

    # Model loading
    num_class = 15
    model_mod = importlib.import_module(args.model)

    # ✅ Pass ablation params into model
    classifier = model_mod.get_model(
        num_class=num_class,
        n=1,
        gating_layers=args.gating_layers,
        ema_decay=args.ema_decay,
        gate_alpha=args.gate_alpha,
        use_sigmoid_gate=args.use_sigmoid_gate,
    )
    criterion = model_mod.get_loss()

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
        optimizer = optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.decay_rate
        )
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.lr_decay
    )

    # Checkpoint dirs
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger.info('Start training...')

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        # warm-up hook (keep your original behavior)
        if epoch == 300:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate

        classifier.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        consistency_loss_sum = 0.0
        consistency_applied_cnt = 0

        for _, (points, target) in tqdm(
            enumerate(trainDataLoader, 0),
            total=len(trainDataLoader),
            smoothing=0.9
        ):
            # Decide whether to apply consistency this batch
            apply_consistency = (
                args.lambda_consistency > 0.0 and
                np.random.rand() < args.consistency_batch_ratio
            )

            points_weak = points.numpy()
            points_strong = points.numpy()

            points_weak = weak_augment(points_weak)
            points_strong = strong_augment(points_strong, args)

            points_weak = torch.tensor(points_weak, dtype=torch.float32)
            points_strong = torch.tensor(points_strong, dtype=torch.float32)

            if not args.use_cpu:
                points_weak = points_weak.cuda(non_blocking=True)
                points_strong = points_strong.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_weak, _ = classifier(points_weak)
            classification_loss = F.nll_loss(pred_weak, target.long())

            total_loss = classification_loss

            if apply_consistency:
                pred_strong, _ = classifier(points_strong)

                consistency_loss = F.kl_div(
                    F.log_softmax(pred_strong, dim=1),
                    F.softmax(pred_weak.detach(), dim=1),
                    reduction='batchmean'
                )

                total_loss = classification_loss + args.lambda_consistency * consistency_loss
                consistency_loss_sum += float(consistency_loss.item())
                consistency_applied_cnt += 1

            total_loss.backward()
            optimizer.step()

            train_loss_sum += float(total_loss.item())
            pred_choice = pred_weak.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).sum().item()
            train_correct += correct
            train_total += points_weak.size(0)
            global_step += 1

        train_instance_acc = train_correct / max(train_total, 1)
        train_loss = train_loss_sum / max(len(trainDataLoader), 1)
        avg_consistency_loss = (
            consistency_loss_sum / max(consistency_applied_cnt, 1)
            if consistency_applied_cnt > 0 else 0.0
        )

        log_string(f'Train Instance Accuracy: {train_instance_acc:.6f}, '
                   f'Train Loss: {train_loss:.6f}, '
                   f'Avg Consistency Loss (applied): {avg_consistency_loss:.6f}, '
                   f'Applied Batches: {consistency_applied_cnt}/{len(trainDataLoader)}')
        log_string('lr: %f' % optimizer.param_groups[0]['lr'])

        # Evaluation
        with torch.no_grad():
            test_instance_acc, test_class_acc = test(classifier.eval(), testDataLoader, args, num_class=num_class)

            if test_class_acc >= best_class_acc:
                best_class_acc = test_class_acc

            log_string('Test Instance Accuracy: %f, Test Class Accuracy: %f' % (test_instance_acc, test_class_acc))
            log_string('Best Instance Accuracy: %f, Best Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if test_instance_acc >= best_instance_acc:
                best_instance_acc = test_instance_acc
                best_epoch = epoch + 1

                savepath = os.path.join(checkpoints_dir, 'best_model.pth')
                log_string('Saving best model at epoch %d...' % best_epoch)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_instance_acc,
                    'class_acc': test_class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
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
        'args': vars(args),
    }
    torch.save(state, savepath)

    log_string('Final Best Instance Accuracy: %f at epoch %d' % (best_instance_acc, best_epoch))


if __name__ == '__main__':
    args = parse_args()
    main(args)
