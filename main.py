# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time  # 只保留这一个time导入
import shutil
import json
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from datetime import timedelta  # 添加这行，用于格式化时间显示

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool
from tensorboardX import SummaryWriter
from pytorchtools import EarlyStopping
from torchsummary import summary

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)

    print('args.root_path:', args.root_path)
    print('args.train_list:', args.train_list)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['4GCDDFN', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)
    print(args.shift)

    check_rootfolders()

    # 创建保存准确率、损失、violence指标和混淆矩阵的文件
    acc_save_path = os.path.join(args.root_log, args.store_name, 'validation_accuracy.txt')
    loss_save_path = os.path.join(args.root_log, args.store_name, 'validation_loss.txt')
    violence_metrics_save_path = os.path.join(args.root_log, args.store_name, 'violence_metrics.txt')
    confusion_matrix_save_path = os.path.join(args.root_log, args.store_name, 'confusion_matrix.txt')

    # 初始化文件，写入表头
    with open(acc_save_path, 'w') as f:
        f.write('epoch\tprecision@1\n')

    with open(loss_save_path, 'w') as f:
        f.write('epoch\tloss\n')

    with open(violence_metrics_save_path, 'w') as f:
        f.write('epoch\tprecision\trecall\tf1\n')

    with open(confusion_matrix_save_path, 'w') as f:
        f.write('epoch\tTP\tFN\tFP\tTN\n')

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)


    # ========== 在这里插入模型测量代码（模型刚创建完） ==========
    print("\n" + "="*60)
    print("MODEL BENCHMARK (Before DataParallel)")
    print("="*60)
    
    # 设置模型为评估模式进行测量
    model.eval()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_mb = total_params * 4 / (1024**2)
    
    print(f"📊 Model: {args.arch}")
    print(f"   Total parameters: {total_params/1e6:.2f}M ({params_mb:.2f}MB)")
    print(f"   Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # 计算 GFLOPs（需要 thop 库）
    # try:
    #     from thop import profile
    #     # 构造输入：batch=1, channels=3, num_segments=8, height=224, width=224
    #     input_tensor = torch.randn(1, 3* args.num_segments, 224, 224)
    #     flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    #     flops_g = flops / 1e9
    #     print(f"   GFLOPs: {flops_g:.2f}")
    # except ImportError:
    #     print("   GFLOPs: thop not installed, skip")
    # except Exception as e:
    #     print(f"   GFLOPs: calculation failed - {e}")
    # 3. 计算推理内存占用（关键修改：加上 no_grad）
    if torch.cuda.is_available():
        device = 'cuda'
        model = model.to(device)
        input_tensor = torch.randn(1, 3* args.num_segments, 224, 224).to(device)
        
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # 预热（使用 no_grad）
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        torch.cuda.reset_peak_memory_stats(device)
        # 正式测量（使用 no_grad）
        with torch.no_grad():
            _ = model(input_tensor)
        torch.cuda.synchronize()
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        
        print(f"   Memory (inference): {memory_mb:.2f} MB")
        model = model.cpu()
    else:
        print("   Memory: CUDA not available, skip")
    
    print("="*60 + "\n")
    # ========== 测量代码结束 ==========

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    # params = filter(lambda p: p.requires_grad, model.parameters())
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)


    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()


    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(policies, lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    print(model)
    summary(model, (8, 3, 224, 224))

    # 初始化early_stopping对象
    # patience = 20
    # early_stopping = EarlyStopping(patience, verbose=True)



    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        # print('=====MODEL=====')
        # for k, v in model_dict.items():
        #     print(k)
        # # print(model_dict)
        # print('=====MODEL=====')
        replace_dict = []

        sd2 = {}
        sd1 = {}

        for k, v in sd.items():
            # print(k)
            o = 'module.' + k
            sd2[o] = v

        # print('=====load=====')

        for k, v in sd2.items():
            # print(k)
            ch = '.net'
            if '.net' in k:
                # print(k)
                # o = k.replace(ch, '')
                # sd1[o] = v
                sd1[k] = v
            else:
                sd1[k] = v

        # print('=====load=====')
        for k, v in sd1.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd1 and k.replace('.net', '') in sd1:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        # print('replace_dict:', replace_dict)
        for k, k_new in replace_dict:
            sd1[k_new] = sd1.pop(k)
        keys1 = set(list(sd1.keys()))
        keys2 = set(list(model_dict.keys()))
        # print('keys1:', keys1)
        # print('keys2:', keys2)
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('args.dataset:', args.dataset)
            print('args.tune_from:', args.tune_from)
            print('=> New dataset, do not load fc weights')
            # sd1 = {k: v for k, v in sd1.items() if 'fc' not in k}
            for k, v in list(sd1.items()):
                if 'classifier' in k:
                    print('k:', k)
            sd1 = {k: v for k, v in sd1.items() if 'classifier' not in k}
        if args.dataset in args.tune_from:  # new dataset
            print('args.dataset:', args.dataset)
            print('args.tune_from:', args.tune_from)
            print('=> load fc weights')

        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd1 = {k: v for k, v in sd1.items() if 'conv1.weight' not in k}
        model_dict.update(sd1)
        # print(type(model_dict))
        # print(model_dict)
        model.load_state_dict(model_dict)




    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    save_path = '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/quanzhong/tsm-mobilenetv2.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    image_list = {'0':'violence', '1':'Noviolence'}
    cla_dict = dict((key, val) for key, val in image_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    # ========== 添加计时开始 =========
    training_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
    print(f"Total epochs to train: {args.epochs - args.start_epoch}")
    print(f"{'='*60}\n")
    # ========== 计时添加结束 ==========

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, valid_loss, metrics = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # 保存每个epoch的验证准确率和损失到单独的文件
            with open(acc_save_path, 'a') as f:
                f.write(f'{epoch + 1}\t{prec1:.3f}\n')

            with open(loss_save_path, 'a') as f:
                f.write(f'{epoch + 1}\t{valid_loss:.5f}\n')

            # 保存 violence 类 Precision / Recall / F1
            with open(violence_metrics_save_path, 'a') as f:
                f.write(
                    f"{epoch + 1}\t"
                    f"{metrics['precision']:.6f}\t"
                    f"{metrics['recall']:.6f}\t"
                    f"{metrics['f1']:.6f}\n"
                )

            # 保存混淆矩阵 TP / FN / FP / TN
            with open(confusion_matrix_save_path, 'a') as f:
                f.write(
                    f"{epoch + 1}\t"
                    f"{metrics['TP']}\t"
                    f"{metrics['FN']}\t"
                    f"{metrics['FP']}\t"
                    f"{metrics['TN']}\n"
                )

            print(f'[Saved] Epoch {epoch + 1}: Validation Acc = {prec1:.3f}%, Loss = {valid_loss:.5f}')

            print(
                f"[Violence] Epoch {epoch + 1}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"TP={metrics['TP']}, FN={metrics['FN']}, FP={metrics['FP']}, TN={metrics['TN']}"
            )

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

    # ========== 添加计时结束 ==========
    total_training_time = time.time() - training_start_time
    total_minutes = total_training_time / 60
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"Total training time: {total_minutes:.2f} minutes")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Formatted time: {timedelta(seconds=total_training_time)}")
    print(f"Best accuracy achieved: {best_prec1:.3f}%")
    print(f"{'='*60}")
    
    # 将训练时间写入日志文件
    log_training.write(f"\n{'='*50}\n")
    log_training.write(f"Total training time: {total_minutes:.2f} minutes ({total_training_time:.2f} seconds)\n")
    log_training.write(f"Best accuracy: {best_prec1:.3f}%\n")
    log_training.write(f"{'='*50}\n")
    log_training.flush()
    
    # 可选：记录到TensorBoard
    tf_writer.add_text('Training/TotalTime', f'{total_minutes:.2f} minutes', 0)
    tf_writer.add_text('Training/BestAccuracy', f'{best_prec1:.3f}%', 0)
    # ========== 计时添加结束 ==========

    #     early_stopping(valid_loss, model)
    #     # 若满足 early stopping 要求
    #     if early_stopping.early_stop and epoch >= 50:
    #         print("Early stopping")
    #         # 结束模型训练
    #         break
    # # 获得 early stopping 时的模型参数
    # model.load_state_dict(torch.load('checkpoint.pt'))

# start_time = time.time()

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))


        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,  lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)

    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


# end_time = time.time()
# training_time = end_time - start_time
# print(f"Total training took {training_time:.2f} seconds.")


def compute_binary_metrics(all_targets, all_preds, positive_class=0):
    """
    计算二分类任务中 positive_class 的 Precision / Recall / F1 和混淆矩阵。

    对你的任务:
        positive_class = 0 表示 violence
        0: violence
        1: Noviolence

    混淆矩阵:
                    Pred violence     Pred Noviolence
    True violence        TP                 FN
    True Noviolence      FP                 TN
    """

    all_targets = torch.tensor(all_targets)
    all_preds = torch.tensor(all_preds)

    TP = ((all_targets == positive_class) & (all_preds == positive_class)).sum().item()
    FN = ((all_targets == positive_class) & (all_preds != positive_class)).sum().item()
    FP = ((all_targets != positive_class) & (all_preds == positive_class)).sum().item()
    TN = ((all_targets != positive_class) & (all_preds != positive_class)).sum().item()

    eps = 1e-12

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # 保存整个验证集的预测结果和真实标签
    all_preds = []
    all_targets = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # ==============================
            # 新增：收集预测类别和真实标签
            # output: [batch_size, num_class]
            # pred:   [batch_size]
            # ==============================
            pred = torch.argmax(output, dim=1)

            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_targets.extend(target.detach().cpu().numpy().tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output_str = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1,))
                print(output_str)
                if log is not None:
                    log.write(output_str + '\n')
                    log.flush()

    # ==============================
    # 新增：计算 violence 类指标
    # 0: violence
    # 1: Noviolence
    # ==============================
    metrics = compute_binary_metrics(all_targets, all_preds, positive_class=0)

    output_str = ('Testing Results: Prec@1 {top1.avg:.3f}  Loss {loss.avg:.5f}'
                  .format(top1=top1, loss=losses))
    print(output_str)

    metrics_str = (
        '\nViolence class metrics:\n'
        '  Precision: {precision:.4f}\n'
        '  Recall:    {recall:.4f}\n'
        '  F1-score:  {f1:.4f}\n'
        '\nConfusion Matrix:\n'
        '                 Pred violence    Pred Noviolence\n'
        'True violence        {TP:<8d}         {FN:<8d}\n'
        'True Noviolence      {FP:<8d}         {TN:<8d}\n'
    ).format(**metrics)

    print(metrics_str)

    if log is not None:
        log.write(output_str + '\n')
        log.write(metrics_str + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)

        # 新增：写入 TensorBoard
        tf_writer.add_scalar('violence/precision', metrics['precision'], epoch)
        tf_writer.add_scalar('violence/recall', metrics['recall'], epoch)
        tf_writer.add_scalar('violence/f1', metrics['f1'], epoch)

        tf_writer.add_scalar('confusion_matrix/TP', metrics['TP'], epoch)
        tf_writer.add_scalar('confusion_matrix/FN', metrics['FN'], epoch)
        tf_writer.add_scalar('confusion_matrix/FP', metrics['FP'], epoch)
        tf_writer.add_scalar('confusion_matrix/TN', metrics['TN'], epoch)

    return top1.avg, losses.avg, metrics



def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        # decay = 0.98 ** ((epoch + 1) // 2)
        # decay = 0.9 ** ((epoch + 1) // 2)
        # decay = 0.95 ** ((epoch + 1) // 2)
        # decay = 0.85 ** ((epoch + 1) // 2) #1.7wan88.5 lr=0.001
        # decay = 0.85 ** ((epoch + 1) // 2) #1.7wan88.5 lr=0.0005
        decay = 0.8 ** ((epoch + 1) // 2) #1.8,86.25
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

# end_time = time.time()
# training_time = end_time - start_time
# print(f"Total training took {training_time:.2f} seconds.")

if __name__ == '__main__':
    main()
