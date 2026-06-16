# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.mobilenet_v2 import mobilenet_v2



class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        # print('1')
        # x = self.shift1(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        # print('1')
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    @staticmethod
    def shift2(x, n_segment, fold_div=3, inplace=False):
        # print('1')
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-2, :fold] = x[:, 2:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

class TemporalShiftFC(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShiftFC, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold_div2 = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=8, inplace=self.inplace)
        # print('x1')
        # x = self.shift2(x, self.n_segment, fold_div=self.fold_div,inplace=self.inplace)
        # print('x1:',x.type)
        x = self.shift(x, self.n_segment, fold_div=4, inplace=self.inplace)
        # print('x2')
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div, inplace=False):
        # print('shift yici')
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            # --- Fusion Part (:fold) ---
            x_fuse_part = x[:, :, : 2 *fold, :, :]
            # F(t) component: multiply by 'a'
            out[:, :, : 2 *fold, :, :] =  x_fuse_part

            # F(t-1) component: take x[t, :, :fold] and put into out[t-1, :, :fold] (shift backward)
            if n_segment > 1:
                out[:, :-1, : 2 *fold, :, :] +=  0.8*x_fuse_part[:, 1:, :, :, :]

            # F(t+1) component: take x[t, :, :fold] and put into out[t+1, :, :fold] (shift forward)
                out[:, 1:, : 2 *fold, :, :] -=  0.8*x_fuse_part[:, :-1, :, :, :]

            # --- Rest part (2*fold -> end) ---
            # Simply copy the original values
            out[:, :, 2 *fold:, :, :] = x[:, :, 2 *fold:, :, :]

        return out.view(nt, c, h, w)


class TemporalShiftMultiScaleDiffFusion(nn.Module):
    def __init__(self,
                 net,
                 n_segment=3,
                 n_div=10,
                 inplace=False,
                 init_mode='medium'):
        super(TemporalShiftMultiScaleDiffFusion, self).__init__()

        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using multi-scale differential temporal fusion')

        assert self.n_segment >= 5, \
            'Multi-scale differential fusion needs n_segment >= 5 because it uses t±2 frames.'

        # ---------------------------------------------------------
        # 可学习标量权重：
        #
        # alpha 对应 [0, fold)：
        #   短期一阶中心差分
        #   X_t + alpha * (X_{t+1} - X_{t-1})
        #
        # beta 对应 [fold, 2*fold)：
        #   长期一阶中心差分
        #   X_t + beta * (X_{t+2} - X_{t-2})
        #
        # gamma 对应 [2*fold, 3*fold)：
        #   二阶中心差分
        #   X_t + gamma * (X_{t+2} - 2X_t + X_{t-2})
        # ---------------------------------------------------------

        if init_mode == 'zero':
            # 初始时等价于不做差分扰动：
            # Z_t = X_t
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.gamma = nn.Parameter(torch.tensor(0.0))

        elif init_mode == 'small':
            # 推荐初始化方式：
            # 对 MobileNetV2 比较稳，不会一开始破坏特征分布。
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.beta = nn.Parameter(torch.tensor(0.05))
            self.gamma = nn.Parameter(torch.tensor(0.05))

        elif init_mode == 'medium':
            # 稍强的运动建模初始化。
            self.alpha = nn.Parameter(torch.tensor(0.3))
            self.beta = nn.Parameter(torch.tensor(0.15))
            self.gamma = nn.Parameter(torch.tensor(0.15))

        elif init_mode == 'strong':
            # 较强差分初始化，不建议轻量网络一开始就使用。
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.25))
            self.gamma = nn.Parameter(torch.tensor(0.25))

        else:
            raise ValueError(
                "init_mode should be one of ['zero', 'small', 'medium', 'strong'], "
                "but got {}".format(init_mode)
            )

    def forward(self, x):
        x = self.shift(
            x,
            self.n_segment,
            fold_div=self.fold_div,
            inplace=self.inplace,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma
        )

        return self.net(x)

    @staticmethod
    def shift(x,
              n_segment,
              fold_div=8,
              inplace=False,
              alpha=None,
              beta=None,
              gamma=None):
        """
        x shape: [N*T, C, H, W]

        通道分组：

        1. [0, fold)：短期一阶运动
            Z_t =
                X_t
              + alpha * (X_{t+1} - X_{t-1})

        2. [fold, 2*fold)：长期一阶运动
            Z_t =
                X_t
              + beta * (X_{t+2} - X_{t-2})

        3. [2*fold, 3*fold)：二阶运动变化
            Z_t =
                X_t
              + gamma * (X_{t+2} - 2X_t + X_{t-2})

        4. [3*fold, C)：其余通道不变
            Z_t = X_t

        边界处理：
        - 短期一阶运动需要 t-1 和 t+1，因此有效位置为 t = 1, ..., T-2。
        - 长期一阶运动和二阶运动需要 t-2 和 t+2，因此有效位置为 t = 2, ..., T-3。
        - 边界位置默认保留原始 X_t，避免置零导致特征分布突变。
        """

        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        if inplace:
            raise NotImplementedError(
                'In-place version is not implemented for multi-scale differential fusion.'
            )
        else:
            # -----------------------------------------------------
            # 默认先复制原始特征。
            # 这样：
            # 1. 边界帧默认保持 X_t；
            # 2. 其余通道默认保持 X_t；
            # 3. 只需要覆盖三个差分通道组的有效时间位置。
            # -----------------------------------------------------
            out = x.clone()

            # -----------------------------------------------------
            # 参数准备
            # -----------------------------------------------------
            if alpha is None:
                alpha = torch.tensor(
                    0.1,
                    device=x.device,
                    dtype=x.dtype
                )
            else:
                alpha = alpha.to(device=x.device, dtype=x.dtype)

            if beta is None:
                beta = torch.tensor(
                    0.05,
                    device=x.device,
                    dtype=x.dtype
                )
            else:
                beta = beta.to(device=x.device, dtype=x.dtype)

            if gamma is None:
                gamma = torch.tensor(
                    0.05,
                    device=x.device,
                    dtype=x.dtype
                )
            else:
                gamma = gamma.to(device=x.device, dtype=x.dtype)

            if fold <= 0:
                return out.view(nt, c, h, w)

            # -----------------------------------------------------
            # 第 1 组：[0, fold)
            # 短期一阶运动：
            #
            #   Z_t = X_t + alpha * (X_{t+1} - X_{t-1})
            #
            # 有效位置：
            #   t = 1, ..., T-2
            # -----------------------------------------------------
            if fold > 0 and n_segment >= 3:
                out[:, 1:-1, :fold] = \
                    x[:, 1:-1, :fold] + \
                    alpha * (
                        x[:, 2:, :fold] -
                        x[:, :-2, :fold]
                    )

            # -----------------------------------------------------
            # 第 2 组：[fold, 2*fold)
            # 长期一阶运动：
            #
            #   Z_t = X_t + beta * (X_{t+2} - X_{t-2})
            #
            # 有效位置：
            #   t = 2, ..., T-3
            # -----------------------------------------------------
            c2_start = fold
            c2_end = min(2 * fold, c)

            if c2_start < c2_end and n_segment >= 5:
                out[:, 2:-2, c2_start:c2_end] = \
                    x[:, 2:-2, c2_start:c2_end] + \
                    beta * (
                        x[:, 4:, c2_start:c2_end] -
                        x[:, :-4, c2_start:c2_end]
                    )

            # -----------------------------------------------------
            # 第 3 组：[2*fold, 3*fold)
            # 二阶运动变化：
            #
            #   Z_t = X_t + gamma * (X_{t+2} - 2X_t + X_{t-2})
            #
            # 有效位置：
            #   t = 2, ..., T-3
            # -----------------------------------------------------
            c3_start = 2 * fold
            c3_end = min(3 * fold, c)

            if c3_start < c3_end and n_segment >= 5:
                out[:, 2:-2, c3_start:c3_end] = \
                    x[:, 2:-2, c3_start:c3_end] + \
                    gamma * (
                        x[:, 4:, c3_start:c3_end] -
                        2.0 * x[:, 2:-2, c3_start:c3_end] +
                        x[:, :-4, c3_start:c3_end]
                    )

            # -----------------------------------------------------
            # 第 4 组：[3*fold, C)
            # 其余通道不变：
            #
            #   Z_t = X_t
            #
            # 因为 out = x.clone()，所以这里不需要额外处理。
            # -----------------------------------------------------

        return out.view(nt, c, h, w)

    def get_fusion_weight(self):
        """
        用于查看当前融合权重。
        """
        return {
            'alpha': self.alpha.detach(),
            'beta': self.beta.detach(),
            'gamma': self.gamma.detach()
        }
    

def standard_tsm_shift(x, n_segment, fold_div=8):
    """
    标准 TSM shift.
    输入:
        x: [N*T, C, H, W]
    输出:
        out: [N*T, C, H, W]
    """
    nt, c, h, w = x.size()
    assert nt % n_segment == 0, "nt must be divisible by n_segment"

    n_batch = nt // n_segment
    x = x.contiguous().view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    out = torch.zeros_like(x)

    # 一部分通道向前移动：使用后一帧特征
    out[:, :-1, :fold, :, :] = x[:, 1:, :fold, :, :]

    # 一部分通道向后移动：使用前一帧特征
    out[:, 1:, fold:2 * fold, :, :] = x[:, :-1, fold:2 * fold, :, :]

    # 其余通道保持不变
    out[:, :, 2 * fold:, :, :] = x[:, :, 2 * fold:, :, :]

    return out.contiguous().view(nt, c, h, w)

def temporal_difference(x, n_segment):
    """
    中心时间差分.
    输入:
        x: [N*T, C, H, W]
    输出:
        diff: [N*T, C, H, W]
    """
    nt, c, h, w = x.size()
    assert nt % n_segment == 0, "nt must be divisible by n_segment"

    n_batch = nt // n_segment
    x = x.contiguous().view(n_batch, n_segment, c, h, w)

    diff = torch.zeros_like(x)

    if n_segment > 1:
        if n_segment == 2:
            diff[:, 0, :, :, :] = x[:, 1, :, :, :] - x[:, 0, :, :, :]
            diff[:, 1, :, :, :] = x[:, 1, :, :, :] - x[:, 0, :, :, :]
        else:
            # 中间帧：中心差分 x[t+1] - x[t-1]
            diff[:, 1:-1, :, :, :] = x[:, 2:, :, :, :] - x[:, :-2, :, :, :]

            # 边界帧：单边差分
            diff[:, 0, :, :, :] = x[:, 1, :, :, :] - x[:, 0, :, :, :]
            diff[:, -1, :, :, :] = x[:, -1, :, :, :] - x[:, -2, :, :, :]

    return diff.contiguous().view(nt, c, h, w)



class AdaptiveMultiBranchTSMFC(nn.Module):
    """
    方案三：多分支自适应融合 TSM

    Y = w0 * X + w1 * TSM(X) + w2 * Diff(X)

    w0, w1, w2 通过 softmax 得到。
    """

    def __init__(self, net, n_segment=8, n_div=8, inplace=False):
        super(AdaptiveMultiBranchTSMFC, self).__init__()

        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        # 三个分支权重：Identity, TSM, Diff
        # 初始化偏向 TSM，减少训练初期不稳定
        self.branch_logits = nn.Parameter(torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32))

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using Adaptive Multi-Branch TSM')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        identity = x

        x_tsm = self.shift_tsm(x, self.n_segment, self.fold_div, self.inplace)
        x_diff = self.shift_diff(x, self.n_segment)

        weights = torch.softmax(self.branch_logits, dim=0)

        out = weights[0] * identity + weights[1] * x_tsm + weights[2] * x_diff

        return self.net(out)

    @staticmethod
    def shift_tsm(x, n_segment, fold_div=8, inplace=False):
        if inplace:
            raise NotImplementedError
        return standard_tsm_shift(x, n_segment, fold_div)

    @staticmethod
    def shift_diff(x, n_segment):
        return temporal_difference(x, n_segment)


class TemporalShift2(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift2, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold_div2 = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=8, inplace=self.inplace)
        # print('x1')
        # x = self.shift2(x, self.n_segment, fold_div=self.fold_div,inplace=self.inplace)
        # print('x1:',x.type)
        x = self.shift(x, self.n_segment, fold_div=8, inplace=self.inplace)
        # print('x2')
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div, inplace=False):
        # print('shift yici')
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

            
        return out.view(nt, c, h, w)


class  TemporalShiftD(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShiftD, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold_div2 = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=8, inplace=self.inplace)
        # print('x1')
        # x = self.shift2(x, self.n_segment, fold_div=self.fold_div,inplace=self.inplace)
        # print('x1:',x.type)
        x = self.shift(x, self.n_segment, fold_div=8, inplace=self.inplace)
        # print('x2')
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div, inplace=False):
        # print('shift yici')
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            # out[:, :, :2 * fold] = x[:, :, :2 * fold]
            out[:, :-1, :fold] += x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] -= x[:, :-1, fold: 2 * fold]  # shift right

            # out[:, :, 2 * fold: 3 * fold] += out[:, :, 2 * fold: 3 * fold].roll(shifts=1, dims=3)  # shift up in space
            # out[:, :, 3 * fold: 4 * fold] -= out[:, :, 3 * fold: 4 * fold].roll(shifts=-1, dims=3)  # shift down in space
            # out[:, :, 4 * fold: 5 * fold] += out[:, :, 4 * fold: 5 * fold].roll(shifts=1, dims=4)  # shift left in space
            # out[:, :, 5 * fold: 6 * fold] -= out[:, :, 5 * fold: 6 * fold].roll(shifts=-1, dims=4)  # shift right in space

            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    def shift2(x, n_segment, fold_div2, inplace=False):
        # x = torch.from_numpy(x)
        print('x2:',x.type)
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div2
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

            out[:, :-1, :fold] += x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] += x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)
    


class TemporalShiftSoftmaxFusionAblation1(nn.Module):
    """
    Temporal Shift with Softmax Differential Fusion for Ablation Study.

    支持三种权重模式：

    1. weight_mode='fixed'
        固定权重，不参与训练。

    2. weight_mode='learnable_scalar'
        可学习标量权重。
        A 分支一组 alpha_logits: [3]
        B 分支一组 beta_logits:  [3]

    3. weight_mode='learnable_group'
        分组可学习标量权重。
        A 分支 alpha_logits: [num_groups, 3]
        B 分支 beta_logits:  [num_groups, 3]

    init_mode:
        'uniform': softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
        'tsm':     softmax([-2, -2, 2]) ≈ [0.0177, 0.0177, 0.9647]
        'diff':    softmax([1, 1, 0]) ≈ [0.4223, 0.4223, 0.1554]
    """

    def __init__(
        self,
        net,
        n_segment=3,
        n_div=8,
        inplace=False,
        weight_mode='learnable_scalar',
        init_mode='tsm',
        num_groups=4
    ):
        super(TemporalShiftSoftmaxFusionAblation1, self).__init__()

        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        self.weight_mode = weight_mode
        self.init_mode = init_mode
        self.num_groups = num_groups

        assert self.n_segment >= 3, \
            'Softmax differential fusion needs n_segment >= 3 because it uses t±2 frames.'

        assert weight_mode in ['fixed', 'learnable_scalar', 'learnable_group'], \
            "weight_mode should be one of ['fixed', 'learnable_scalar', 'learnable_group']"

        assert init_mode in ['uniform', 'tsm', 'diff'], \
            "init_mode should be one of ['uniform', 'tsm', 'diff']"

        assert num_groups in [1, 2, 4, 8], \
            "num_groups should be one of [1, 2, 4, 8]"

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using softmax weighted differential temporal shift')
        print('=> Weight mode: {}'.format(self.weight_mode))
        print('=> Init mode: {}'.format(self.init_mode))
        print('=> Num groups: {}'.format(self.num_groups))

        base_logits = self._get_init_logits(init_mode)

        # ---------------------------------------------------------
        # 1. 固定权重
        # ---------------------------------------------------------
        if weight_mode == 'fixed':
            if num_groups == 1:
                alpha_logits = base_logits.clone()
                beta_logits = base_logits.clone()
            else:
                alpha_logits = base_logits.view(1, 3).repeat(num_groups, 1)
                beta_logits = base_logits.view(1, 3).repeat(num_groups, 1)

            self.register_buffer('alpha_logits', alpha_logits)
            self.register_buffer('beta_logits', beta_logits)

        # ---------------------------------------------------------
        # 2. 可学习标量权重
        # ---------------------------------------------------------
        elif weight_mode == 'learnable_scalar':
            self.num_groups = 1

            self.alpha_logits = nn.Parameter(base_logits.clone())
            self.beta_logits = nn.Parameter(base_logits.clone())

        # ---------------------------------------------------------
        # 3. 分组可学习标量权重
        # ---------------------------------------------------------
        elif weight_mode == 'learnable_group':
            alpha_logits = base_logits.view(1, 3).repeat(num_groups, 1)
            beta_logits = base_logits.view(1, 3).repeat(num_groups, 1)

            self.alpha_logits = nn.Parameter(alpha_logits)
            self.beta_logits = nn.Parameter(beta_logits)

    @staticmethod
    def _get_init_logits(init_mode):
        """
        返回 softmax 前的 logits。
        """

        if init_mode == 'uniform':
            return torch.zeros(3, dtype=torch.float32)

        elif init_mode == 'tsm':
            # softmax([-2, -2, 2]) ≈ [0.0177, 0.0177, 0.9647]
            return torch.tensor([-2.0, -2.0, 2.0], dtype=torch.float32)

        elif init_mode == 'diff':
            # softmax([1, 1, 0]) ≈ [0.4223, 0.4223, 0.1554]
            return torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

        else:
            raise ValueError(
                "init_mode should be one of ['uniform', 'tsm', 'diff'], "
                "but got {}".format(init_mode)
            )

    def forward(self, x):
        """
        x shape: [N*T, C, H, W]
        """

        # ---------------------------------------------------------
        # 根据不同权重模式生成 softmax 后的融合权重
        # ---------------------------------------------------------

        if self.weight_mode in ['fixed', 'learnable_scalar']:
            alpha_weight = F.softmax(self.alpha_logits, dim=-1)
            beta_weight = F.softmax(self.beta_logits, dim=-1)

        elif self.weight_mode == 'learnable_group':
            alpha_weight = F.softmax(self.alpha_logits, dim=-1)
            beta_weight = F.softmax(self.beta_logits, dim=-1)

        else:
            raise ValueError('Unknown weight_mode: {}'.format(self.weight_mode))

        x = self.shift(
            x,
            self.n_segment,
            fold_div=self.fold_div,
            inplace=self.inplace,
            alpha_weight=alpha_weight,
            beta_weight=beta_weight,
            weight_mode=self.weight_mode,
            num_groups=self.num_groups
        )

        return self.net(x)

    @staticmethod
    def shift(
        x,
        n_segment,
        fold_div=8,
        inplace=False,
        alpha_weight=None,
        beta_weight=None,
        weight_mode='learnable_scalar',
        num_groups=4
    ):
        """
        x shape: [N*T, C, H, W]

        A 通道，前向差分：
            Z_t^A =
                w_a1 * (X_{t+1}^A - X_t^A)
              + w_a2 * (X_{t+2}^A - X_{t+1}^A)
              + w_a3 * X_{t+2}^A

        B 通道，后向差分：
            Z_t^B =
                w_b1 * (X_{t-1}^B - X_t^B)
              + w_b2 * (X_{t-2}^B - X_{t-1}^B)
              + w_b3 * X_{t-2}^B

        C 通道：
            Z_t^C = X_t^C
        """

        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        if inplace:
            raise NotImplementedError(
                'In-place version is not implemented for softmax differential fusion.'
            )

        out = torch.zeros_like(x)

        # 说明：
        # 如果希望边界位置保持原始特征，则保留下面这句。
        # 如果希望边界位置为 0，则删除下面这句。
        out[:, :, :2 * fold] = x[:, :, :2 * fold]

        # C 通道不移位
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        if fold <= 0:
            return out.view(nt, c, h, w)

        # ---------------------------------------------------------
        # 权重准备
        # ---------------------------------------------------------

        if alpha_weight is None:
            alpha_weight = torch.tensor(
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
                device=x.device,
                dtype=x.dtype
            )

        if beta_weight is None:
            beta_weight = torch.tensor(
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
                device=x.device,
                dtype=x.dtype
            )

        alpha_weight = alpha_weight.to(device=x.device, dtype=x.dtype)
        beta_weight = beta_weight.to(device=x.device, dtype=x.dtype)

        # ---------------------------------------------------------
        # 情况一：固定权重或者全局可学习标量权重
        # alpha_weight shape: [3]
        # beta_weight shape:  [3]
        # ---------------------------------------------------------
        if weight_mode in ['fixed', 'learnable_scalar']:

            wa1, wa2, wa3 = alpha_weight[0], alpha_weight[1], alpha_weight[2]
            wb1, wb2, wb3 = beta_weight[0], beta_weight[1], beta_weight[2]

            # A 通道：t = 0, ..., T-3
            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, 2:, :fold]

            # B 通道：t = 2, ..., T-1
            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                wb2 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        # ---------------------------------------------------------
        # 情况二：分组可学习标量权重
        # alpha_weight shape: [num_groups, 3]
        # beta_weight shape:  [num_groups, 3]
        # ---------------------------------------------------------
        elif weight_mode == 'learnable_group':

            # 防止 num_groups 大于 fold
            actual_groups = min(num_groups, fold)

            # 要求每组至少有一个通道
            # 如果 fold 不能被 actual_groups 整除，最后一组吃掉剩余通道
            group_size = fold // actual_groups

            for g in range(actual_groups):
                c_start = g * group_size

                if g == actual_groups - 1:
                    c_end = fold
                else:
                    c_end = (g + 1) * group_size

                wa1, wa2, wa3 = alpha_weight[g, 0], alpha_weight[g, 1], alpha_weight[g, 2]
                wb1, wb2, wb3 = beta_weight[g, 0], beta_weight[g, 1], beta_weight[g, 2]

                # -----------------------------
                # A 通道第 g 组
                # A 通道整体范围: [0, fold)
                # 当前组范围: [c_start, c_end)
                # -----------------------------
                out[:, :-2, c_start:c_end] = \
                    wa1 * (x[:, 1:-1, c_start:c_end] - x[:, :-2, c_start:c_end]) + \
                    wa2 * (x[:, 2:, c_start:c_end] - x[:, 1:-1, c_start:c_end]) + \
                    wa3 * x[:, 2:, c_start:c_end]

                # -----------------------------
                # B 通道第 g 组
                # B 通道整体范围: [fold, 2*fold)
                # 当前组范围: [fold+c_start, fold+c_end)
                # -----------------------------
                b_start = fold + c_start
                b_end = fold + c_end

                out[:, 2:, b_start:b_end] = \
                    wb1 * (x[:, 1:-1, b_start:b_end] - x[:, 2:, b_start:b_end]) + \
                    wb2 * (x[:, :-2, b_start:b_end] - x[:, 1:-1, b_start:b_end]) + \
                    wb3 * x[:, :-2, b_start:b_end]

        else:
            raise ValueError('Unknown weight_mode: {}'.format(weight_mode))

        return out.view(nt, c, h, w)

    def get_fusion_weight(self):
        """
        查看当前 softmax 后的融合权重。

        返回：
            alpha_weight, beta_weight

        对于 fixed / learnable_scalar:
            alpha_weight shape: [3]
            beta_weight shape:  [3]

        对于 learnable_group:
            alpha_weight shape: [num_groups, 3]
            beta_weight shape:  [num_groups, 3]
        """

        alpha_weight = F.softmax(self.alpha_logits, dim=-1).detach()
        beta_weight = F.softmax(self.beta_logits, dim=-1).detach()

        return alpha_weight, beta_weight



class TemporalShiftSoftmaxFusionVariants(nn.Module):
    def __init__(
        self,
        net,
        n_segment=3,
        n_div=8,
        inplace=False,
        init_mode='tsm',
        variant=2,
        boundary_mode='zero'
    ):
        """
        variant:
            1: 原始差分方向 + 原始移位项使用 X_{t+1} / X_{t-1}
            2: 原始差分方向 + 原始项使用 X_t
            3: 第一项差分反向，第二项保持，原始移位项使用 X_{t+2} / X_{t-2}
            4: 第一项保持，第二项差分反向，原始移位项使用 X_{t+2} / X_{t-2}
            5: 第一项和第二项都反向，原始移位项使用 X_{t+2} / X_{t-2}

        boundary_mode:
            'keep': 边界位置保留原始输入特征
            'zero': 边界位置置 0
        """

        super(TemporalShiftSoftmaxFusionVariants, self).__init__()

        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.variant = variant
        self.boundary_mode = boundary_mode

        if inplace:
            print('=> Using in-place shift...')

        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using softmax weighted differential temporal shift')
        print('=> Using variant: {}'.format(self.variant))
        print('=> Using boundary mode: {}'.format(self.boundary_mode))

        assert self.n_segment >= 3, \
            'Softmax differential fusion needs n_segment >= 3 because it uses t±2 frames.'

        assert self.variant in [0, 1, 2, 3, 4, 5], \
            'variant should be one of [0, 1, 2, 3, 4, 5]'

        assert self.boundary_mode in ['keep', 'zero'], \
            "boundary_mode should be 'keep' or 'zero'"

        # ---------------------------------------------------------
        # 可学习标量权重：
        # alpha_logits 对应 A 通道
        # beta_logits  对应 B 通道
        #
        # softmax 后：
        # alpha = [w_a1, w_a2, w_a3]
        # beta  = [w_b1, w_b2, w_b3]
        # ---------------------------------------------------------

        if init_mode == 'uniform':
            self.alpha_logits = nn.Parameter(torch.zeros(3))
            self.beta_logits = nn.Parameter(torch.zeros(3))

        elif init_mode == 'tsm':
            # 初始时更偏向第三项，即原始特征项
            self.alpha_logits = nn.Parameter(torch.tensor([-2.0, -2.0, 2.0]))
            self.beta_logits = nn.Parameter(torch.tensor([-2.0, -2.0, 2.0]))

        elif init_mode == 'diff':
            # 初始时更偏向差分项
            self.alpha_logits = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
            self.beta_logits = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))

        else:
            raise ValueError(
                "init_mode should be one of ['uniform', 'tsm', 'diff'], "
                "but got {}".format(init_mode)
            )

    def forward(self, x):
        alpha_weight = F.softmax(self.alpha_logits, dim=0)
        beta_weight = F.softmax(self.beta_logits, dim=0)

        x = self.shift(
            x,
            n_segment=self.n_segment,
            fold_div=self.fold_div,
            inplace=self.inplace,
            alpha_weight=alpha_weight,
            beta_weight=beta_weight,
            variant=self.variant,
            boundary_mode=self.boundary_mode
        )

        return self.net(x)

    @staticmethod
    def shift(
        x,
        n_segment,
        fold_div=8,
        inplace=False,
        alpha_weight=None,
        beta_weight=None,
        variant=2,
        boundary_mode='zero'
    ):
        """
        x shape: [N*T, C, H, W]

        通道划分：
            A 通道: [:fold]
            B 通道: [fold:2*fold]
            C 通道: [2*fold:]

        A 通道使用未来信息：
            t, t+1, t+2

        B 通道使用过去信息：
            t, t-1, t-2

        C 通道保持不变：
            Z_t^C = X_t^C
        """

        if inplace:
            raise NotImplementedError(
                'In-place version is not implemented for softmax differential fusion.'
            )

        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        # ---------------------------------------------------------
        # 边界处理
        # ---------------------------------------------------------
        if boundary_mode == 'keep':
            # 边界位置保留原始特征
            out = x.clone()
        elif boundary_mode == 'zero':
            # 边界位置置 0
            out = torch.zeros_like(x)
        else:
            raise ValueError("boundary_mode should be 'keep' or 'zero'")

        # C 通道始终保留原始特征
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # ---------------------------------------------------------
        # 权重准备
        # ---------------------------------------------------------
        if alpha_weight is None:
            alpha_weight = torch.tensor(
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
                device=x.device,
                dtype=x.dtype
            )
        else:
            alpha_weight = alpha_weight.to(device=x.device, dtype=x.dtype)

        if beta_weight is None:
            beta_weight = torch.tensor(
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
                device=x.device,
                dtype=x.dtype
            )
        else:
            beta_weight = beta_weight.to(device=x.device, dtype=x.dtype)

        wa1, wa2, wa3 = alpha_weight[0], alpha_weight[1], alpha_weight[2]
        wb1, wb2, wb3 = beta_weight[0], beta_weight[1], beta_weight[2]

        if fold <= 0:
            return out.view(nt, c, h, w)

        # =========================================================
        # Variant 0
        # =========================================================
        # A:
        # Z_t^A =
        #     wa1 * (X_{t+1}^A - X_t^A)
        #   + wa2 * (X_{t+2}^A - X_{t+1}^A)
        #   + wa3 * X_t^A
        #
        # B:
        # Z_t^B =
        #     wb1 * (X_t^B - X_{t-1}^B)
        #   + wb2 * (X_{t-1}^B - X_{t-2}^B)
        #   + wb3 * X_t^B
        # =========================================================
        if variant == 0:
            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, :-2, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 2:, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb2 * (x[:, 1:-1, fold:2 * fold] - x[:, :-2, fold:2 * fold]) + \
                wb3 * x[:, 2:, fold:2 * fold]


        # =========================================================
        # Variant 1
        # =========================================================
        # A:
        # Z_t^A =
        #     wa1 * (X_{t+1}^A - X_t^A)
        #   + wa2 * (X_{t+2}^A - X_{t+1}^A)
        #   + wa3 * X_{t+1}^A
        #
        # B:
        # Z_t^B =
        #     wb1 * (X_t^B - X_{t-1}^B)
        #   + wb2 * (X_{t-1}^B - X_{t-2}^B)
        #   + wb3 * X_{t-1}^B
        # =========================================================
        elif variant == 1:
            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, 1:-1, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 2:, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb2 * (x[:, 1:-1, fold:2 * fold] - x[:, :-2, fold:2 * fold]) + \
                wb3 * x[:, 1:-1, fold:2 * fold]


        # =========================================================
        # Variant 2              FF
        # =========================================================
        # A:
        # Z_t^A =
        #     wa1 * (X_{t+1}^A - X_t^A)
        #   + wa2 * (X_{t+2}^A - X_{t+1}^A)
        #   + wa3 * X_{t+2}^A
        #
        # B:
        # Z_t^B =
        #     wb1 * (X_t^B - X_{t-1}^B)
        #   + wb2 * (X_{t-1}^B - X_{t-2}^B)
        #   + wb3 * X_{t-2}^B
        # =========================================================
        elif variant == 2:
            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 2:, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb2 * (x[:, 1:-1, fold:2 * fold] - x[:, :-2, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        # =========================================================
        # Variant 3                   FB
        # =========================================================
        # A:
        # Z_t^A =
        #     wa1 * ( X_{t+1}^A- X_t^A)
        #   + wa2 * (X_{t+2}^A - X_{t+1}^A)
        #   + wa3 * X_{t+2}^A
        #
        # B:
        # Z_t^B =
        #     wb1 * (X_{t-1}^B - X_t^B)
        #   + wb2 * (X_{t-2}^B) - X_{t-1}^B)
        #   + wb3 * X_{t-2}^B
        # =========================================================
        elif variant == 3:
            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                wb2 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        # =========================================================
        # Variant 4               BF
        # =========================================================
        # A:
        # Z_t^A =
        #     wa1 * (X_t^A - X_{t+1}^A)
        #   + wa2 * (X_{t+1}^A - X_{t+2}^A)
        #   + wa3 * X_{t+2}^A
        #
        # B:
        # Z_t^B =
        #     wb1 * (X_t^B - X_{t-1}^B)
        #   + wb2 * (X_{t-1}^B- X_{t-2}^B)
        #   + wb3 * X_{t-2}^B
        # =========================================================
        elif variant == 4:
            out[:, :-2, :fold] = \
                wa1 * (x[:, :-2, :fold] - x[:, 1:-1, :fold]) + \
                wa2 * (x[:, 1:-1, :fold] - x[:, 2:, :fold]) + \
                wa3 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 2:, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb2 * (x[:, 1:-1, fold:2 * fold] - x[:, :-2, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        # =========================================================
        # Variant 5                  BB
        # =========================================================
        # A:
        # Z_t^A =
        #     wa1 * (X_t^A - X_{t+1}^A)
        #   + wa2 * (X_{t+1}^A - X_{t+2}^A)
        #   + wa3 * X_{t+2}^A
        #
        # B:
        # Z_t^B =
        #     wb1 * (X_{t-1}^B - X_t^B)
        #   + wb2 * (X_{t-2}^B - X_{t-1}^B)
        #   + wb3 * X_{t-2}^B
        # =========================================================
        elif variant == 5:
            out[:, :-2, :fold] = \
                wa1 * (x[:, :-2, :fold] - x[:, 1:-1, :fold]) + \
                wa2 * (x[:, 1:-1, :fold] - x[:, 2:, :fold]) + \
                wa3 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                wb2 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        else:
            raise ValueError('variant should be one of [1, 2, 3, 4, 5]')

        return out.view(nt, c, h, w)

    def get_fusion_weight(self):
        """
        查看当前 softmax 后的融合权重。
        """
        alpha_weight = F.softmax(self.alpha_logits, dim=0).detach()
        beta_weight = F.softmax(self.beta_logits, dim=0).detach()

        return alpha_weight, beta_weight




class TemporalShiftSoftmaxFusionAblation2(nn.Module):
    def __init__(
        self,
        net,
        n_segment=3,
        n_div=8,
        inplace=False,
        init_mode='tsm',
        ablation_mode=7,
        fixed_alpha_weight=None,
        fixed_beta_weight=None
    ):
        super(TemporalShiftSoftmaxFusionAblation2, self).__init__()

        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.ablation_mode = ablation_mode

        assert self.n_segment >= 3, \
            'This module needs n_segment >= 3 because it uses t±2 frames.'

        assert self.ablation_mode in [1, 2, 3, 4, 5, 6, 7, 8], \
            'ablation_mode should be one of [1, 2, 3, 4, 5, 6, 7, 8].'

        if inplace:
            print('=> Using in-place shift...')
            raise NotImplementedError(
                'In-place version is not implemented for this ablation module.'
            )

        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using ablation mode: {}'.format(self.ablation_mode))

        # ---------------------------------------------------------
        # mode 8 使用可学习权重：
        # alpha_logits -> [wa1, wa2, wa3]
        # beta_logits  -> [wb1, wb2, wb3]
        #
        # wa1/wb1: 第一差分项
        # wa2/wb2: 第二差分项
        # wa3/wb3: shift 项
        # ---------------------------------------------------------
        if init_mode == 'uniform':
            self.alpha_logits = nn.Parameter(torch.zeros(3))
            self.beta_logits = nn.Parameter(torch.zeros(3))

        elif init_mode == 'tsm':
            self.alpha_logits = nn.Parameter(torch.tensor([-2.0, -2.0, 2.0]))
            self.beta_logits = nn.Parameter(torch.tensor([-2.0, -2.0, 2.0]))

        elif init_mode == 'diff':
            self.alpha_logits = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
            self.beta_logits = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))

        else:
            raise ValueError(
                "init_mode should be one of ['uniform', 'tsm', 'diff'], "
                "but got {}".format(init_mode)
            )

        # ---------------------------------------------------------
        # mode 7 使用固定权重
        # 默认固定为 [1/3, 1/3, 1/3]
        # ---------------------------------------------------------
        if fixed_alpha_weight is None:
            fixed_alpha_weight = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        if fixed_beta_weight is None:
            fixed_beta_weight = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        self.register_buffer(
            'fixed_alpha_weight',
            torch.tensor(fixed_alpha_weight, dtype=torch.float32)
        )

        self.register_buffer(
            'fixed_beta_weight',
            torch.tensor(fixed_beta_weight, dtype=torch.float32)
        )

    def forward(self, x):
        if self.ablation_mode == 8:
            alpha_weight = F.softmax(self.alpha_logits, dim=0)
            beta_weight = F.softmax(self.beta_logits, dim=0)
        else:
            alpha_weight = None
            beta_weight = None

        x = self.shift(
            x,
            n_segment=self.n_segment,
            fold_div=self.fold_div,
            inplace=self.inplace,
            ablation_mode=self.ablation_mode,
            alpha_weight=alpha_weight,
            beta_weight=beta_weight,
            fixed_alpha_weight=self.fixed_alpha_weight,
            fixed_beta_weight=self.fixed_beta_weight
        )

        return self.net(x)

    @staticmethod
    def shift(
        x,
        n_segment,
        fold_div=8,
        inplace=False,
        ablation_mode=7,
        alpha_weight=None,
        beta_weight=None,
        fixed_alpha_weight=None,
        fixed_beta_weight=None
    ):
        """
        x shape: [N*T, C, H, W]

        A 通道：
            第一差分: X_{t+1} - X_t
            第二差分: X_{t+2} - X_{t+1}
            shift项 : X_{t+2}

        B 通道：
            第一差分: X_{t-1} - X_t
            第二差分: X_{t-2} - X_{t-1}
            shift项 : X_{t-2}

        C 通道：
            Z_t^C = X_t^C
        """

        if inplace:
            raise NotImplementedError(
                'In-place version is not implemented.'
            )

        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)

        # ---------------------------------------------------------
        # C 通道：不移位，保持原始特征
        # ---------------------------------------------------------
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        if fold == 0:
            return out.view(nt, c, h, w)

        # =========================================================
        # Mode 1
        # 只有 shift 项
        #
        # A:
        #   Z_t^A = X_{t+2}^A
        #
        # B:
        #   Z_t^B = X_{t-2}^B
        # =========================================================
        if ablation_mode == 1:
            out[:, :-2, :fold] = \
                x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                x[:, :-2, fold:2 * fold]

        # =========================================================
        # Mode 2
        # 只有第一差分
        #
        # A:
        #   Z_t^A = X_{t+1}^A - X_t^A
        #
        # B:
        #   Z_t^B = X_{t-1}^B - X_t^B
        # =========================================================
        elif ablation_mode == 2:
            out[:, :-2, :fold] = \
                x[:, 1:-1, :fold] - x[:, :-2, :fold]

            out[:, 2:, fold:2 * fold] = \
                x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]

        # =========================================================
        # Mode 3
        # 只有第二差分
        #
        # A:
        #   Z_t^A = X_{t+2}^A - X_{t+1}^A
        #
        # B:
        #   Z_t^B = X_{t-2}^B - X_{t-1}^B
        # =========================================================
        elif ablation_mode == 3:
            out[:, :-2, :fold] = \
                x[:, 2:, :fold] - x[:, 1:-1, :fold]

            out[:, 2:, fold:2 * fold] = \
                x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]

        # =========================================================
        # Mode 4
        # shift + 第一差分
        #
        # A:
        #   Z_t^A =
        #       0.5 * (X_{t+1}^A - X_t^A)
        #     + 0.5 * X_{t+2}^A
        #
        # B:
        #   Z_t^B =
        #       0.5 * (X_{t-1}^B - X_t^B)
        #     + 0.5 * X_{t-2}^B
        # =========================================================
        elif ablation_mode == 4:
            out[:, :-2, :fold] = \
                0.5 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                0.5 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                0.5 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                0.5 * x[:, :-2, fold:2 * fold]

        # =========================================================
        # Mode 5
        # shift + 第二差分
        #
        # A:
        #   Z_t^A =
        #       0.5 * (X_{t+2}^A - X_{t+1}^A)
        #     + 0.5 * X_{t+2}^A
        #
        # B:
        #   Z_t^B =
        #       0.5 * (X_{t-2}^B - X_{t-1}^B)
        #     + 0.5 * X_{t-2}^B
        # =========================================================
        elif ablation_mode == 5:
            out[:, :-2, :fold] = \
                0.5 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                0.5 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                0.5 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                0.5 * x[:, :-2, fold:2 * fold]

        # =========================================================
        # Mode 6
        # 第一差分 + 第二差分
        #
        # A:
        #   Z_t^A =
        #       0.5 * (X_{t+1}^A - X_t^A)
        #     + 0.5 * (X_{t+2}^A - X_{t+1}^A)
        #
        # B:
        #   Z_t^B =
        #       0.5 * (X_{t-1}^B - X_t^B)
        #     + 0.5 * (X_{t-2}^B - X_{t-1}^B)
        # =========================================================
        elif ablation_mode == 6:
            out[:, :-2, :fold] = \
                0.5 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                0.5 * (x[:, 2:, :fold] - x[:, 1:-1, :fold])

            out[:, 2:, fold:2 * fold] = \
                0.5 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                0.5 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold])

        # =========================================================
        # Mode 7
        # shift + 第一差分 + 第二差分
        # 固定权重
        #
        # A:
        #   Z_t^A =
        #       wa1 * (X_{t+1}^A - X_t^A)
        #     + wa2 * (X_{t+2}^A - X_{t+1}^A)
        #     + wa3 * X_{t+2}^A
        #
        # B:
        #   Z_t^B =
        #       wb1 * (X_{t-1}^B - X_t^B)
        #     + wb2 * (X_{t-2}^B - X_{t-1}^B)
        #     + wb3 * X_{t-2}^B
        # =========================================================
        elif ablation_mode == 7:
            fixed_alpha_weight = fixed_alpha_weight.to(
                device=x.device,
                dtype=x.dtype
            )
            fixed_beta_weight = fixed_beta_weight.to(
                device=x.device,
                dtype=x.dtype
            )

            wa1, wa2, wa3 = fixed_alpha_weight[0], fixed_alpha_weight[1], fixed_alpha_weight[2]
            wb1, wb2, wb3 = fixed_beta_weight[0], fixed_beta_weight[1], fixed_beta_weight[2]

            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                wb2 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        # =========================================================
        # Mode 8
        # shift + 第一差分 + 第二差分
        # Softmax 可学习权重
        #
        # A:
        #   Z_t^A =
        #       wa1 * (X_{t+1}^A - X_t^A)
        #     + wa2 * (X_{t+2}^A - X_{t+1}^A)
        #     + wa3 * X_{t+2}^A
        #
        # B:
        #   Z_t^B =
        #       wb1 * (X_{t-1}^B - X_t^B)
        #     + wb2 * (X_{t-2}^B - X_{t-1}^B)
        #     + wb3 * X_{t-2}^B
        # =========================================================
        elif ablation_mode == 8:
            alpha_weight = alpha_weight.to(device=x.device, dtype=x.dtype)
            beta_weight = beta_weight.to(device=x.device, dtype=x.dtype)

            wa1, wa2, wa3 = alpha_weight[0], alpha_weight[1], alpha_weight[2]
            wb1, wb2, wb3 = beta_weight[0], beta_weight[1], beta_weight[2]

            out[:, :-2, :fold] = \
                wa1 * (x[:, 1:-1, :fold] - x[:, :-2, :fold]) + \
                wa2 * (x[:, 2:, :fold] - x[:, 1:-1, :fold]) + \
                wa3 * x[:, 2:, :fold]

            out[:, 2:, fold:2 * fold] = \
                wb1 * (x[:, 1:-1, fold:2 * fold] - x[:, 2:, fold:2 * fold]) + \
                wb2 * (x[:, :-2, fold:2 * fold] - x[:, 1:-1, fold:2 * fold]) + \
                wb3 * x[:, :-2, fold:2 * fold]

        else:
            raise ValueError(
                "ablation_mode should be one of [1, 2, 3, 4, 5, 6, 7, 8], "
                "but got {}".format(ablation_mode)
            )

        return out.view(nt, c, h, w)

    def get_fusion_weight(self):
        """
        查看 mode 8 的 softmax 可学习权重。
        """
        alpha_weight = F.softmax(self.alpha_logits, dim=0).detach()
        beta_weight = F.softmax(self.beta_logits, dim=0).detach()

        return alpha_weight, beta_weight




class WeightedTemporalShiftD(nn.Module):
    """
    Weighted TemporalShiftD
    结合 TemporalShiftD 的 t±2 思路，并引入可学习权重。
    
    A 通道：当前帧 + 未来两帧
    B 通道：当前帧 + 过去两帧
    C 通道：保持不变
    """
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(WeightedTemporalShiftD, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        # 可学习权重
        self.forward_shift_weight = nn.Parameter(torch.tensor(0.8))
        self.backward_shift_weight = nn.Parameter(torch.tensor(0.8))
        self.preserve_weight = nn.Parameter(torch.tensor(0.5))

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using Weighted TemporalShiftD')

    def forward(self, x):
        x = self.shift(
            x,
            self.n_segment,
            fold_div=self.fold_div,
            inplace=self.inplace,
            forward_shift_weight=self.forward_shift_weight,
            backward_shift_weight=self.backward_shift_weight,
            preserve_weight=self.preserve_weight
        )
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False,
              forward_shift_weight=None,
              backward_shift_weight=None,
              preserve_weight=None):
        nt, c, h, w = x.size()

        if nt % n_segment != 0:
            raise ValueError(
                'Input batch dimension {} is not divisible by n_segment {}.'
                .format(nt, n_segment)
            )

        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        if inplace:
            raise NotImplementedError

        out = torch.zeros_like(x)

        wf = torch.sigmoid(forward_shift_weight.to(device=x.device, dtype=x.dtype))
        wb = torch.sigmoid(backward_shift_weight.to(device=x.device, dtype=x.dtype))
        wp = torch.sigmoid(preserve_weight.to(device=x.device, dtype=x.dtype))

        # 当前帧保留部分
        out[:, :, :2 * fold] = wp * x[:, :, :2 * fold]

        # A 通道：融合未来两帧
        if fold > 0:
            out[:, :-2, :fold] += wf * x[:, 2:, :fold]

        # B 通道：融合过去两帧
        if 2 * fold <= c:
            out[:, 2:, fold:2 * fold] += wb * x[:, :-2, fold:2 * fold]

        # C 通道：保持不变
        if 2 * fold < c:
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)

class SoftmaxTemporalKernelShift(nn.Module):
    """
    Softmax Temporal Kernel Shift
    使用 softmax 学习时间核权重，形成 3-tap temporal kernel。

    A 通道：
        Z_t^A = w_a1 * X_t + w_a2 * X_{t+1} + w_a3 * X_{t+2}

    B 通道：
        Z_t^B = w_b1 * X_t + w_b2 * X_{t-1} + w_b3 * X_{t-2}

    C 通道：
        Z_t^C = X_t
    """
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, init_mode='uniform'):
        super(SoftmaxTemporalKernelShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using Softmax Temporal Kernel Shift')

        if init_mode == 'uniform':
            self.alpha_logits = nn.Parameter(torch.zeros(3))
            self.beta_logits = nn.Parameter(torch.zeros(3))
        elif init_mode == 'tsm':
            # 更偏向远端位置，初始行为更接近“移位”
            self.alpha_logits = nn.Parameter(torch.tensor([-2.0, -2.0, 2.0]))
            self.beta_logits = nn.Parameter(torch.tensor([-2.0, -2.0, 2.0]))
        else:
            raise ValueError(
                "init_mode should be one of ['uniform', 'tsm'], "
                "but got {}".format(init_mode)
            )

    def forward(self, x):
        alpha_weight = F.softmax(self.alpha_logits, dim=0)
        beta_weight = F.softmax(self.beta_logits, dim=0)

        x = self.shift(
            x,
            self.n_segment,
            fold_div=self.fold_div,
            inplace=self.inplace,
            alpha_weight=alpha_weight,
            beta_weight=beta_weight
        )
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False,
              alpha_weight=None, beta_weight=None):
        nt, c, h, w = x.size()

        if nt % n_segment != 0:
            raise ValueError(
                'Input batch dimension {} is not divisible by n_segment {}.'
                .format(nt, n_segment)
            )

        if n_segment < 3:
            raise ValueError(
                'Softmax Temporal Kernel Shift needs n_segment >= 3.'
            )

        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        if inplace:
            raise NotImplementedError

        out = torch.zeros_like(x)

        if alpha_weight is None:
            alpha_weight = torch.tensor(
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
                device=x.device,
                dtype=x.dtype
            )
        else:
            alpha_weight = alpha_weight.to(device=x.device, dtype=x.dtype)

        if beta_weight is None:
            beta_weight = torch.tensor(
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
                device=x.device,
                dtype=x.dtype
            )
        else:
            beta_weight = beta_weight.to(device=x.device, dtype=x.dtype)

        wa1, wa2, wa3 = alpha_weight[0], alpha_weight[1], alpha_weight[2]
        wb1, wb2, wb3 = beta_weight[0], beta_weight[1], beta_weight[2]

        # A 通道：当前 + 未来1 + 未来2
        if fold > 0:
            out[:, :-2, :fold] = (
                wa1 * x[:, :-2, :fold] +
                wa2 * x[:, 1:-1, :fold] +
                wa3 * x[:, 2:, :fold]
            )

        # B 通道：当前 + 过去1 + 过去2
        if 2 * fold <= c:
            out[:, 2:, fold:2 * fold] = (
                wb1 * x[:, 2:, fold:2 * fold] +
                wb2 * x[:, 1:-1, fold:2 * fold] +
                wb3 * x[:, :-2, fold:2 * fold]
            )

        # C 通道：保持不变
        if 2 * fold < c:
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)

    def get_kernel_weight(self):
        alpha_weight = F.softmax(self.alpha_logits, dim=0).detach()
        beta_weight = F.softmax(self.beta_logits, dim=0).detach()
        return alpha_weight, beta_weight


class WeightedTemporalShift(nn.Module):
    """权重增强的时间偏移模块，直接在时间偏移过程中引入权重融合"""
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(WeightedTemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        
        # 可学习的权重参数
        self.forward_shift_weight = nn.Parameter(torch.ones(1) * 0.8)  # 前向偏移强度
        self.backward_shift_weight = nn.Parameter(torch.ones(1) * 0.8)  # 后向偏移强度
        self.preserve_weight = nn.Parameter(torch.ones(1) * 0.5)  # 保留原始特征的权重
        
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        print('=> Using weighted temporal shift...')

    def forward(self, x):
        # 应用权重增强的时间偏移
        x = self.weighted_shift(x, self.n_segment, fold_div=self.fold_div, 
                               forward_weight=self.forward_shift_weight,
                               backward_weight=self.backward_shift_weight,
                               preserve_weight=self.preserve_weight)
        x = self.weighted_shift(x, self.n_segment, fold_div=self.fold_div, 
                               forward_weight=self.forward_shift_weight,
                               backward_weight=self.backward_shift_weight,
                               preserve_weight=self.preserve_weight)
        return self.net(x)

    @staticmethod
    def weighted_shift(x, n_segment, fold_div=8, forward_weight=0.8, backward_weight=0.8, preserve_weight=0.5, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            
            # 带权重的前向移动
            if fold > 0:
                shifted_forward = x[:, 1:, :fold].clone()  # 移动到前一位置的特征
                preserved_forward = x[:, :-1, :fold].clone()  # 保留在原位置的特征
                # 融合移动特征和保留特征
                out[:, :-1, :fold] = (
                    torch.sigmoid(forward_weight) * shifted_forward + 
                    (1 - torch.sigmoid(forward_weight)) * preserved_forward
                )
            
            # 带权重的后向移动
            if 2 * fold <= c:
                shifted_backward = x[:, :-1, fold:2*fold].clone()  # 移动到后一位置的特征
                preserved_backward = x[:, 1:, fold:2*fold].clone()  # 保留在原位置的特征
                # 融合移动特征和保留特征
                out[:, 1:, fold:2*fold] = (
                    torch.sigmoid(backward_weight) * shifted_backward + 
                    (1 - torch.sigmoid(backward_weight)) * preserved_backward
                )
            
          
            if 2 * fold < c:
                out[:, :, 2*fold:] =  x[:, :, 2*fold:]

        return out.view(nt, c, h, w)





class TemporalShiftFB(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShiftFB, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.n_div = n_div
        self.inplace = inplace
        
        if inplace:
            raise NotImplementedError
        
        # 添加可学习的缩放参数
        self.scale_short = nn.Parameter(torch.ones(1))
        self.scale_long = nn.Parameter(torch.ones(1))
        
        # 添加一个可学习的通道权重
        self.channel_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 添加调试信息
        # print(f"Input x shape: {x.shape}")
        
        C_fold = x.size(1) // 8
        x_fold = x[:, :2*C_fold, :, :]  # 前1/4通道用于多尺度移位
        x_rest = x[:, 2*C_fold:, :, :]  # 其余3/4通道保持不变
        
        # print(f"x_fold shape: {x_fold.shape}")
        # print(f"x_rest shape: {x_rest.shape}")
        
        # 两个分支都处理相同的前1/4通道
        shifted_x1 = self.shift_scale_1(x_fold, self.n_segment)
        shifted_x2 = self.shift_scale_2(x_fold, self.n_segment)
        
        # 高级差分引导融合
        fused_features = self.advanced_differential_guided_fusion(
            shifted_x1, shifted_x2, x_fold
        )
        
        # 确保维度匹配
        if fused_features.shape[1] != x_rest.shape[1]:
            # 如果通道数不匹配，调整
            if fused_features.shape[1] < x_rest.shape[1]:
                # padding
                pad_size = x_rest.shape[1] - fused_features.shape[1]
                fused_features = torch.nn.functional.pad(fused_features, (0, 0, 0, 0, 0, pad_size))
            else:
                # truncate
                fused_features = fused_features[:, :x_rest.shape[1], :, :]
        
        # 将融合后的移位特征与非移位特征拼接
        output = torch.cat([fused_features, x_rest], dim=1)
        
        return self.net(output)
    
    def advanced_differential_guided_fusion(self, shifted_x1, shifted_x2, original_x):
        """
        高级差分引导融合：考虑局部和全局运动信息
        """
        nt, c, h, w = original_x.size()
        n_batch = nt // self.n_segment
        
        # 调试信息
        # print(f"advanced_differential_guided_fusion - nt:{nt}, c:{c}, h:{h}, w:{w}, n_batch:{n_batch}")
        # print(f"shifted_x1 shape: {shifted_x1.shape}")
        # print(f"shifted_x2 shape: {shifted_x2.shape}")
        
        original_x = original_x.view(n_batch, self.n_segment, c, h, w)
        
        # 重塑 shifted tensors
        shifted_x1_reshaped = shifted_x1.view(n_batch, self.n_segment, c, h, w)
        shifted_x2_reshaped = shifted_x2.view(n_batch, self.n_segment, c, h, w)
        
        # 确保两个 shifted tensors 的通道数一致
        if shifted_x1_reshaped.shape[2] != shifted_x2_reshaped.shape[2]:
            min_c = min(shifted_x1_reshaped.shape[2], shifted_x2_reshaped.shape[2])
            # print(f"Channel mismatch: {shifted_x1_reshaped.shape[2]} vs {shifted_x2_reshaped.shape[2]}, truncating to {min_c}")
            shifted_x1_reshaped = shifted_x1_reshaped[:, :, :min_c, :, :]
            shifted_x2_reshaped = shifted_x2_reshaped[:, :, :min_c, :, :]
            c = min_c
        
        # 1. 计算多层次运动强度（返回 [n_batch, c] 维度）
        global_motion = self.compute_global_motion_intensity(original_x, c)  # [n_batch, c]
        local_motion = self.compute_local_motion_intensity(original_x, c)    # [n_batch, c]
        
        # 2. 结合全局和局部运动强度
        motion_intensity = global_motion * 0.7 + local_motion * 0.3  # [n_batch, c]
        
        # 3. 为两个尺度生成自适应权重 [n_batch, c]
        weight_short = torch.sigmoid(motion_intensity * self.scale_short)
        weight_long = torch.sigmoid(motion_intensity * self.scale_long * 0.8)
        
        # 扩展权重维度以匹配 [n_batch, c, 1, 1, 1]
        weight_short_expanded = weight_short.view(n_batch, c, 1, 1, 1)
        weight_long_expanded = weight_long.view(n_batch, c, 1, 1, 1)
        
        # 融合
        fused_output = (weight_short_expanded * shifted_x1_reshaped + 
                       weight_long_expanded * shifted_x2_reshaped)
        
        return fused_output.view(nt, c, h, w)
    
    def compute_global_motion_intensity(self, x, target_c):
        """
        计算全局运动强度
        Args:
            x: [n_batch, n_segment, c, h, w]
            target_c: 目标通道数
        Returns:
            motion: [n_batch, target_c]
        """
        n_batch, n_segment, c, h, w = x.shape
        
        if n_segment <= 1:
            return torch.zeros(n_batch, target_c, device=x.device)
        
        # 计算时间差分 [n_batch, n_segment-1, c, h, w]
        forward_diff = torch.abs(x[:, 1:, :, :, :] - x[:, :-1, :, :, :])
        
        # 全局平均池化，得到 [n_batch, n_segment-1, c]
        global_motion = forward_diff.mean(dim=[3, 4])  # 空间维度池化
        
        # 在时间维度上平均，得到 [n_batch, c]
        global_motion = global_motion.mean(dim=1)
        
        # 调整通道数到 target_c
        if global_motion.shape[1] != target_c:
            if global_motion.shape[1] < target_c:
                # 重复通道
                repeat_times = target_c // global_motion.shape[1] + 1
                global_motion = global_motion.repeat(1, repeat_times)[:, :target_c]
            else:
                # 截断通道
                global_motion = global_motion[:, :target_c]
        
        return global_motion
    
    def compute_local_motion_intensity(self, x, target_c):
        """
        计算局部运动强度（考虑空间变化）
        Args:
            x: [n_batch, n_segment, c, h, w]
            target_c: 目标通道数
        Returns:
            motion: [n_batch, target_c]
        """
        n_batch, n_segment, c, h, w = x.shape
        
        if n_segment <= 1:
            return torch.zeros(n_batch, target_c, device=x.device)
        
        # 1. 计算空间梯度（局部纹理变化）
        # 水平梯度 [n_batch, n_segment, c, h, w-1]
        grad_h = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        # 垂直梯度 [n_batch, n_segment, c, h-1, w]
        grad_w = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        
        # 空间运动强度 [n_batch, n_segment, c]
        spatial_motion_h = grad_h.mean(dim=[3, 4])  # 平均空间维度
        spatial_motion_w = grad_w.mean(dim=[3, 4])
        spatial_motion = (spatial_motion_h + spatial_motion_w) / 2
        
        # 2. 计算时间差分 [n_batch, n_segment-1, c, h, w]
        time_diff = torch.abs(x[:, 1:, :, :, :] - x[:, :-1, :, :, :])
        # 时间运动强度 [n_batch, n_segment-1, c]
        temporal_motion = time_diff.mean(dim=[3, 4])
        
        # 3. 在时间维度上平均
        spatial_motion = spatial_motion.mean(dim=1)  # [n_batch, c]
        temporal_motion = temporal_motion.mean(dim=1)  # [n_batch, c]
        
        # 4. 组合时空运动
        combined_motion = (spatial_motion + temporal_motion) / 2
        
        # 调整通道数到 target_c
        if combined_motion.shape[1] != target_c:
            if combined_motion.shape[1] < target_c:
                # 重复通道
                repeat_times = target_c // combined_motion.shape[1] + 1
                combined_motion = combined_motion.repeat(1, repeat_times)[:, :target_c]
            else:
                # 截断通道
                combined_motion = combined_motion[:, :target_c]
        
        return combined_motion
    
    @staticmethod
    def shift_scale_1(x, n_segment):
        """短距离移位（相邻帧）"""
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        
        out = x.clone()
        
        if n_segment > 1:
            # 前向差分
            out[:, :-1, :, :, :] = out[:, :-1, :, :, :] + x[:, 1:, :, :, :]
            out[:, 1:, :, :, :] = out[:, 1:, :, :, :] - x[:, :-1, :, :, :]
        
        return out.view(nt, c, h, w)
    
    @staticmethod
    def shift_scale_2(x, n_segment):
        """长距离移位（间隔帧）"""
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        
        out = x.clone()
        
        if n_segment > 2:
            # 长距离差分
            out[:, :-2, :, :, :] = out[:, :-2, :, :, :] + x[:, 2:, :, :, :]
            out[:, 2:, :, :, :] = out[:, 2:, :, :, :] - x[:, :-2, :, :, :]
        
        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    # if isinstance(net, torchvision.models.ResNet):
    # if isinstance(net, torchvision.models._resnet):
    # if isinstance(net, MODEL.eca_resnet1.ResNet1):
    if isinstance(net, archs.mobilenet_v2.mobilenet_v2):
    # if isinstance(net, MODEL.ACmix.ResNet.ResNet_ImageNet.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShiftFC(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShiftFC(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')




