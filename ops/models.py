# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_

from MODEL.eca_resnet1 import eca_resnet50

from archs.mobilenet_v2 import mobilenet_v2
from archs.ghostnet import ghostnet
from archs.ghostnetv2 import ghostnetv2
# from torchvision.models import resnext50_32x4d

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=True, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.print_spec = print_spec

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        # feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        feature_dim = 1280
        # print('self.dropout:', self.dropout)
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            self.base_model = eca_resnet50()
            # self.base_model = ACmix_ResNet()
            # self.base_model = resnext50_32x4d()
            # self.base_model = resnet50()
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            # self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)
            self.base_model = mobilenet_v2(False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import  TemporalShiftSoftmaxFusionAblation1
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        # print('self.print_spec:', self.print_spec)
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] =  TemporalShiftSoftmaxFusionAblation1(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':                                                                                                                                                                                                      
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'ghostnet':
            from archs.ghostnet import ghostnet, GhostBottleneck
            from ops.temporal_shift import TemporalShiftMultiScaleDiffFusion
            import torch
            import os

            print('=> using GhostNet')

            # =========================================================
            # 0. 手动选择 TSF 插入位置
            # =========================================================
            # 可选：
            # 'before_ghost1' : TSF → ghost1
            # 'after_ghost1'  : ghost1 → TSF
            # 'whole_block'   : TSF → 整个 GhostBottleneck
            ghost_tsf_position = 'after_ghost1'

            # 建议先只插中后层，不要全插
            # GhostNet blocks 索引一般为 0~9，其中 9 通常是最后 ConvBnAct
            # 推荐从轻量配置开始：[6, 8]
            # 稍强一些：[4, 6, 8]
            # 更强但风险更大：[3, 4, 5, 6, 7, 8]
            tsm_stages = [4, 6, 8]

            print('=> GhostNet TSF position:', ghost_tsf_position)
            print('=> GhostNet TSF stages:', tsm_stages)

            #========
            # 1. 创建 GhostNet
            # =========================================================
            # 保持 num_classes=1000，便于加载 ImageNet 预训练
            self.base_model = ghostnet(num_classes=1000, width=1.0)

            # GhostNet 的最后分类层是 self.classifier
            self.base_model.last_layer_name = 'classifier'

            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            # GhostNet forward 里通常使用 global_pool
            # 这里保留 avgpool 是为了兼容原工程，不影响 GhostNet forward
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            # =========================================================
            # 2. 加载 GhostNet ImageNet 预训练权重
            #    注意：必须先加载原始 GhostNet 权重，再插入 TSF
            # =========================================================
            ghostnet_pretrained_path = r'/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/pretrained/state_dict_73.98.pth'

            if self.pretrain == 'imagenet' and os.path.isfile(ghostnet_pretrained_path):
                print('=> loading GhostNet ImageNet pretrained weights from:')
                print('   {}'.format(ghostnet_pretrained_path))

                checkpoint = torch.load(ghostnet_pretrained_path, map_location='cpu')

                # 兼容不同保存格式
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                print('=> Raw GhostNet checkpoint keys example:')
                for idx, k in enumerate(state_dict.keys()):
                    if idx < 10:
                        print('   ', k)
                    else:
                        break

                new_state_dict = {}

                for k, v in state_dict.items():
                    # 去掉 DataParallel 前缀
                    if k.startswith('module.'):
                        k = k[7:]

                    # 去掉 model. 前缀
                    if k.startswith('model.'):
                        k = k[6:]

                    # 去掉 base_model. 前缀
                    if k.startswith('base_model.'):
                        k = k[len('base_model.'):]

                    new_state_dict[k] = v

                # 删除 ImageNet 分类头
                # 后面 _prepare_tsn(num_class) 会重新构建任务分类头
                new_state_dict.pop('classifier.weight', None)
                new_state_dict.pop('classifier.bias', None)

                load_msg = self.base_model.load_state_dict(new_state_dict, strict=False)

                print('=> GhostNet pretrained weight loading result:')
                print('   Missing keys:', len(load_msg.missing_keys))
                print('   Unexpected keys:', len(load_msg.unexpected_keys))

                if len(load_msg.missing_keys) > 0:
                    print('   First 20 missing keys:')
                    for k in load_msg.missing_keys[:20]:
                        print('      ', k)

                if len(load_msg.unexpected_keys) > 0:
                    print('   First 20 unexpected keys:')
                    for k in load_msg.unexpected_keys[:20]:
                        print('      ', k)

                # =====================================================
                # 2.1 判断 GhostNet 预训练是否加载成功
                # =====================================================
                # 理想情况：
                # Missing keys 很少，通常主要是 classifier.weight/bias
                # Unexpected keys 很少或为 0
                #
                # 如果 missing/unexpected 很多，说明：
                # state_dict_73.98.pth 对应的 ghostnet.py
                # 和你当前 archs/ghostnet.py 不一致
                # =====================================================
                if len(load_msg.unexpected_keys) > 20 or len(load_msg.missing_keys) > 50:
                    print('!' * 80)
                    print('WARNING: GhostNet pretrained weights may NOT be loaded correctly.')
                    print('Possible reason:')
                    print('  1. Your archs/ghostnet.py is different from the checkpoint source implementation.')
                    print('  2. The checkpoint is not GhostNet-1.0x ImageNet pretrained weight.')
                    print('  3. Key names or module structures are inconsistent.')
                    print('Suggestion:')
                    print('  Use the ghostnet.py from the same repository as state_dict_73.98.pth.')
                    print('!' * 80)
                else:
                    print('=> GhostNet pretrained weights seem to be loaded correctly.')

            else:
                print('!' * 80)
                print('WARNING: GhostNet ImageNet pretrained weights not loaded.')
                print('self.pretrain = {}'.format(self.pretrain))
                print('pretrained path exists:', os.path.isfile(ghostnet_pretrained_path))
                print('ghostnet_pretrained_path:', ghostnet_pretrained_path)
                print('!' * 80)

            # =========================================================
            # 3. 插入 TemporalShiftMultiScaleDiffFusion
            # =========================================================
            if self.is_shift:
                print('=> Adding TemporalShiftMultiScaleDiffFusion to GhostNet...')
                print('=> Insert position:', ghost_tsf_position)

                # -----------------------------------------------------
                # 位置 1：TSF → ghost1
                # -----------------------------------------------------
                # 结构：
                # Input
                #  ↓
                # TSF
                #  ↓
                # ghost1
                #  ↓
                # DWConv / SE
                #  ↓
                # ghost2
                #  ↓
                # Add
                #
                # 对应代码：
                # m.ghost1 = TemporalShiftMultiScaleDiffFusion(m.ghost1, ...)
                # -----------------------------------------------------
                if ghost_tsf_position == 'before_ghost1':
                    for stage_idx, stage in enumerate(self.base_model.blocks):
                        if stage_idx not in tsm_stages:
                            continue

                        for block_idx, m in enumerate(stage):
                            if isinstance(m, GhostBottleneck) and len(m.shortcut) == 0:
                                if self.print_spec:
                                    print('Adding TSF before ghost1: stage {}, block {}'.format(
                                        stage_idx, block_idx
                                    ))

                                m.ghost1 = TemporalShift(
                                    m.ghost1,
                                    n_segment=self.num_segments,
                                    n_div=self.shift_div
                                )

                # -----------------------------------------------------
                # 位置 2：ghost1 → TSF
                # -----------------------------------------------------
                # 结构：
                # Input
                #  ↓
                # ghost1
                #  ↓
                # TSF
                #  ↓
                # DWConv / SE
                #  ↓
                # ghost2
                #  ↓
                # Add
                #
                # 注意：
                # 这里假设 TemporalShiftMultiScaleDiffFusion 是 wrapper 型模块：
                # TemporalShiftMultiScaleDiffFusion(net, n_segment, n_div)
                #
                # 所以使用 nn.Identity() 作为被包装模块。
                # -----------------------------------------------------
                elif ghost_tsf_position == 'after_ghost1':
                    for stage_idx, stage in enumerate(self.base_model.blocks):
                        if stage_idx not in tsm_stages:
                            continue

                        for block_idx, m in enumerate(stage):
                            if isinstance(m, GhostBottleneck) and len(m.shortcut) == 0:
                                if self.print_spec:
                                    print('Adding TSF after ghost1: stage {}, block {}'.format(
                                        stage_idx, block_idx
                                    ))

                                m.ghost1 = nn.Sequential(
                                    m.ghost1,
                                    TemporalShiftMultiScaleDiffFusion(
                                        nn.Identity(),
                                        n_segment=self.num_segments,
                                        n_div=self.shift_div
                                    )
                                )

                # -----------------------------------------------------
                # 位置 3：TSF → 整个 GhostBottleneck
                # -----------------------------------------------------
                # 结构：
                # Input
                #  ↓
                # TSF
                #  ↓
                # GhostBottleneck
                #
                # 注意：
                # 这种方式会使 main branch 和 shortcut branch
                # 都接收到经过 TSF 的输入。
                # -----------------------------------------------------
                elif ghost_tsf_position == 'whole_block':
                    for stage_idx, stage in enumerate(self.base_model.blocks):
                        if stage_idx not in tsm_stages:
                            continue

                        for block_idx in range(len(stage)):
                            m = stage[block_idx]

                            if isinstance(m, GhostBottleneck) and len(m.shortcut) == 0:
                                if self.print_spec:
                                    print('Adding TSF before whole GhostBottleneck: stage {}, block {}'.format(
                                        stage_idx, block_idx
                                    ))

                                stage[block_idx] = TemporalShiftMultiScaleDiffFusion(
                                    stage[block_idx],
                                    n_segment=self.num_segments,
                                    n_div=self.shift_div
                                )

                else:
                    raise ValueError('Unknown ghost_tsf_position: {}'.format(ghost_tsf_position))

            # =========================================================
            # 4. Flow / RGBDiff 兼容
            # =========================================================
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length


        elif base_model == 'ghostnetv2':
            from archs.ghostnetv2 import ghostnetv2, GhostBottleneckV2

            print('=> using GhostNetV2')
            self.base_model = ghostnetv2(num_classes=1000, width=1.0)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.is_shift:
                from ops.temporal_shift import TemporalShiftSoftmaxFusionAblation

                print('Adding TemporalShiftSoftmaxFusionAblation to GhostNetV2...')

                for m in self.base_model.modules():
                    if isinstance(m, GhostBottleneckV2) and len(m.shortcut) == 0:
                        if self.print_spec:
                            print('Adding temporal shift to GhostNetV2 GhostBottleneckV2...')

                        m.ghost1 = TemporalShiftSoftmaxFusionAblation(
                            m.ghost1,
                            n_segment=self.num_segments,
                            n_div=self.shift_div
                        )

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length


        elif base_model == 'mobilenetv2x2':
            # print("!!!")
            from MODEL.mobilenetv2x2 import mobilenet_v2, InvertedResidual
            # from MODEL.MobileNetV2 import MobileNetV2, InvertedResidual
            # print(self.pretrain)
            # self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)
            self.base_model = mobilenet_v2(pretrained=False)
            # print('self.base_model:', self.base_model)
            # self.base_model = MobileNetV2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            print('self.is_shift：', self.is_shift)
            if self.is_shift:
                from ops.g_temporal_shift import SpatioTemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = SpatioTemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)

        elif base_model == 'eca_mobilenet_v2':
            # print("!!!")
            from MODEL.eca_mobilenetv2 import eca_mobilenet_v2, InvertedResidual
            # from MODEL.MobileNetV2 import MobileNetV2, InvertedResidual
            # print(self.pretrain)
            # self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)
            self.base_model = eca_mobilenet_v2(pretrained=False)
            # self.base_model = MobileNetV2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            print('self.is_shift：', self.is_shift)
            if self.is_shift:
                from ops.g_temporal_shift import SpatioTemporalShift
                for m in self.base_model.modules():
                    # if isinstance(m, InvertedResidual):
                    #     print('len(m.conv):', len(m.conv))

                    if isinstance(m, InvertedResidual) and len(m.conv) == 9 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = SpatioTemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)

        elif base_model == 'mobilenetv2x3':
            from MODEL.mobilenetv2x3 import mobilenet_v2, InvertedResidual
            # self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)
            self.base_model = mobilenet_v2(False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.g_temporal_shift import SpatioTemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        # print('self.print_spec:', self.print_spec)
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = SpatioTemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2x4':
            from MODEL.mobilenetv2x4 import mobilenet_v2, InvertedResidual
            # self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)
            self.base_model = mobilenet_v2(False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        # print('self.print_spec:', self.print_spec)
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length




        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])


            elif isinstance(m, torch.nn.BatchNorm1d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
