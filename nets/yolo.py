#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import time

import torch
import torch.nn as nn

from nets.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from torch.profiler import profile, record_function, ProfilerActivity


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs, prof_wrapper):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            if k == 0:
                prof_wrapper.scale.dependency_check(tensor_name="x",
                                                    src="C3_p3",
                                                    dest="stems_" + str(k))
            elif k == 1:
                prof_wrapper.scale.dependency_check(tensor_name="x",
                                                    src="C3_n3",
                                                    dest="stems_" + str(k))
            elif k == 2:
                prof_wrapper.scale.dependency_check(tensor_name="x",
                                                    src="C3_n4",
                                                    dest="stems_" + str(k))

            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            t_0 = time.time()
            x = self.stems[k](x)
            t_1 = time.time()
            prof_wrapper.tt.get_time("stems_" + str(k), t_1 - t_0)
            prof_wrapper.scale.weight(tensor_src="stems_" + str(k),  data=x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="stems_" + str(k),
                                                dest="cls_convs_" + str(k))
            t0 = time.time()
            cls_feat = self.cls_convs[k](x)
            t1 = time.time()
            prof_wrapper.tt.get_time("cls_convs_" + str(k), t1 - t0)
            prof_wrapper.scale.weight(tensor_src="cls_convs_" + str(k), data=cls_feat)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="cls_convs_" + str(k),
                                                dest="cls_preds_" + str(k))

            t2 = time.time()
            cls_output = self.cls_preds[k](cls_feat)
            t3 = time.time()
            prof_wrapper.tt.get_time("cls_preds_" + str(k), t3 - t2)
            prof_wrapper.scale.weight(tensor_src="cls_preds_" + str(k), data=cls_output)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="stems_" + str(k),
                                                dest="reg_convs_" + str(k))

            t4 = time.time()
            reg_feat = self.reg_convs[k](x)
            t5 = time.time()
            prof_wrapper.tt.get_time("reg_convs_" + str(k), t5 - t4)
            prof_wrapper.scale.weight(tensor_src="reg_convs_" + str(k), data=reg_feat)

            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="reg_convs_" + str(k),
                                                dest="reg_preds_" + str(k))

            t6 = time.time()
            reg_output = self.reg_preds[k](reg_feat)
            t7 = time.time()
            prof_wrapper.tt.get_time("reg_preds_" + str(k), t7 - t6)
            prof_wrapper.scale.weight(tensor_src="reg_preds_" + str(k), data=reg_output)

            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="reg_convs_" + str(k),
                                                dest="obj_preds_" + str(k))

            t8 = time.time()
            obj_output = self.obj_preds[k](reg_feat)
            t9 = time.time()
            prof_wrapper.tt.get_time("obj_preds_" + str(k), t9 - t8)
            prof_wrapper.scale.weight(tensor_src="obj_preds_" + str(k), data=obj_output)

            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="reg_preds_" + str(k),
                                                dest="partial_out_" + str(k))
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="obj_preds_" + str(k),
                                                dest="partial_out_" + str(k))
            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="cls_preds_" + str(k),
                                                dest="partial_out_" + str(k))


            t10 = time.time()
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            t11 = time.time()
            prof_wrapper.tt.get_time("partial_out_" + str(k), t11 - t10)
            prof_wrapper.scale.weight(tensor_src="partial_out_" + str(k), data=output)


            prof_wrapper.scale.dependency_check(tensor_name="x",
                                                src="partial_out_" + str(k),
                                                dest="output_decode")
            outputs.append(output)
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input, prof_wrapper):
        out_features = self.backbone.forward(input, prof_wrapper)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]
        # "dark3", "dark4", "dark5"

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="dark5",
                                            dest="lateral_conv0")

        t0 = time.time()
        P5 = self.lateral_conv0(feat3)
        t1 = time.time()
        prof_wrapper.tt.get_time("lateral_conv0", t1 - t0)
        prof_wrapper.scale.weight(tensor_src="lateral_conv0", data=P5)
        # -------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="lateral_conv0",
                                            dest="upsample_P5")

        t2 = time.time()
        P5_upsample = self.upsample(P5)
        t3 = time.time()
        prof_wrapper.tt.get_time("upsample_P5", t3 - t2)
        prof_wrapper.scale.weight(tensor_src="upsample_P5", data=P5_upsample)
        # -------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="upsample_P5",
                                            dest="C3_p4")
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="dark4",
                                            dest="C3_p4")
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#

        t4 = time.time()
        P5_upsample = self.C3_p4(P5_upsample)
        t5 = time.time()
        prof_wrapper.tt.get_time("C3_p4", t5 - t4)
        prof_wrapper.scale.weight(tensor_src="C3_p4", data=P5_upsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="C3_p4",
                                            dest="reduce_conv1")

        t6 = time.time()
        P4 = self.reduce_conv1(P5_upsample)
        t7 = time.time()
        prof_wrapper.tt.get_time("reduce_conv1", t7 - t6)
        prof_wrapper.scale.weight(tensor_src="reduce_conv1", data=P4)
        # -------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="reduce_conv1",
                                            dest="upsample_P4")

        t8 = time.time()
        P4_upsample = self.upsample(P4)
        t9 = time.time()
        prof_wrapper.tt.get_time("upsample_P4", t9 - t8)
        prof_wrapper.scale.weight(tensor_src="upsample_P4", data=P4_upsample)
        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="upsample_P4",
                                            dest="C3_p3")
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="dark3",
                                            dest="C3_p3")
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#

        t10 = time.time()
        P3_out = self.C3_p3(P4_upsample)
        t11 = time.time()
        prof_wrapper.tt.get_time("C3_p3", t11 - t10)
        prof_wrapper.scale.weight(tensor_src="C3_p3", data=P3_out)

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="C3_p3",
                                            dest="bu_conv2")

        t12 = time.time()
        P3_downsample = self.bu_conv2(P3_out)
        t13 = time.time()
        prof_wrapper.tt.get_time("bu_conv2", t13 - t12)
        prof_wrapper.scale.weight(tensor_src="bu_conv2", data=P3_downsample)
        # -------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="bu_conv2",
                                            dest="C3_n3")
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="reduce_conv1",
                                            dest="C3_n3")
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#

        t14 = time.time()
        P4_out = self.C3_n3(P3_downsample)
        t15 = time.time()
        prof_wrapper.tt.get_time("C3_n3", t15 - t14)
        prof_wrapper.scale.weight(tensor_src="C3_n3", data=P4_out)

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="C3_n3",
                                            dest="bu_conv1")

        t16 = time.time()
        P4_downsample = self.bu_conv1(P4_out)
        t17 = time.time()
        prof_wrapper.tt.get_time("bu_conv1", t17 - t16)
        prof_wrapper.scale.weight(tensor_src="bu_conv1", data=P4_downsample)
        # -------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        # -------------------------------------------#
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="bu_conv1",
                                            dest="C3_n4")
        prof_wrapper.scale.dependency_check(tensor_name="x",
                                            src="lateral_conv0",
                                            dest="C3_n4")
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#

        t18 = time.time()
        P5_out = self.C3_n4(P4_downsample)
        t19 = time.time()
        prof_wrapper.tt.get_time("C3_n4", t19 - t18)
        prof_wrapper.scale.weight(tensor_src="C3_n4", data=P5_out)

        return (P3_out, P4_out, P5_out)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x, prof_wrapper):
        fpn_outs = self.backbone.forward(x, prof_wrapper)
        outputs = self.head.forward(fpn_outs, prof_wrapper)
        prof_wrapper.tt.report()
        return outputs
