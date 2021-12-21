'''
FishNet
Author: Shuyang Sun
'''
from __future__ import division
import tensorflow as tf
import math
from model_tf2.fish_block import *
import tensorflow_addons as tfa
import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

__all__ = ['fish']


class Fish(tf.keras.Model):
    def __init__(self, block, num_cls=1000, num_down_sample=5, num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
                 network_planes=None, num_res_blks=None, num_trans_blks=None):
        super(Fish, self).__init__()
        self.block = block
        self.trans_map = trans_map
        self.upsample = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_first")
        self.down_sample = tf.keras.layers.MaxPool2D(2, strides=2, data_format="channels_first")
        self.num_cls = num_cls
        self.num_down = num_down_sample
        self.num_up = num_up_sample
        self.network_planes = network_planes[1:]
        self.depth = len(self.network_planes)
        self.num_trans_blks = num_trans_blks
        self.num_res_blks = num_res_blks
        self.fish = self._make_fish(network_planes[0])

    def _make_score(self, in_ch, out_ch=1000, has_pool=False):
        bn = tf.keras.layers.BatchNormalization(axis=1)
        relu = tf.keras.layers.ReLU()
        conv_trans = tf.keras.layers.Conv2D(filters= in_ch // 2, kernel_size=1, use_bias=False, data_format="channels_first")
        bn_out = tf.keras.layers.BatchNormalization(axis=1)
        conv = tf.keras.Sequential([bn, relu, conv_trans, bn_out, relu])
        if has_pool:
            fc = tf.keras.Sequential(
                [tfa.layers.AdaptiveAveragePooling2D(1,data_format="channels_first"),
                tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, use_bias=True, data_format="channels_first")])
        else:
            fc = tf.keras.layers.Conv2D(filters= out_ch, kernel_size=1, use_bias=True, data_format="channels_first")
        return [conv, fc]

    def _make_se_block(self, in_ch, out_ch):
        bn = tf.keras.layers.BatchNormalization(axis=1)
        sq_conv = tf.keras.layers.Conv2D(filters = out_ch // 16, kernel_size=1,data_format="channels_first")
        ex_conv = tf.keras.layers.Conv2D(filters = out_ch, kernel_size=1,data_format="channels_first")
        return tf.keras.Sequential([bn,
                             tf.keras.layers.ReLU(),
                             tfa.layers.AdaptiveAveragePooling2D(1,data_format="channels_first"),
                             sq_conv,
                             tf.keras.layers.ReLU(),
                             ex_conv,
                             tf.keras.layers.Activation('sigmoid')])

    def _make_residual_block(self, inplanes, outplanes, nstage, is_up=False, k=1, dilation=1):
        layers = []

        if is_up:
            layers.append(self.block(inplanes, outplanes, mode='UP', dilation=dilation, k=k))
        else:
            layers.append(self.block(inplanes, outplanes, stride=1))
        for i in range(1, nstage):
            layers.append(self.block(outplanes, outplanes, stride=1, dilation=dilation))
        return tf.keras.Sequential(layers)

    def _make_stage(self, is_down_sample, inplanes, outplanes, n_blk, has_trans=True,
                    has_score=False, trans_planes=0, no_sampling=False, num_trans=2, **kwargs):
        sample_block = []
        if has_score:
            sample_block.extend(self._make_score(outplanes, outplanes * 2, has_pool=False))

        if no_sampling or is_down_sample:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, **kwargs)
        else:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, is_up=True, **kwargs)

        sample_block.append(res_block)

        if has_trans:
            trans_in_planes = self.in_planes if trans_planes == 0 else trans_planes
            sample_block.append(self._make_residual_block(trans_in_planes, trans_in_planes, num_trans))

        if not no_sampling and is_down_sample:
            sample_block.append(self.down_sample)
        elif not no_sampling:  # Up-Sample
            sample_block.append(self.upsample)

        return sample_block

    def _make_fish(self, in_planes):
        def get_trans_planes(index):
            map_id = self.trans_map[index-self.num_down-1] - 1
            p = in_planes if map_id == -1 else cated_planes[map_id]
            return p

        def get_trans_blk(index):
            return self.num_trans_blks[index-self.num_down-1]

        def get_cur_planes(index):
            return self.network_planes[index]

        def get_blk_num(index):
            return self.num_res_blks[index]

        cated_planes, fish = [in_planes] * self.depth, []
        for i in range(self.depth):
            # even num for down-sample, odd for up-sample
            is_down, has_trans, no_sampling = i not in range(self.num_down, self.num_down+self.num_up+1),\
                                              i > self.num_down, i == self.num_down
            cur_planes, trans_planes, cur_blocks, num_trans =\
                get_cur_planes(i), get_trans_planes(i), get_blk_num(i), get_trans_blk(i)

            stg_args = [is_down, cated_planes[i - 1], cur_planes, cur_blocks]

            if is_down or no_sampling:
                k, dilation = 1, 1
            else:
                k, dilation = cated_planes[i - 1] // cur_planes, 2 ** (i-self.num_down-1)

            sample_block = self._make_stage(*stg_args, has_trans=has_trans, trans_planes=trans_planes,
                                            has_score=(i==self.num_down), num_trans=num_trans, k=k, dilation=dilation,
                                            no_sampling=no_sampling)
            if i == self.depth - 1:
                conv, fc = self._make_score(cur_planes + trans_planes, out_ch=self.num_cls, has_pool=True)
                sample_block.append(conv)
                sample_block.append(fc)
            elif i == self.num_down:
                sample_block.append(tf.keras.Sequential([self._make_se_block(cur_planes*2, cur_planes)]))

            if i == self.num_down-1:
                cated_planes[i] = cur_planes * 2
            elif has_trans:
                cated_planes[i] = cur_planes + trans_planes
            else:
                cated_planes[i] = cur_planes
            fish.append(sample_block)
            # fish是list里面嵌套list，再里面才是sequential之类的
        return fish

    def _fish_forward(self, all_feat):
        def _concat(a, b):
            return tf.concat([a, b], axis=1)

        def stage_factory(blks):
            #这里可能有点问题 不确定input前面要不要去掉*
            def stage_forward(*inputs):
                if stg_id < self.num_down:  # tail
                    tail_blk = tf.keras.Sequential(blks[:2])
                    return tail_blk(*inputs)
                elif stg_id == self.num_down:
                    score_blks = tf.keras.Sequential(blks[:2])
                    score_feat = score_blks(inputs[0])
                    att_feat = blks[3](score_feat)
                    return blks[2](score_feat) * att_feat + att_feat
                else:  # refine
                    feat_trunk = blks[2](blks[0](inputs[0]))
                    feat_branch = blks[1](inputs[1])
                return _concat(feat_trunk, feat_branch)
            return stage_forward

        stg_id = 0
        # tail:
        while stg_id < self.depth:
            stg_blk = stage_factory(self.fish[stg_id])
            if stg_id <= self.num_down:
                in_feat = [all_feat[stg_id]]
            else:
                trans_id = self.trans_map[stg_id-self.num_down-1]
                in_feat = [all_feat[stg_id], all_feat[trans_id]]

            all_feat[stg_id + 1] = stg_blk(*in_feat)
            stg_id += 1
            # loop exit
            if stg_id == self.depth:
                score_feat = self.fish[self.depth-1][-2](all_feat[-1])
                score = self.fish[self.depth-1][-1](score_feat)
                return score

    def call(self, x):
        all_feat = [None] * (self.depth + 1)
        all_feat[0] = x
        return self._fish_forward(all_feat)


class FishNet(tf.keras.Model):
    def __init__(self, block, **kwargs):
        super(FishNet, self).__init__()

        inplanes = kwargs['network_planes'][0]
        # resolution: 64x64
        self.conv1 = self._conv_bn_relu(3, inplanes // 2, stride=2)
        self.conv2 = self._conv_bn_relu(inplanes // 2, inplanes // 2)
        self.conv3 = self._conv_bn_relu(inplanes // 2, inplanes)
        self.pool1 = tf.keras.layers.MaxPool2D(3, strides=2, data_format='channels_first')
        # construct fish, resolution 56x56
        self.fish = Fish(block, **kwargs)
        self.softmax = tf.keras.layers.Softmax()
        # self._init_weights()

    def _conv_bn_relu(self, in_ch, out_ch, stride=1):
        return tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[1,1],[1,1]])),
            tf.keras.layers.Conv2D(filters= out_ch, kernel_size=3, strides=stride, use_bias=False, data_format='channels_first'),
                                   tf.keras.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON),
                                   tf.keras.layers.ReLU()])

    # def _init_weights(self):
    #     for m in self.layers():
    #         if isinstance(m, tf.keras.layers.Conv2D):
    #             initializer = tf.keras.initializers.GlorotNormal()
    #             if len(m.get_weights()) == 2:
    #                 m.set_weights = [initializer(m.get_weights()[0].shape), m.get_weights()[1]]
    #             elif len(m.get_weights()) == 1:
    #                 m.set_weights = initializer(m.get_weights()[0].shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]])
        x = self.pool1(x)
        score = self.fish(x)
        # 1*1 output
        out = tf.reshape(score, (x.shape[0], -1))
        out = self.softmax(out)

        return out


def fish(**kwargs):
    return FishNet(Bottleneck, **kwargs)
'''
**kwargs参数是Fish的可选参数
num_cls=1000, num_down_sample=5, 
num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
network_planes=None, num_res_blks=None, num_trans_blks=None
'''
