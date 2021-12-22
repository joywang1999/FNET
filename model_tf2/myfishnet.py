#####################################
# Original Author: Shuyang Sun      #
# Reproduced by E4040.2021Fall.FNET #
#####################################


import tensorflow as tensorflow
import tensorflow_addons as tfa
import numpy as np

from model_tf2.fish_block import *

# Global Variables

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

tf.keras.backend.image_data_format("channels_first")

class Fish(tf.keras.Model):
	''' This contains the main structure of 'Fish' (head, body, tail).
		
	'''

	def __init__(self, bottleneck, num_cls=200, num_down_sample=3, num_up_sample=3, trans_map=(2,1,0,6,5,4),
		         network_planes=None, num_res_blks=None, num_trans_blks=None):

		# inherit tf.keras.Model
		super(Fish, self).__init__()

		# private params
		self.block = bottleneck
		self.num_cls = num_cls
		self.num_down_sample = num_down_sample
		self.num_up_sample = num_up_sample
		self.trans_map = trans_map

		self.num_res_blks = num_res_blks
		self.num_trans_blks = num_trans_blks

		# TODO: some comment
		self.fish = self._make_fish(network_planes[0])
		self.network_planes = network_planes[1:]
		self.depth = len(self.network_planes)

		# upsample and downsample layers
		self.upsample = tf.keras.layers.UpSampling2D(size=(2,2))
		self.downsample = tf.keras.layers.MaxPool2D(2, stirdes=2)


	def _stage_block(self, inplanes, outplanes, sampling, num_blocks, 
		             add_score=False, add_trans=False, trans_planes=0, num_trans=2, **kwargs):

		# sampling = one of ['upsampling', 'downsampling', 'nosampling']

		score_layer = self._score_layer(inplanes, outplanes, pooling=False) if add_score else []
		residual_block = [self._residual_block(inplanes, outplanes, num_blocks, upsampling=(sampling == "upsampling"), **kwargs)]

		planes = max(trans_planes, inplanes)
		trans_block = [self._residual_block(planes, planes, num_trans)] if add_trans else []

		down_block = [self.downsample] if sampling == "downsampling" else []
		up_block = [self.upsample] if sampling == "upsampling" else []

		blocks = score_layer + residual_block + trans_block + down_block + up_block

		return blocks

	def _make_fish(self, planes):

		cated_planes = [planes] * self.depth
		fish = []

		for i in range(self.depth):

			add_score = False
			add_trans = False
			# down, up, trans
			if i == self.num_down_sample:
				sampling = "nosampling"
				add_score = True
			elif self.num_down_sample < i < self.num_down_sample+self.num_up_sample:
				sampling = "upsampling"
				
			else: 
				sampling = "downsampling"
				if i > self.num_down_sample: 
					add_trans = True

			# network config
			trans_index = i-self.num_down_sample-1

			map_id = self.trans_map[trans_index] - 1
			trans_planes = planes if map_id == -1 else cated_planes[map_id]

			num_trans = self.num_trans_blks[trans_index]
			num_blocks = self.num_res_blks[i]

			inplanes = cated_planes[i-1]
			outplanes = self.network_planes[i]

			# update cated_planes
			if i == self.num_down_sample - 1:
				cated_planes[i] = 2 * outplanes 
			elif has_trans:
				cated_planes[i] = outplanes + trans_planes
			else:
				cated_planes[i] = outplanes

			# make stage
			stg_args = [inplanes, outplanes, sampling, num_blocks, 
						add_score, add_trans, trans_planes, num_trans] 

			if sampling == "upsampling":
				dilation = 2 ** trans_index
				k = inplanes // outplanes
			else:
				dilation = 1
				k = 1 

			blocks = self._stage_block(*stg_args, dilation=dilation, k=k)

			# last layer
			if i == self.depth - 1:
				blocks += self._score_layer(outplanes + trans_planes, outplanes=self.num_cls, pooling=True)
			elif i == self.num_down_sample:
				blocks += [self._squeeze_block(2*outplanes, outplanes)]

			fish.append(blocks)

		return fish


	def _fish_forward(self, features):

		# python decorator that returns the function itself
		def stage_factory(blocks):
			def stage_forward(*inputs):
				# we can access i from outside
				if i < self.num_down:
					tail_block = tf.keras.Sequential(blocks[:2])
					return tail_block(*inputs)
				elif i == self.num_down:
					score_block = tf.keras.Sequential(blocks[:2])
					score_feature = score_block(input[0])
					next_feature = blocks[3](score_feature)
					return block[2](score_feature) * next_feature + next_feature
				else: # i > self.num_down
					in_feature0 = blocks[0](inputs[0])
					in_feature1 = blocks[1](inputs[1])
					feature_trunk = blocks[2](in_feature0)
					return tf.concat([feature_trunk, in_feature1], axis=1)
			return stage_forward

		for i in range(self.depth):
			stage = stage_factory(self.fish[i])
			if i <= self.num_down:
				in_features = [features[i]]
			else:
				trans_index = i-self.num_down_sample-1
				map_id = self.trans_map[trans_index]
				in_features = [features[i], features[map_id]]

			features[i+1] = stage(*in_features)

		# the last layer is for output score
		# we need to manually compute by getting the second last layer
		score_feature = self.fish[i][-2](features[-1])
		score = self.fish[i][-1](score_feature)
		return score


	def _score_layer(self, inplanes, outplanes=200, pooling=False):

		# part1
		conv_layer = tf.keras.Sequential([
			tf.keras.layers.BatchNormalization(axis=1),
			tf.keras.layers.Activation('relu'),
			tf.keras.layers.Conv2D(filters=inplanes//2, kernel_size=1, use_bias=False),
			tf.keras.layers.BatchNormalization(axis=1),
			tf.keras.layers.Activation('relu')
		])

		# part2 
		pooling = [tfa.layers.AdaptiveAveragePooling2D(1)] if pooling else []
		out_layer = tf.Sequential(pooling + 
			[tf.keras.layers.Conv2D(filters=outplanes, kernel_size=1, use_bias=True)])

		return [conv_layer, out_layer]

	def _squeeze_block(self, planes):
		# map to make se block
		block = tf.keras.Sequential([
			tf.keras.layers.BatchNormalization(axis=1),
			tf.keras.layers.Activation('relu'),
			tfa.layers.AdaptiveAveragePooling2D(1),
			tf.keras.layers.Conv2D(filters=planes//16, kernel_size=1),
			tf.keras.layers.Activation('relu'),
			tf.keras.layers.Conv2D(filters=planes, kernel_size=1),
			tf.keras.layers.Activation('sigmoid')
		])
		return block

	def _residual_block(self, inplanes, outplanes, num_stage, upsampling=False, dilation=1, k=1):
		# map to make residual block
		# blocks are in structure of bottleneck

		if upsampling:
			layers = [self.block(inplanes, outplanes, mode="UP", dilation=dilation, k=k)]
		else: # no up sampling
			layers = [self.block(inplanes, outplanes, mode="NORM", stides=1)]

		for i in range(1, num_stage):
			layers += [self.block(outplanes, outplanes, mode="NORM", stides=1, dilation=dilation)]

		block = tf.keras.Sequential(layers)

		return block

	def call(self, x):
		features = [x] + [None] * self.depth
		return self._fish_forward(features)



class FishNet(tf.keras.Model):
	''' This is the complete structure of FishNet, combining layers of very deep CNN on top
	with 'fish'. 

	'''

	def __init__(self, bottleneck, **kwargs):
		super(FishNet, self).__init__()

		planes = kwargs['network_planes'][0]

		# three deep convulution layers added before fish to lower resolution
		# init resolution: 64*64
		self.conv1 = self._deep_conv_layer(planes//2, strides=2)
		self.conv2 = self._deep_conv_layer(planes//2, strides=1)
		self.conv3 = self._deep_conv_layer(planes, strides=1)
		self.pool1 = tf.keras.layers.MaxPool2D(3, strides=2)
		# after resolution: 16*16

		self.fish = Fish(bottleneck, **kwargs)
		self.softmax = tf.keras.layers.Activation('softmax')


	def _deep_conv_layer(self, planes, strides=1):

		layer = tf.keras.Sequential([
			tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [0,0], [1,1], [1,1]])),
			tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=strides, use_bias=False),
			tf.keras.layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, axis=1),
			tf.keras.layers.Activation('relu')
		])

		return layer

	def call(self, x):

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = tf.pad(out, [[0,0], [0,0], [1,1], [1,1]])
		out = self.pool1(out)
		
		score = self.fish(out)
		out = tf.reshape(score, (out.shape[0], -1))
		out = self.softmax(out)

		return out



def fish(**kwargs):
	return FishNet(Bottleneck, **kwargs)