import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class Bottleneck(tf.keras.Model):
    def __init__(self, inplanes, planes, stride=1, mode='NORM', k=1, dilation=1):
        """
        Pre-act residual block, the middle transformations are bottle-necked
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        :param mode: NORM | UP
        :param k: times of additive
        """

        super(Bottleneck, self).__init__()
        self.mode = mode
        self.relu = tf.keras.layers.ReLU()
        self.k = k
        self.dilation = dilation

        btnk_ch = planes // 4
        # inplanes考虑怎么办
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)
        self.conv1 = tf.keras.layers.Conv2D(filters=btnk_ch, kernel_size=1, use_bias=False, data_format='channels_first')

        self.bn2 = tf.keras.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)
        self.conv2 = tf.keras.layers.Conv2D(filters=btnk_ch, kernel_size=3, strides=stride, dilation_rate=dilation, use_bias=False, data_format='channels_first')

        self.bn3 = tf.keras.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)
        self.conv3 = tf.keras.layers.Conv2D(filters=planes, kernel_size=1, use_bias=False, data_format='channels_first')

        if mode == 'UP':
            self.shortcut = None
        elif inplanes != planes or stride > 1:
            self.shortcut = tf.keras.Sequential(
                [tf.keras.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON),
                self.relu,
                tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, data_format="channels_first")]
            )
        else:
            self.shortcut = None

    def _pre_act_forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = tf.pad(out,[[0,0],[0,0],[self.dilation,self.dilation],[self.dilation,self.dilation]])
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.mode == 'UP':
            residual = self.squeeze_idt(x)
        elif self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual

        return out

    def squeeze_idt(self, idt):
        n, c, h, w = idt.shape
        return tf.reduce_sum(tf.reshape(idt, (n, c // self.k, self.k, h, w)), axis=2)

    def call(self, x):
        out = self._pre_act_forward(x)
        return out
