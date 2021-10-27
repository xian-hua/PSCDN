import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Dense, LeakyReLU, Add, Reshape, Layer, ReLU, \
    Concatenate
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.optimizers import Adam
import scipy.io

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



def biterr(inputs_bits, ori_bits):
    symbols = inputs_bits.shape[0]
    num_bits = inputs_bits.shape[1]
    total_bits = symbols * num_bits
    a = np.reshape(inputs_bits, (1, total_bits))
    b = np.reshape(ori_bits, (1, total_bits))
    errors = (a != b).sum()
    # errors = errors.astype(int)
    ber = errors / total_bits
    return ber


class convbn(Layer):
    def __init__(self, neurons):
        super(convbn, self).__init__()
        self.con = Conv1D(neurons, 3, strides=1, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.neurons = neurons

    def call(self, input):
        output = self.con(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class Dblock(Layer):
    def __init__(self, neurons):
        super(Dblock, self).__init__()
        self.convbn1 = convbn(neurons)
        self.convbn2 = convbn(neurons)
        self.convbn3 = convbn(neurons)
        self.convbn4 = convbn(neurons)

    def call(self, input):
        output = self.convbn1(input)
        output = self.convbn2(output)
        output = self.convbn3(output)
        output = self.convbn4(output)
        return output


class DRblock(Layer):
    def __init__(self, neurons):
        super(DRblock, self).__init__()
        self.conv1 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv2 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv3 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv4 = Conv1D(neurons, 3, padding='same', activation='relu')

    def call(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output


class Encoder(Model):
    def __init__(self, inbits, cbits, neurons):
        super(Encoder, self).__init__()
        self.cbits = cbits
        self.inbits = inbits
        self.reshape = Reshape((1, neurons * inbits))
        self.conv1 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv2 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv3 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.conv4 = Conv1D(cbits, 1)

    def call(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.reshape(output)
        output = self.conv4(output)
        return output


class Decoder(Model):
    def __init__(self, inbits, neurons):
        super(Decoder, self).__init__()
        self.neurons = neurons
        self.inbits = inbits
        self.inilayer = Conv1D(neurons * inbits, 1, activation='relu')
        self.reshape = Reshape((inbits, neurons))
        self.DRblock = DRblock(neurons)
        self.DRblock = DRblock(neurons)
        self.cu = Conv1D(neurons, 1)
        self.conv = Conv1D(1, 3, padding='same', activation='sigmoid')
        self.concat = Concatenate()

    def call(self, input):
        # reconstruction
        output = self.inilayer(input)
        noise_output = self.reshape(output)
        output = self.DRblock(noise_output)
        output = self.DRblock(output)
        # enhancement
        output = self.concat([noise_output, output])
        output = self.cu(output)
        # denoising   
        output = noise_output - output
        # final reconstruction
        output = self.conv(output)
        return output


class Autoencoder(Model):
    def __init__(self, inbits, cbits, neurons):
        super(Autoencoder, self).__init__()
        self.cbits = cbits
        self.inbits = inbits
        self.neurons = neurons
        self.enc = Encoder(inbits, cbits, neurons)
        self.dec = Decoder(inbits, neurons)

    def call(self, input):
        ouputenc = self.enc(input)
        #         ouputenc = awgn(ouputenc,10)
        ouputdec = self.dec(ouputenc)
        return ouputdec


neurons = 64
inbits = 9
cbits = 2
Autoencoder = Autoencoder(inbits, cbits, neurons)
# Autoencoder.build(input_shape = (100,9,1))
# Autoencoder.call(tf.keras.layers.Input(shape = (9,1)))
# Autoencoder.summary()
checkpoint_filepath = '2PSCDN/weight'
Autoencoder.load_weights(checkpoint_filepath)
