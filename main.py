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
checkpoint_filepath = '2PSCDN/weight'
Autoencoder.load_weights(checkpoint_filepath)
mat = scipy.io.loadmat('train.mat')
train_ori = mat['phasebit']
train_ori = (np.reshape(train_ori, (train_ori.shape[0], train_ori.shape[1], 1)))
train_ori = tf.dtypes.cast(train_ori, tf.float32)

mat = scipy.io.loadmat('validation_csi.mat')
test_ori = mat['phasebit']
test_ori = (np.reshape(test_ori, (test_ori.shape[0], test_ori.shape[1], 1)))
test_ori = tf.dtypes.cast(test_ori, tf.float32)

checkpoint_filepath = '2PSCDN/weight'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.99,
    staircase=True)
optimizer = Adam(learning_rate=lr_schedule)
Autoencoder.compile(optimizer, loss=MSE())
Autoencoder.fit(x=train_ori, y=train_ori, batch_size=128, epochs=1000, callbacks=[model_checkpoint_callback], validation_data=(test_ori, test_ori),
                validation_batch_size=1000)
checkpoint_filepath = '2PSCDN/weight'
Autoencoder.load_weights(checkpoint_filepath)
#for testing 
import time
from numpy.linalg import norm
from numpy import arange,reshape,round,power


matv = scipy.io.loadmat('test10w.mat')
test_ori=matv['phasebit'] 
phase=matv['phase'] 
test_ori = (np.reshape(test_ori,(test_ori.shape[0],test_ori.shape[1],1)))
test_ori = tf.dtypes.cast(test_ori, tf.float32)

start1 = time.time()
Encoder=Autoencoder.enc.predict(x=test_ori, batch_size=1000)
end1 = time.time()

nosie = awgn(Encoder,10)

start2 = time.time()
output = Autoencoder.dec.predict(x=nosie, batch_size=1000)
end2 = time.time()

nmse_round=power(norm(test_ori-output),2) / power(norm(test_ori),2)
nmse_round
