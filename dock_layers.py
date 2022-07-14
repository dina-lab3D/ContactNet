import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, Add, Activation, \
    MaxPool2D, BatchNormalization, Permute, Multiply, Reshape, Dropout, Attention, GlobalAvgPool2D, GlobalMaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
import random as rd
import datetime
import os.path as osp
import os
import matplotlib.pyplot as plt
import seaborn as sn



class CloseDist_loss(tf.keras.losses.Loss):
    def __init__(self, alpha=-0.1):
        super(CloseDist_loss, self).__init__()
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return tf.keras.losses.mean_absolute_error(tf.math.exp(y_true * self.alpha), tf.math.exp(y_pred * self.alpha))


class weighted_BCE(tf.keras.losses.Loss):
    def __init__(self, weight):
        super(weighted_BCE, self).__init__()
        self.pos_weight = weight

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=self.pos_weight))


class ResBlock1D(layers.Layer):
    def __init__(self, channels, kernal=5):
        super(ResBlock1D, self).__init__(name='ResBlock1d')
        self.conv1 = Conv1D(channels, kernal, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(channels, kernal, dilation_rate=(2), padding='same')
        self.bn2 = BatchNormalization()

    def call(self, x, training=False):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = Activation('relu')(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = layers.add([x1, x])
        x1 = Activation('relu')(x1)
        return x1


class ResBlock2D(layers.Layer):
    def __init__(self, channels, kernal=3, name='ResBlock2d'):
        super(ResBlock2D, self).__init__(name=name)
        # self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, kernal, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(channels, kernal, padding='same')
        self.bn2 = BatchNormalization()
        # if self.flag:
        self.bn3 = BatchNormalization()
        self.conv3 = Conv2D(channels, 1, padding="same")
        self.pool = MaxPool2D()

    def call(self, x, training=False):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = Activation('relu')(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        # if self.flag:
        x = self.conv3(x)
        x = self.bn3(x)
        x1 = layers.add([x, x1])
        x1 = Activation('relu')(x1)
        return self.pool(x1)


class ResBlock2Dv2(layers.Layer):
    def __init__(self, channels, kernal=3, name='ResBlock2d', pool=True, global_p=False,trainable=True):
        super(ResBlock2Dv2, self).__init__(name=name)
        self.conv1 = Conv2D(channels, kernal, padding='same',trainable=trainable)
        self.bn1 = BatchNormalization(trainable=trainable)
        self.conv2 = Conv2D(channels, kernal, padding='same',trainable=trainable)
        self.bn2 = BatchNormalization(trainable=trainable)
        self.bn3 = BatchNormalization(trainable=trainable)
        self.conv3 = Conv2D(channels, 1, padding="same",trainable=trainable)
        # self.conv4 = Conv2D(channels,3)
        self.to_pool = pool
        if global_p:
            self.pool = GlobalAvgPool2D()
        else:
            self.pool = AveragePooling2D()

    def call(self, x, training=False):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = Activation('relu')(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x = self.conv3(x)
        x = self.bn3(x)
        x1 = Activation('relu')(x1)
        x1 = layers.add([x, x1])
        if self.to_pool:
            return self.pool(x1)
        return x1


class ResBlockChain(layers.Layer):
    def __init__(self, kernal=5, channels_size=5, name='ResBlock2d_CHAIN'):
        super(ResBlockChain, self).__init__(name=name)
        self.RB1 = ResBlock2D(channels_size, kernal=7)
        self.RB2 = ResBlock2D(channels_size, kernal=kernal)
        self.RB3 = ResBlock2D(channels_size, kernal=kernal)
        self.pool = MaxPool2D()
        # self.RB4 = ResBlock2D(channels_size)

    def call(self, x, training=False):
        x = self.RB1(x)
        x = self.pool(x)
        x = self.RB2(x)
        # x = self.pool(x)
        x = self.RB3(x)
        # x = self.RB4(x)
        return x


# class dist_transformer(layers.Layer):
#     def __init__(self,size,number_of_heads=100):
#         super(dist_transformer,self).__init__()
#         self.self_attention=Attention()
#         self.query_layer=Conv2D(number_of_heads,13,padding='same',activation='relu')
#         self.keys_layer=Conv2D(number_of_heads,13,padding='same',activation='relu')
#         self.values_layer=Conv2D(number_of_heads,13,padding='same',activation='relu')
#         self.fc=layers.Dense(number_of_heads,activation='relu')
#
#
#     def call(self,input ,traning=False):
#         query=self.query_layer(input)
#         query_embeding=layers.GlobalAveragePooling2D()(query)
#         keys=self.keys_layer(input)
#         keys_embeding=layers.GlobalAveragePooling2D()(keys)
#         values=self.values_layer(input)
#         values_embedding=layers.GlobalAveragePooling2D()(values)
#         x=self.self_attention([query_embeding,values_embedding,keys_embeding])
#         return self.fc(x)
class smallMLP3(layers.Layer):
    def __init__(self, size=[100, 100, 100], dropout_rate=0.0, name="", activation="relu",training=True):
        if name:
            super(smallMLP3, self).__init__(name=name)
        else:
            super(smallMLP3, self).__init__()
        self.f1 = Dense(size[0], activation=activation,trainable=training)
        self.bn=BatchNormalization(trainable=training)
        self.f2=Dense(size[1],trainable=training)
        self.out = Dense(size[2],trainable=training)

    def call(self, input ):
        x = self.f1(input)
        x=self.bn(x)
        x=self.f2(x)
        return self.out(x)



class smallMLP(layers.Layer):
    def __init__(self, size=[100, 100, 100], dropout_rate=0.0, activation='relu',training=True):
        super(smallMLP, self).__init__()
        self.f1 = Dense(size[0], activation=activation,trainable=training)
        self.out = Dense(size[2],trainable=training)

    def call(self, input):
        x = self.f1(input)
        return self.out(x)


class smallMLP_bn(layers.Layer):
    def __init__(self, size=[100, 100, 100], dropout_rate=0.0, activation='relu', bias=False,training=True):
        super(smallMLP_bn, self).__init__()
        self.act = tf.keras.activations.get(activation)
        # self.bn_1=layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization(trainable=training)
        self.f1 = Dense(size[0], use_bias=bias,trainable=training)
        # self.f2=Dense(size[1])
        self.out = Dense(size[2], use_bias=bias,trainable=training)

    def call(self, input):
        x = self.f1(input)
        # x=self.bn_1(x)
        x = self.act(x)
        # x=self.f2(x)
        # x=self.act(x)
        return self.act(self.bn_2(self.out(x)))


class Block_conv_block(layers.Layer):
    def __init__(self, filters=[50, 30, 15], activation='relu'):
        super(Block_conv_block, self).__init__()
        self.c1 = Conv2D(filters[0], 5, strides=(2, 2), activation=activation)
        self.c2 = Conv2D(filters[1], 5, strides=(1, 1), activation=activation)
        self.c3 = Conv2D(filters[2], 3, strides=(1, 1), activation=activation)
        self.bn = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.out = Conv2D(filters[2], 3, strides=(1, 1), activation=activation)

    def call(self, input, training=False):
        x = self.c1(input)
        x = self.bn(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.bn1(x)
        return self.out(x)


class chem_transformer1D(layers.Layer):
    def __init__(self, size=80, is_dist=True, drop_rate=0.0,training=True):
        super(chem_transformer1D, self).__init__()
        self.keys = smallMLP_bn(size=[size for i in range(3)], dropout_rate=drop_rate, bias=False,training=training)
        self.queries = smallMLP_bn(size=[size for i in range(3)], dropout_rate=drop_rate, bias=False,training=training)
        self.values = smallMLP_bn(size=[size for i in range(3)], dropout_rate=drop_rate, bias=False,training=training)
        self.size = size
        self.is_dist = is_dist

    def call(self, input, traning=False):
        seq1M, distogram = input
        # seq1M=self.fc1(seq1M)
        if seq1M.shape[1] != distogram.shape[1] or seq1M.shape[1] != distogram.shape[2]:
            print("WTF")
        keys = self.keys(seq1M)
        queries = self.queries(seq1M)
        values = self.values(seq1M)
        if self.is_dist:
            weights = tf.nn.softmax((tf.linalg.matmul(queries, keys, transpose_b=True) * distogram) / self.size, axis=1)
        else:
            weights = tf.nn.softmax(tf.linalg.matmul(queries, keys, transpose_b=True) / self.size, axis=1)
        attention_pruduct = tf.transpose(tf.linalg.matmul(values, weights, transpose_a=True), perm=[0, 2, 1])
        # output=self.bn(self.fc(attention_pruduct+seq1M))
        # return output
        return attention_pruduct


class multiheaded_chem_transformer1D(layers.Layer):
    def __init__(self, number_of_heads=3, size=80,training=True):
        super(multiheaded_chem_transformer1D, self).__init__()
        self.transformers = [chem_transformer1D(size=size,training=training) for i in range(number_of_heads)]
        self.number_of_heads = number_of_heads
        self.projection = Dense(size, use_bias=False,trainable=training)
        # self.fc=smallMLP(size=[size for i in range(3)])
        self.fc = layers.Dense(size, use_bias=False,trainable=training)

        # self.bn=layers.BatchNormalization()
        # self.bn1=layers.BatchNormalization()
        self.ln = layers.LayerNormalization(trainable=training)
        self.ln1 = layers.LayerNormalization(trainable=training)

    def call(self, input, training=False):
        # outputs = [self.transformers[i](input) for i in range(self.number_of_heads)]
        # concat = tf.concat(outputs, axis=2)
        # return self.bn(self.fc(concat))
        outputs = [self.transformers[i](input) for i in range(self.number_of_heads)]
        multi_head_output = self.ln(self.projection(tf.concat(outputs, axis=2)) + input[0])
        output = self.fc(multi_head_output)
        return self.ln1(output + multi_head_output)


class chem_transformer2D(layers.Layer):
    def __init__(self, size=100, is_dist=True):
        super(chem_transformer2D, self).__init__()
        self.keys = smallMLP(size=[size for i in range(3)])
        self.queries = smallMLP(size=[size for i in range(3)])
        self.values = smallMLP(size=[size for i in range(3)])
        self.fc = smallMLP(size=[size for i in range(3)])
        self.size = size
        self.is_dist = is_dist
        self.bn = layers.BatchNormalization()
        self.fc1 = Dense(size, activation='relu')
        self.fc2 = Dense(size, activation='relu')

    def call(self, input, traning=False):
        seq1M, seq2M, distogram = input
        keys1 = self.keys(seq1M)
        queries1 = self.queries(seq1M)
        values1 = self.values(seq1M)
        keys2 = self.keys(seq2M)
        queries2 = self.queries(seq2M)
        values2 = self.values(seq2M)

        if self.is_dist:
            weights1 = tf.nn.softmax((tf.linalg.matmul(queries1, keys2, transpose_b=True) * distogram) / self.size,
                                     axis=1)
            weights2 = tf.nn.softmax((tf.linalg.matmul(queries2, keys1, transpose_b=True) * tf.transpose(distogram,
                                                                                                         perm=[0, 2,
                                                                                                               1])) / self.size,
                                     axis=1)

        else:
            weights1 = tf.nn.softmax((tf.linalg.matmul(queries1, keys2, transpose_b=True)) / self.size,
                                     axis=1)
            weights2 = tf.nn.softmax((tf.linalg.matmul(queries2, keys1, transpose_b=True)) / self.size,
                                     axis=1)
        attention_product1 = tf.transpose(tf.linalg.matmul(values1, weights1, transpose_a=True), perm=[0, 2, 1])
        attention_product2 = tf.transpose(tf.linalg.matmul(values2, weights2, transpose_a=True), perm=[0, 2, 1])
        return self.bn(self.fc1(attention_product1)), self.bn(self.fc2(attention_product2))
        # return tf.transpose(tf.linalg.matmul(values1, weights1,transpose_a=True),perm=[0,2,1]), tf.transpose(tf.linalg.matmul(values2, weights2,transpose_a=True),perm=[0,2,1])


def to_patches_block(geo_patch, patch, seq1M, seq2M, patch_size=30):
    print(patch)
    seq1_row = seq1M[patch[0]:patch[0] + 30]
    seq2_row = seq2M[patch[1]:patch[1] + 30]
    patch_block = mergaeData_cmat(geo_patch, seq1_row, seq2_row, patch_size, patch_size)
    return patch_block


class patch_extract(layers.Layer):
    def __init__(self, patch_size=30):
        super(patch_extract, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs, training=False):
        geo_patch = inputs[0]
        patch = inputs[1]
        seq1M = inputs[2]
        seq2M = inputs[3]
        print("here ", patch)
        return tf.map_fn(lambda x: to_patches_block(x[0], x[1], x[2], x[3], self.patch_size),
                         (geo_patch, patch, seq1M, seq2M))


class encoder(layers.Layer):
    def __init__(self, size=100, drop_rate=0):
        super(encoder, self).__init__()
        # self.fc1=layers.Dense(300,activation='relu')
        self.fc1 = layers.Dense(200, activation='relu')  # was 400
        self.drop = Dropout(drop_rate)  # desable
        # self.fc2=layers.Dense(150,activation='relu')
        self.out = layers.Dense(size, activation='relu')
        self.bn = layers.BatchNormalization()

    def call(self, input, traning=False):
        x = self.fc1(input)
        # x=self.drop(x)
        # x=self.fc2(x)
        x = self.bn(x)
        return self.out(x)


class small_transformer(layers.Layer):
    def __init__(self, size=80,training=True):
        super(small_transformer, self).__init__()
        self.fc = layers.Dense(size,trainable=training)
        self.self_attention = Attention(trainable=training)
        self.query_layer = smallMLP(size=[size for i in range(3)],training=training)
        self.keys_layer = smallMLP(size=[size for i in range(3)],training=training)
        self.values_layer = smallMLP(size=[size for i in range(3)],training=training)
        self.fc1 = layers.Dense(size, activation='relu',trainable=training)
        self.bn = layers.BatchNormalization(trainable=training)

    def call(self, input):
        input = self.fc(input)
        query = self.query_layer(input)
        keys = self.keys_layer(input)
        values = self.values_layer(input)
        x = self.self_attention([query, values, keys])
        return x
        # return self.bn(self.fc1(x)+input)


class multiHeader(layers.Layer):
    def __init__(self, num_heads=3, size=100,training=True):
        super(multiHeader, self).__init__()
        self.heads = [small_transformer(size=size,training=training) for i in range(num_heads)]
        self.projecten = layers.Dense(size, activation='relu',trainable=training)
        # self.fc=smallMLP(size=[size for i in range(3)])
        self.fc = Dense(size, activation='relu',trainable=training)
        # self.bn=layers.BatchNormalization()
        # self.bn1=layers.BatchNormalization()
        self.ln = layers.LayerNormalization(trainable=training)
        self.ln1 = layers.LayerNormalization(trainable=training)

    def call(self, input, training=False):
        outputs = [self.heads[i](input) for i in range(len(self.heads))]
        # return self.fc(tf.concat(outputs,axis=2))
        multi_headed_projecten = self.ln(self.projecten(tf.concat(outputs, axis=2)) + input)
        return self.ln1(self.fc(multi_headed_projecten) + multi_headed_projecten)


class seqLayer(layers.Layer):
    def __init__(self, size, channels=6):
        super(seqLayer, self).__init__()
        self.conv1 = Conv1D(channels, 7, activation='elu')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.mp = layers.MaxPool1D()
        self.resBlock1 = ResBlock1D(channels, kernal=7)
        self.resBlock2 = ResBlock1D(channels, kernal=7)
        self.resBlock3 = ResBlock1D(channels, kernal=5)
        self.resBlock4 = ResBlock1D(channels, kernal=5)
        self.resBlock5 = ResBlock1D(channels, kernal=5)
        self.pool1 = layers.MaxPool1D()
        self.pool2 = layers.MaxPool1D()
        self.pool3 = layers.GlobalAveragePooling1D()
        # self.dense1 = Dense(3000,activation='relu')
        self.out = Dense(size, activation='elu')

    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.bn1(x)
        x = Activation("relu")(x)
        x = self.mp(x)
        x = self.resBlock1(x, training=training)
        x = self.resBlock2(x, training=training)
        x = self.pool1(x)
        x = self.bn2(x, training=training)
        x = self.resBlock3(x, training=training)
        x = self.bn3(x)
        x = self.resBlock4(x, training=training)
        # x = self.pool2(x)
        x = self.pool3(x)
        # x=self.resBlock5(x,training=training)
        # print("1dRes:" ,tf.shape(x))
        x = layers.Flatten()(x)
        # x=self.dense1(x)
        return self.out(x)


class seq2vec_Layer(layers.Layer):
    def __init__(self, channels=20):
        super(seq2vec_Layer, self).__init__()
        self.conv1 = Conv1D(channels, 13, padding="same", activation='elu')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.resBlock1 = ResBlock1D(channels, kernal=7)
        self.resBlock2 = ResBlock1D(channels, kernal=5)
        self.resBlock3 = ResBlock1D(channels, kernal=5)
        self.resBlock4 = ResBlock1D(channels, kernal=5)
        self.out = Dense(1)

    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.bn1(x)
        x = Activation("relu")(x)
        x = self.resBlock1(x, training=training)
        x = self.resBlock2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.resBlock3(x, training=training)
        x = self.bn1(x)
        x = self.resBlock4(x)
        return x
        # return layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        # return layers.Flatten()(self.out(x))


def concatTensor(small, big):
    small = tf.expand_dims(small, -1)
    return tf.keras.layers.concatenate([big, small])


def pad_chem(seq1, seq2, size_r, size_l):
    seq1Vec = seq2vec_Layer()(seq1)  ##size r
    seq2Vec = seq2vec_Layer()(seq2)  ##size l
    # seq1VecT = Reshape((1, -1))(seq1Vec)  ##row vec
    seq1VecT = tf.expand_dims((seq1Vec), axis=2)  ##shape (bs,size r,1,filters
    seq2VecT = tf.expand_dims(seq2Vec, axis=1)  ##shape (bs,1,size l,filters

    seq1Mat = tf.tile(seq1VecT, (1, 1, size_l, 1))  ###shape (bs,size r,size l,filters
    seq2Mat = tf.tile(seq2VecT, (1, size_r, 1, 1))  ### shape (bs,size r,size l,filters
    return seq1Mat, seq2Mat


def mergaeData_cmat(geoMat, seq1, seq2, size_r, size_l):
    # seq1Vec = seq2vec_Layer()(seq1)##size r
    # seq2Vec = seq2vec_Layer()(seq2)##size l
    # # seq1VecT = Reshape((1, -1))(seq1Vec)  ##row vec
    # seq1VecT = tf.expand_dims((seq1Vec), axis=2)##shape (bs,size r,1,filters
    # seq2VecT=tf.expand_dims(seq2Vec,axis=1)##shape (bs,1,size l,filters
    #
    # seq1Mat=tf.tile(seq1VecT,(1,1,size_l,1))###shape (bs,size r,size l,filters
    # seq2Mat=tf.tile(seq2VecT,(1,size_r,1,1))### shape (bs,size r,size l,filters
    seq1Mat, seq2Mat = pad_chem(seq1, seq2, size_r, size_l)
    return tf.concat([geoMat, seq1Mat, seq2Mat], axis=3)


def mergaeData(geoMat, seq1, seq2, size_r, size_l):
    seq1Vec = seqLayer(size_r, channels=14)(seq1)
    seq1VecT = Reshape((1, -1))(seq1Vec)  ##row vec
    seq2Vec = seqLayer(size_l, channels=14)(seq2)
    seq2VecT = Reshape((-1, 1))(seq2Vec)  ##colvec
    outer = Multiply()([seq1VecT, seq2VecT])  ## matrix
    print("outer product: ", tf.shape(outer))
    seqMat = Permute((2, 1))(outer)  ## match to geo matrix
    print("seq mat: ", tf.shape(seqMat), " geo mat: ", tf.shape(geoMat))
    return concatTensor(seqMat, geoMat)


@tf.function
def cut_patch(seqM, patch, horizontial=True):
    ind = patch[0] if horizontial else patch[1]
    return seqM[ind:ind + 30, :]


@tf.function
def cut_patches(seqM, patches, num_of_patches=6, horizontial=True):
    rows = []
    for i in range(num_of_patches):
        rows.append(cut_patch(seqM, patches[i], horizontial=horizontial))
    return tf.stack(rows)


@tf.function
def make_block(row, col, patch_size, geo):
    row = tf.expand_dims(row, axis=0)
    col = tf.expand_dims(col, axis=1)
    geo = tf.expand_dims(geo, axis=2)
    row_tiled = tf.tile(row, (patch_size, 1, 1))
    col_tiled = tf.tile(col, (1, patch_size, 1))
    block = tf.concat([row_tiled, col_tiled, geo], axis=2)
    return block


def to_blocks(rows, cols, geo, num_of_blocks, seq_lataent_space=60, patch_size=30):
    blocks = []
    for i in range(num_of_blocks):
        block = make_block(rows[i], cols[i], patch_size, geo[i])
        tf.ensure_shape(block, (patch_size, patch_size, seq_lataent_space * 2 + 1))
        blocks.append(block)
    return tf.stack(blocks)


class make_blocks(layers.Layer):
    def __init__(self, num_of_batches=20, num_of_patches=8, seq_latent_space=60):
        super(make_blocks, self).__init__()
        self.num_of_batches = num_of_batches
        self.num_of_patches = num_of_patches
        self.seq_latent_space = seq_latent_space

    def call(self, input, traning=False):
        seq1M, seq2M, patches, geo = input
        # tf.ensure_shape(seq1M,(None,size_r,self.seq_latent_space))
        # tf.ensure_shape(seq2M,(None,size_l,self.seq_latent_space))
        tf.ensure_shape(geo, (None, self.num_of_patches, 30, 30))
        tf.ensure_shape(patches, (None, self.num_of_patches, 2))
        batches_of_blocks = []
        for i in range(self.num_of_batches):
            rows = cut_patches(seq1M[i], patches[i], num_of_patches=self.num_of_patches)
            cols = cut_patches(seq2M[i], patches[i], num_of_patches=self.num_of_patches, horizontial=False)
            tf.ensure_shape(rows, (self.num_of_patches, 30, self.seq_latent_space))
            tf.ensure_shape(rows, (self.num_of_patches, 30, self.seq_latent_space))
            batches_of_blocks.append(to_blocks(rows, cols, geo[i], num_of_blocks=self.num_of_patches,
                                               seq_lataent_space=self.seq_latent_space))
        return tf.stack(batches_of_blocks)


def apply_dists_on_block(block):
    chem_block = block[:, :, :-1]
    distogram = tf.expand_dims(block[:, :, -1], axis=2)
    distogram_tiled = tf.tile(distogram, (1, 1, chem_block.shape[2]))
    normelized = distogram_tiled * chem_block
    return normelized


class dist_normalized(layers.Layer):
    def __init__(self, num_of_batches=20, num_of_patches=6):
        super(dist_normalized, self).__init__()

    def call(self, input, traning=False):
        return tf.map_fn(lambda b: tf.map_fn(lambda p: apply_dists_on_block(p), b), input)


@tf.function
def binary_crossentropy_relaxed(y_true, y_pred):
    labels = y_true[:, 0]
    rec_rmsd = y_true[:, 1]
    lig_rmsd = y_true[:, 2]
    binary_crossentropy = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.expand_dims(labels, 1), y_pred))
    distance_arg = tf.reduce_mean(tf.math.log(rec_rmsd) / 10) + tf.reduce_mean(tf.math.log(lig_rmsd) / 5)
    return binary_crossentropy + distance_arg
