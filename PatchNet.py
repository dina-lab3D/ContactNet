import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, Add, Activation, \
    MaxPool2D, BatchNormalization, Permute, Multiply, Reshape, Dropout, Attention
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.config.run_functions_eagerly(False)


import sys
import random as rd
import datetime
from decay_class import CustomSchedule,WarmUp,LRLogger
from generators import gen_cycler,build_prot_dict
import os.path as osp
import os
from dock_layers import ResBlock2D,ResBlock2Dv2, chem_transformer1D, small_transformer,Block_conv_block,multiHeader,\
    multiheaded_chem_transformer1D,encoder,binary_crossentropy_relaxed,smallMLP3
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sn
from config import config4 as config
import os



def veifyDir(path):
    if not osp.isdir(path):
        os.mkdir(path)

@tf.function
def cut_patch(seqM,patch,horizontial=True,patch_size=20):
    ind=patch[0] if horizontial else patch[1]
    # if (seqM.shape[0] is not None) :
    #     if  seqM.shape[0]<=ind+patch_size :
    #         print("XXXXXXXXXXXXXXXXX   ",ind,"   ", horizontial)
    #         print(seqM.shape)
    #         print(patch_size)
    #         raise IndexError
    seq_patch=seqM[ind:ind+patch_size,:]
    # tf.debugging.assert_none_equal(tf.reduce_sum(seq_patch), tf.constant(0, dtype=tf.float32))
    return seq_patch

def cut_patches(seqM,patches,num_of_patches=8,horizontial=True,patch_size=20):
    rows=[]
    for i in range(num_of_patches):
        rows.append(cut_patch(seqM,patches[i],horizontial=horizontial,patch_size=patch_size))
    return tf.stack(rows)


def make_block(row,col,patch_size,geo):
    row=tf.expand_dims(row,axis=0)
    col=tf.expand_dims(col,axis=1)
    geo=tf.expand_dims(geo,axis=2)
    row_tiled=tf.tile(row,(patch_size,1,1))
    col_tiled=tf.tile(col,(1,patch_size,1))
    block=tf.concat([row_tiled,col_tiled,geo],axis=2)
    return block


def to_blocks(rows, cols, geo, num_of_blocks,seq_lataent_space=60, patch_size=30):
    blocks=[]
    for i in range(num_of_blocks):
        block=make_block(rows[i],cols[i],patch_size,geo[i])
        tf.ensure_shape(block,(patch_size,patch_size,seq_lataent_space*2+1))
        blocks.append(block)
    return tf.stack(blocks)

def cut_to_blocks(input,num_of_batches=20,num_of_patches=6,patch_size=30):
    seq1M, seq2M, patches, geo = input
    tf.ensure_shape(geo, (None, 8, 20, 20))
    tf.ensure_shape(patches, (None, 8, 2))
    batches_of_blocks = []
    for i in range(num_of_batches):
        rows = cut_patches(seq1M[i], patches[i], num_of_patches=num_of_patches,patch_size=patch_size)
        cols = cut_patches(seq2M[i], patches[i], num_of_patches=num_of_patches, horizontial=False,patch_size=patch_size)
        tf.ensure_shape(rows, (config["arch"]["number_of_patches"], config["arch"]["patch_size"], config["arch"]["seq_latent_dim"]))
        tf.ensure_shape(rows, (config["arch"]["number_of_patches"], config["arch"]["patch_size"], config["arch"]["seq_latent_dim"]))
        batches_of_blocks.append(to_blocks(rows, cols, geo[i], num_of_blocks=num_of_patches))
    return tf.stack(batches_of_blocks)

class make_blocks(layers.Layer):
    def __init__(self,num_of_batches=20,num_of_patches=8,seq_latent_space=27,size_r=700,size_l=250,patch_size=20):
        super(make_blocks,self).__init__()
        self.num_of_batches=num_of_batches
        self.num_of_patches = num_of_patches
        self.seq_latent_space=seq_latent_space
        self.patch_size=patch_size
        self.size_r=size_r
        self.size_l=size_l

    def call(self,input ,traning=False):
        seq1M, seq2M, patches, geo=input
        tf.ensure_shape(seq1M,(None,self.size_r,self.seq_latent_space))
        tf.ensure_shape(seq2M,(None,self.size_l,self.seq_latent_space))
        tf.ensure_shape(geo,(None,self.num_of_patches,self.patch_size,self.patch_size))
        tf.ensure_shape(patches, (None, self.num_of_patches,2))
        batches_of_blocks = []
        for i in range(self.num_of_batches):
            rows = cut_patches(seq1M[i], patches[i], num_of_patches=self.num_of_patches,patch_size=self.patch_size)
            cols = cut_patches(seq2M[i], patches[i], num_of_patches=self.num_of_patches,patch_size=self.patch_size, horizontial=False)
            tf.ensure_shape(rows,(self.num_of_patches,self.patch_size,self.seq_latent_space))
            tf.ensure_shape(rows, (self.num_of_patches,self.patch_size, self.seq_latent_space))
            batches_of_blocks.append(to_blocks(rows, cols, geo[i], num_of_blocks=self.num_of_patches
                                               ,seq_lataent_space=self.seq_latent_space,patch_size=self.patch_size))
        return tf.stack(batches_of_blocks)

class dist_normalized(layers.Layer):
    def __init__(self,num_of_batches=20,num_of_patches=6):
        super(dist_normalized,self).__init__()


    def call(self,input ,traning=False):
        return tf.map_fn(lambda b: tf.map_fn(lambda p: apply_dists_on_block(p), b), input)

def patches_for_batches(seq1M,seq2M,patches,geo,num_of_batches=20,num_of_patches=6,horizontial=True):
    batches_of_blocks=[]
    for i in range(num_of_batches):
        rows=cut_patches(seq1M[i],patches[i],num_of_patches=num_of_patches,horizontial=horizontial)
        cols=cut_patches(seq2M[i],patches[i],num_of_patches=num_of_patches,horizontial=False)
        batches_of_blocks.append(to_blocks(rows, cols, geo[i], num_of_blocks=num_of_patches))
    return tf.stack(batches_of_blocks)


@tf.function
def apply_dists_on_block(block):
    chem_block=block[:,:,:-1]
    distogram=tf.expand_dims(block[:,:,-1],axis=2)
    distogram_tiled=tf.tile(distogram,(1,1,chem_block.shape[2]))
    normelized=distogram_tiled*chem_block
    return normelized


def build_single_prot_stage(seq1,geo_seq1,seq2,geo_seq2,patches,geo,config,training=True):
    projections = Dense(config["arch"]["seq_latent_dim"],trainable=training)
    seq1M = projections(seq1)
    seq2M = projections(seq2)
    for i in range(config["arch"]["number_of_1dtransformer"]):
        seq1M = multiheaded_chem_transformer1D(size=config["arch"]["seq_latent_dim"], number_of_heads=config["arch"]["heads_1d"],training=training )(
            [seq1M, geo_seq1])
        seq2M = multiheaded_chem_transformer1D(size=config["arch"]["seq_latent_dim"], number_of_heads=config["arch"]["heads_1d"],training=training)(
            [seq2M, geo_seq2])

    meraged_data = make_blocks(num_of_batches=config["hyper"]["batch_size"], seq_latent_space=config["arch"]["seq_latent_dim"],
                               num_of_patches=config["arch"]["number_of_patches"],
                               patch_size=config["arch"]["patch_size"],size_r=config["arch"]["size_r"],size_l=config["arch"]["size_l"])([seq1M, seq2M, patches, geo])
    normlized = dist_normalized()(meraged_data)
    return normlized

def build_interaction_stage(normlized,config,training=True):
    conv_filters = config["arch"]["conv_filters"]
    pooling = config["arch"]["pooling_layers"]
    kernal_size=config["arch"]["kernal_size"]
    conv_layer1 = ResBlock2Dv2(conv_filters[0], kernal=kernal_size[0], pool=pooling[0],trainable=training)
    x = layers.TimeDistributed(conv_layer1)(normlized)
    conv_layer2 = ResBlock2Dv2(conv_filters[1], kernal=kernal_size[1], pool=pooling[1],trainable=training)
    x = layers.TimeDistributed(conv_layer2)(x)
    conv_layer3 = ResBlock2Dv2(config["arch"]["graph_latent_dim"], kernal=kernal_size[2], pool=pooling[2],global_p=config["arch"]["global_pool"],trainable=training)
    x = layers.TimeDistributed(conv_layer3)(x)
    x = tf.concat([x, tf.ones((config["hyper"]["batch_size"], 1,config["arch"]["graph_latent_dim"]))], axis=1)
    for i in range(config["arch"]["number_of_2dtransformer"]):
        x = multiHeader(size=config["arch"]["graph_latent_dim"], num_heads=config["arch"]["heads_2d"],training=training)(x)
    labels = x[:, config["arch"]["number_of_patches"], :]
    return labels


def build_dockQ(x,dockq_classes=10):
    x=Dense(dockq_classes*2,activation="tanh")(x)
    return Dense(dockq_classes,activation="softmax",name="dockQ")(x)


def build_patch_model2(number_of_patches,config,training=True,training_dockQ_head=True):

    seq1 = Input(shape=(None, 25))
    geo_seq1 = Input(shape=(None,None))
    seq2 = Input(shape=(None, 25))
    geo_seq2 = Input(shape=(None, None))
    geo = Input(shape=(number_of_patches, config["arch"]["patch_size"],config["arch"]["patch_size"]))
    patches = Input(shape=(number_of_patches, 2),dtype=tf.int32)
    normalized=build_single_prot_stage(seq1,geo_seq1,seq2,geo_seq2,patches,geo,config,training=training)
    labels=build_interaction_stage(normalized,config,training=training)
    x=smallMLP3(size=config["arch"]["class_predictor"],name="label_predictor",training=training)(labels)
    out=Dense(1,activation='sigmoid',name="classification",trainable=training)(x)
    predicted_dockQ=tf.keras.layers.Softmax(name="dockQ")(smallMLP3(size=config["arch"]["dockQ_predictor"],name="dockQ_predictor",training=training_dockQ_head)(labels))
    predicted_lig_rmsd=Dense(1,activation='relu',name="lig_rmsd",trainable=training)(x)
    return Model(inputs=[seq1,geo_seq1, seq2,geo_seq2, geo,patches],
                 outputs={"classification":out,"lig_rmsd":predicted_lig_rmsd,"dockQ":predicted_dockQ})


