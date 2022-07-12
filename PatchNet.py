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


# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.config.run_functions_eagerly(False)
# tf.config.functions_run_eagerly()
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[1],'GPU')

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
#from helper import plot_metrics
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
    # tf.ensure_shape(seq1M,(None,size_r,25))
    # tf.ensure_shape(seq1M,(None,size_r,25))
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
    # conv_layer4 = ResBlock2Dv2(conv_filters[3], kernal=kernal_size[3], pool=pooling[3])
    # x = layers.TimeDistributed(conv_layer4)(x)
    # conv_layer5 = ResBlock2Dv2(conv_filters[4], kernal=3,pool=pooling[4])
    # x = layers.TimeDistributed(conv_layer5)(x)
    # x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]))(x)
    # x = encoder(size=config["arch"]["graph_latent_dim"], drop_rate=config["arch"]["encoder_drop_rate"])(x)
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
    x=smallMLP3(size=config["arch"]["class_predictor"],name="label_predictor",training=training)(labels)#20,30
    # x=Dense(40,activation='relu')(labels)#was 80
    # x=layers.BatchNormalization()(x)
    # x=Dense(30,activation='relu')(x)# was 30
    # x=layers.BatchNormalization()(x)
    # x=Dense(20,activation='relu')(x)
    out=Dense(1,activation='sigmoid',name="classification",trainable=training)(x)
    # predicted_dockQ =Dense(10,activation="softmax")(x)
    predicted_dockQ=tf.keras.layers.Softmax(name="dockQ")(smallMLP3(size=config["arch"]["dockQ_predictor"],name="dockQ_predictor",training=training_dockQ_head)(labels))
    # predicted_dockQ=build_dockQ(x)
    # predicted_rec_rmsd=Dense(1,activation='relu',name="rec_rmsd")(x)
    predicted_lig_rmsd=Dense(1,activation='relu',name="lig_rmsd",trainable=training)(x)
    return Model(inputs=[seq1,geo_seq1, seq2,geo_seq2, geo,patches],
                 outputs={"classification":out,"lig_rmsd":predicted_lig_rmsd,"dockQ":predicted_dockQ})
    # return Model(inputs=[seq1,geo_seq1, seq2,geo_seq2, geo,patches], outputs={"classification":out,"lig_rmsd":predicted_lig_rmsd})



Metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name="tp_rate"),
    tf.keras.metrics.FalsePositives(name="fp_rate"),
    tf.keras.metrics.TrueNegatives(name="tn_rate"),
    tf.keras.metrics.FalseNegatives(name="fn_rate"),
    tf.keras.metrics.AUC(name="AUC"),
    tf.metrics.Precision(name="precision",thresholds=0.3),
    tf.metrics.Recall(name="recall",thresholds=0.3),
]

@tf.function
def train_step(x,y,model):
    with tf.GradientTape() as tape:
        logit = model(x, training=True)
        loss_val = binary_crossentropy_relaxed(y, logit)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print(optimizer)
    return loss_val

# def update_metrics
def train(model,gentrator,num_of_epochs,optimizer,train_steps=60):
    iterator=gentrator.__iter__()
    for epoch in range(num_of_epochs):
        for i in range(train_steps):
            x,y =iterator.__next__()
            loss_val=train_step(x,y,model)
        print("train loss at epoch %d : %.4f "%(epoch,float(loss_val.numpy())))
        wandb.log({"loss":loss_val.numpy()})



test_prot="/cs/labs/dina/matanhalfon/CAPRI/"+config["workdir"]+"/test_pdbs"+config["pdb_file_suffix"]
train_prot="/cs/labs/dina/matanhalfon/CAPRI/"+config["workdir"]+"/train_pdbs"+config["pdb_file_suffix"]


if __name__ == '__main__':

    if len(sys.argv) == 1:
        trains_file = config["data_file"]
        lr = config["hyper"]["lr"]
        opt = config["optimizer"]
        transformer_1d = config["arch"]["number_of_1dtransformer"]
        transformer_2d = config["arch"]["number_of_2dtransformer"]
        # is_seq=False
    else:
        trains_file = sys.argv[1]
        EPOCHS = int(sys.argv[2])
        lr = float(sys.argv[3])
        opt = int(sys.argv[4])
        transformer_1d = int(sys.argv[5])
        transformer_2d = int(sys.argv[6])
    file_name = "lr-" + str(lr) + "_" +str(opt)+"_"+ trains_file + "_transformer_"+config["workdir"].split("/")[0]+config["model_suffix"]
    veifyDir(osp.join("NNscripts", file_name))

    train_prot_dict=build_prot_dict(train_prot,config["arch"]["size_r"],config["arch"]["size_l"],
                                    workdir=config["workdir"].split("/")[0],line_len_R=config["line_len_R"],line_len_L=config["line_len_L"],
                                    suffixL=config["suffix_l"],suffixR=config["suffix_r"],self_distogram=config["self_distogram_dir"])
    test_prot_dict=build_prot_dict(test_prot,config["arch"]["size_r"],config["arch"]["size_l"],
                                   workdir=config["workdir"].split("/")[0],line_len_R=config["line_len_R"],line_len_L=config["line_len_L"]
                                   ,suffixL=config["suffix_l"],suffixR=config["suffix_r"],self_distogram=config["self_distogram_dir"])
    types=({"input_1": tf.float32, "input_2": tf.float32, "input_3": tf.float32,
                          "input_4": tf.float32,"input_5": tf.float32,"input_6":tf.int32
            },
             {"classification":tf.float32,
            "dockQ":tf.float32,
              "lig_rmsd":tf.float32,
              # "names":tf.string
              })
    data=tf.data.Dataset.from_generator(lambda :gen_cycler("balanced", trains_file,prot_dict=train_prot_dict,sample_rate=config["hyper"]["sample_rate"], workdir=config["workdir"],
                                      batch_size=config["hyper"]["batch_size"],
                                    size_r=config["arch"]["size_r"], size_l=config["arch"]["size_l"],patch_size=config["arch"]["patch_size"],data_dir=config["patch_dir"],
                                    title_len=config["line_len_R"],data_postfix=config["data_type_train"],batched=True), output_types=types).prefetch(30)
    types_val = ({"input_1": tf.float64, "input_2": tf.float32, "input_3": tf.float64,
              "input_4": tf.float32, "input_5": tf.float32, "input_6": tf.int32,
                  },
             {"classification": tf.float32,
              "dockQ": tf.float32,
              "lig_rmsd": tf.float32})
    print(config["workdir"]+"-"+"test_trans"+"-"+config["data_type_train"])
    data_validation = tf.data.Dataset.from_generator(
        lambda: gen_cycler("comp","test_trans",prot_dict=test_prot_dict,workdir=config["workdir"],
                           batch_size=config["hyper"]["batch_size"],data_postfix="_"+config["data_type_test"],patch_size=config["arch"]["patch_size"],
                           size_r=config["arch"]["size_r"], size_l=config["arch"]["size_l"],data_dir=config["patch_dir_complex"]
                           ,title_len=config["line_len_R"],batched=True), output_types=types_val).prefetch(30)
    model = build_patch_model2(config["arch"]["number_of_patches"],config)
    model.summary()
    if opt == 1:
        optimizer = tf.keras.optimizers.Adam(epsilon=1e-8, learning_rate=lr_schedule)
    elif opt == 2:
        optimizer = tf.keras.optimizers.Nadam(epsilon=1e-7, learning_rate=lr)
    else:
        print("yolo!!!")
        step = tf.Variable(0, trainable=False)
        # schedule=WarmUp(initial_learning_rate=config["hyper"]["cosine_start"],
        #                 decay_schedule_fn=tf.keras.optimizers.schedules.CosineDecayRestarts(config["hyper"]["cosine_start"],
        #                 config["hyper"]["cosine_steps"],alpha=config["hyper"]["cosine_alpha"],t_mul=config["hyper"]["t_mul"],
        #                 m_mul=config["hyper"]["m_mul"]),warmup_steps=100)
        schedule = WarmUp(initial_learning_rate=config["hyper"]["cosine_start"],
                          decay_schedule_fn=tf.keras.optimizers.schedules.CosineDecay(
                              config["hyper"]["cosine_start"],
                              config["hyper"]["cosine_steps"], alpha=config["hyper"]["cosine_alpha"],), warmup_steps=600)
        wd = lambda x: config["wd"] * schedule(x)
        optimizer = tfa.optimizers.AdamW(learning_rate=schedule, weight_decay=lambda : None)
        # optimizer = tfa.optimizers.SGDW(learning_rate=schedule, weight_decay=wd,momentum=0.9)
        optimizer.weight_decay = lambda: wd(optimizer.iterations)
        optimizer=tfa.optimizers.SWA(optimizer)
    run = wandb.init(reinit=True, config=config,job_type="train")

    lossWeights = {"classification": config["loss_weights"]["classification"],"dockQ":config["loss_weights"]["dockQ"], "lig_rmsd":config["loss_weights"]["lig_rmsd"]}
    # lossWeights = {"classification": config["loss_weights"]["classification"],"lig_rmsd":config["loss_weights"]["lig_rmsd"]}

    metrics={"classification":Metrics}
    model.compile(optimizer=optimizer, loss=config["losses_type"], loss_weights=lossWeights,metrics=metrics)
    print("######################started######################")
    callbacks = [
        ModelCheckpoint(
            filepath=osp.join("NNscripts", file_name + "/mymodel_{epoch}"),
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            # monitor="val_classification_loss",
            monitor="val_classification_precision",
            mode='max',
            verbose=1,
            save_weights_only=True
        ),
        WandbCallback(),
        # LRLogger(optimizer,config["hyper"]["steps_per_epoch"])
    ]
    history = model.fit(data, validation_data=data_validation, steps_per_epoch=config["hyper"]["steps_per_epoch"],max_queue_size=32,
                        validation_steps=int(config["hyper"]["steps_per_epoch"]/4),
                        epochs=config["hyper"]["num_of_epochs"],verbose=2,callbacks=callbacks,workers=16,validation_freq=1)
    # history = model.fit(data, steps_per_epoch=config["hyper"]["steps_per_epoch"],max_queue_size=32,
    #             epochs=config["hyper"]["num_of_epochs"],verbose=2,callbacks=callbacks,workers=14,validation_freq=1)
    print("######################finished######################")

    # plot_metrics(history,
    #              osp.join("NNscripts", file_name, "model_metrics" + str(lr) + "-opt" + str(opt) + "_transformer"))
    # model.evaluate(data_validation, steps=config["hyper"]["steps_per_epoch"] * 4, verbose=2)
    # model.save(osp.join("NNscripts",file_name,"PatchNet_"+str(trains_file)+"_"+str(opt)+"_"+str(transformer_1d)))
    # model.save_weights(osp.join("NNscripts",file_name,"DockNet_"+str(trains_file)+"_"+str(opt)+"_"+str(is_seq)+".weight"))
