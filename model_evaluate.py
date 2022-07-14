import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers, Model ,Input
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, Add, Activation, \
    MaxPool2D, BatchNormalization, Permute, Multiply, Reshape ,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocessing
import matplotlib.pyplot as plt
import sys
import random as rd
import  os
import seaborn as sn
import time
import os.path as osp
import math
import  matplotlib

from PatchNet import build_patch_model2
from generators import file_genarator_pre_batched
from preprocessing import preprosser
import bisect
import matplotlib.pyplot as plt
import gc
import  pandas as pd
import csv

def load_modle(path,config):
    model=build_patch_model2(8,config)
    model.load_weights(path).expect_partial()
    return model


def culc_hitrate(table,k):
    top_k=table[:k,:]
    return np.sum(top_k[:,0].astype(float))/k


def hit_rate(model,prot_ag,prot_ab,config,data_dir="",trans_num=0):
    """
    :param model:  NN model
    :param workdir: dir where trans_file is located
    :param trans_file:  the transoformation file
    :param prot_dict: an chach prot_dict
    :param data_dir:  where the bached patched  are locatd
    :param batch_size: the size of baches
    :param ratio: to present the ratio of acc solutions
    :return:
    """
    i=0
    gen = file_genarator_pre_batched(prot_ag,prot_ab, batch_size=config["hyper"]["batch_size"],
                            patch_size=config["arch"]["patch_size"], size_r=config["arch"]["size_r"], size_l=config["arch"]["size_l"],
                            data_dir=data_dir,single_batched=True,trans_num=trans_num)

    score_by_label=None
    try:
        for b in gen:
            scores=model.predict(b)["classification"]
            print(scores)
            if score_by_label is None:
                score_by_label=scores
            else:
                score_by_label = np.concatenate([score_by_label, scores])
            i += 1
            if (i%10)==0:
                print("batch_num: ",i)
    except EOFError:
        print("finished with "+trans_file )
    gc.collect()
    trans=pd.read_csv(osp.join("PPI","trans.txt"),sep="\t",index_col=0).to_numpy().reshape((-1,1))[:len(score_by_label)]
    score_by_label = np.concatenate([score_by_label, trans],axis=1)
    df=pd.DataFrame(score_by_label,columns=["score","trans"])
    df.to_csv("evaluation", sep="\t", quoting=csv.QUOTE_NONE,
               quotechar="", escapechar="\\")


def get_hit_rates_batched(model_path,test_pdb,test_dir,config):
    test_prot_dict = build_prot_dict(test_pdb,config["arch"]["size_r"],config["arch"]["size_l"],
                    workdir=config["workdir"].split("/")[0],line_len_R=config["line_len_R"],
                    line_len_L=config["line_len_L"],self_distogram=config["self_distogram_dir"],
                    suffixL=config["suffix_l"],suffixR=config["suffix_r"])
    model = load_modle(model_path,config)
    model.summary()
    nnm = []
    with open(test_pdb) as f:
        print(test_dir)
        for prot_dir in os.listdir(test_dir):
            print(osp.join(test_dir,"midfiles",prot_dir))
            if prot_dir+config["suffix_r"] not in test_prot_dict.keys():
                print(prot_dir)
                continue
            try:
                nnm.append(
                    hit_rate(model, osp.join(test_dir,"midfiles",prot_dir), prot_dir,config,  data_dir=osp.join(test_dir,prot_dir),
                             prot_dict=test_prot_dict,ratio=False))
            except OverflowError:
                print("overflow at : ", file)
                continue
            except FileNotFoundError:
                    print(file, " not found")
                    continue
        print(nnm)



if __name__ == '__main__':
    model_path=sys.argv[1]
    test_pdb=sys.argv[2]
    test_dir=sys.argv[3]
    get_hit_rates_batched(model_path,test_pdb,test_dir,config_evaluate)
