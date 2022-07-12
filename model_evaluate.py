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
from generators import complex_genarator,build_prot_dict,gen_cycler,file_genarator_pre_batched
from preprocessing import preprosser
import bisect
import matplotlib.pyplot as plt
from helper import get_pdb_from_file
import gc
import  pandas as pd
import csv

config_evaluate=dict(
    # self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distogram_alphafold",
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/AlphaFold_splited_modeled/self_distograms/",
    line_len_R=14,
    line_len_L=14,
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_nano",
    trans_dir="ABDB/AlphaFold_trans_dir_test_2/",
    # trans_dir="ABDB/AlphaFold_docking_trans/",
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_Alphafold",
    # patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data_Alphafold/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=44,  # 65
        graph_latent_dim=70,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=5,  # 7
        number_of_2dtransformer=4,  # 5
        size_l=700,
        size_r=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=3,
        heads_2d=5,
        conv_filters=[50, 40, 35, 35, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True,
        class_predictor=[40, 30, 20],
        dockQ_predictor=[40, 30, 5]
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        batch_size=30,
    ),
    workdir="ABDB",
    suffix_r="_Ab",
    suffix_l="_Ag",
    optimizer=5,
)

config_evaluate_AlphaFold=dict(
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/AlphaFold_splited_modeled/self_distograms",
    line_len_R=14,
    line_len_L=14,
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_nano",
    trans_dir="ABDB/AlphaFold_trans_dir/",
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_Alphafold",
    # patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data_Alphafold/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=80,  # 65
        graph_latent_dim=120,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=2,  # 7
        number_of_2dtransformer=3,  # 5
        size_r=700,
        size_l=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=4,
        heads_2d=6,
        conv_filters=[50, 40, 35, 35, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True,
        class_predictor=[40, 30, 20],
        dockQ_predictor=[40, 20, 10]
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        batch_size=1,
    ),
    workdir="ABDB",
    suffix_r="_Ab",
    suffix_l="_Ag",
    optimizer=5,
)

config_evaluate_db5=dict(
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/bench5AA/self_distograms",
    line_len_R=4,
    line_len_L=4,
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_nano",
    trans_dir="bench5AA/db5_trans_dir_test",
    # patch_dir="/cs/labs/dina/matanhalfon/bench5AA/patch_data_db5/",
    # patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data_Alphafold/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=45,  # 65
        graph_latent_dim=66,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=4,  # 7
        number_of_2dtransformer=4,  # 5
        size_l=700,
        size_r=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=3,
        heads_2d=6,
        conv_filters=[50, 40, 35, 35, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True,
        class_predictor=[40, 30, 20],
        dockQ_predictor=[40, 30, 10]
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        batch_size=20,
    ),
    workdir="bench5AA",
    suffix_r="_Ab",
    suffix_l="_l_u",
    optimizer=5,
)

config_saxs=dict(
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/saxs/saxs_self_distograms",
    line_len_R=8,
    line_len_L=8,
    trans_dir="saxs/saxs_trans_dir_test",
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_nano",
    # patch_dir="/cs/labs/dina/matanhalfon/CAPRI/saxs/patch_data_saxs",
    # patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data_Alphafold/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=80,  # 65
        graph_latent_dim=120,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=2,  # 7
        number_of_2dtransformer=3,  # 5
        size_l=700,
        size_r=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=4,
        heads_2d=6,
        conv_filters=[50, 40, 35, 35, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True,
        class_predictor=[40, 30, 20],
        dockQ_predictor=[40, 20, 10]
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        batch_size=20,
    ),
    workdir="saxs",
    suffix_r="_sFab",
    suffix_l="_Ag",
    optimizer=5,
)

config_spec=dict(
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distogram_alphafold",
    line_len_R=14,
    line_len_L=14,
    trans_dir="ABDB/specifisity_trans_dir",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=48,  # 65
        graph_latent_dim=72,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=4,  # 7
        number_of_2dtransformer=4,  # 5
        size_l=700,
        size_r=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=3,
        heads_2d=6,
        conv_filters=[50, 40, 35, 35, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True,
        class_predictor=[40, 30, 20],
        dockQ_predictor=[40, 20, 10]
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        batch_size=20,
    ),
    workdir="ABDB",
    suffix_r="_Ab",
    suffix_l="_Ag",
    optimizer=5,
)

def load_modle(path,config):
    model=build_patch_model2(8,config)
    model.load_weights(path).expect_partial()
    return model


def culc_hitrate(table,k):
    top_k=table[:k,:]
    return np.sum(top_k[:,0].astype(float))/k


def hit_rate(model, workdir,trans_file,config, prot_dict=None,data_dir="",ratio=False):
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

    if osp.getsize(osp.join(workdir,trans_file))==0:
        print(trans_file,": file is empty")
        return 0
    i=0
    gen = file_genarator_pre_batched(trans_file,  data_postfix="", prot_dict=prot_dict, batch_size=config["hyper"]["batch_size"],
                            patch_size=config["arch"]["patch_size"], size_r=config["arch"]["size_r"], size_l=config["arch"]["size_l"],
                            data_dir=data_dir, title_len=config["line_len_R"],evaluate=True,
                            workdir=workdir,finite=True,single_batched=True)

    score_by_label=None
    try:

        for b,labels in gen:
            scores=model.predict(b)
            meraged = np.concatenate([np.array(labels["classification"]).reshape((-1, 1)),
                                      scores["classification"],
                                      np.around(np.array(labels["rec_rmsd"]).reshape((-1, 1)).astype(np.float32),3),
                                      np.around(np.array(labels["lig_rmsd"]).reshape((-1, 1)).astype(np.float32),3),
                                      np.around(np.array(labels["dockQ"]/100).reshape((-1, 1)).astype(np.float32), 3),
                                      (np.argmax(scores["dockQ"],axis=1)*0.1).reshape((-1, 1)),
                                      np.array(labels["names"]).reshape((-1, 1)) ], axis=1)
            if score_by_label is None:
                score_by_label=meraged
            else:
                score_by_label = np.concatenate([score_by_label, meraged])
            i += 1
            if (i%10)==0:
                print("batch_num: ",i)
    except EOFError:
        print("finished with "+trans_file )
    gc.collect()
    trans=pd.read_csv(osp.join(config["trans_dir"],trans_file),sep="\t").iloc[:,3].to_numpy().reshape((-1,1))[:len(score_by_label)]
    # trans=pd.read_csv(osp.join(config["trans_dir"],trans_file),sep="\t")["transformation"].to_numpy().reshape((-1,1))
    score_by_label = np.concatenate([score_by_label, trans],axis=1)
    score_by_label=score_by_label[np.flip(np.argsort(score_by_label[:, 1]))]
    df=pd.DataFrame(score_by_label,columns=["label","score","rec_rmsd","lig_rmsd","dockQ","predicted_dockQ","name","trans"])
    # df=pd.DataFrame(score_by_label,columns=["label","score","rec_rmsd","lig_rmsd","dockQ","predicted_dockQ","name"])
    df.to_csv(osp.join(workdir,"evaluation_"+trans_file), sep="\t", quoting=csv.QUOTE_NONE,
               quotechar="", escapechar="\\")
    # if not (((score_by_label[:,2].astype(float)<5 )|(score_by_label[:,3].astype(float)<2)).any()):
    #     print("XXXXX no mid solutions")
    if np.any((score_by_label[:,2].astype(float)<10 )|(score_by_label[:,3].astype(float)<5)):
        if ratio:
            labels=((score_by_label[:,2].astype(float)<10 )|(score_by_label[:,3].astype(float)<5)).astype(float)>0
            print(labels[:10])
            hit_ratio=round((np.cumsum(labels)/np.arange(1,labels.shape[0]+1))[10],4)
            print(hit_ratio)
            return hit_ratio
        else:
            try:
                first_near_native=np.where((score_by_label[:,2].astype(float)<=10 )|(score_by_label[:,3].astype(float)<=4))[0][0]
            except IndexError:
                first_near_native=2500
            print("first hit at ",first_near_native)
            # print(culc_hitrate(score_by_label,i))
            # df=pd.DataFrame(score_by_label,columns=["label","score","rec_dist","lig_dist","name"])
            # df=df.round({'rec_dist':3,"lig_dist":3})
            # df.to_csv(osp.join(wordkdir,"predictions",trans_file))
            return first_near_native
    else:
        print("all trans are neg ")
        if ratio: return 0
        else: return 2500


def count_solutions(test_prot,pdbs_dir):
    docking="docking.res"
    test_pdb = get_pdb_from_file(test_prot)
    acc_souls=[]
    mid_souls=[]
    exe_souls=[]
    for prot_dir in os.listdir(pdbs_dir):
        if prot_dir in test_pdb:
            df = dock_to_pandas(osp.join(pdbs_dir, prot_dir, docking))
            acc_solution=df[(df["rec_rmsd"] < 10) | (df["lig_rmsd"] < 4)].shape[0] / (df.shape[0] / 10)
            acc_souls.append(acc_solution)
            mid_solution=df[(df["rec_rmsd"] < 5) | (df["lig_rmsd"] < 2)].shape[0] / (df.shape[0] / 10)
            mid_souls.append(mid_solution)
            exealent_solution=df[(df["rec_rmsd"] < 2) | (df["lig_rmsd"] < 1)].shape[0] / (df.shape[0] / 10)
            exe_souls.append(exealent_solution)
            print(prot_dir, "acc: ",acc_solution,"mid: ",mid_solution,"exe: ",exealent_solution)
    print("#average-  ", "acc: ", sum(acc_souls)/len(acc_souls), "mid: ", sum(mid_souls)/len(mid_souls), "exe: ", sum(exe_souls)/len(exe_souls))

#
# def get_soap_hitrate(test_prot,pdbs_dir,rates=False):
#     soap_file="soap_score.res"
#     asses="assess.res"
#     docking="docking.res"
#     hit_rates=[]
#     test_pdb=get_pdb_from_file(test_prot)
#     for prot_dir in os.listdir(pdbs_dir):
#             if prot_dir in test_pdb:
#                 try:
#                     df=soap_to_pandas(osp.join(pdbs_dir,prot_dir,soap_file))
#                     df1=dock_to_pandas(osp.join(pdbs_dir,prot_dir,docking))
#                     meraged=pd.concat([df,df1],axis=1,join="inner")
#                     meraged=meraged.reset_index(drop=True)
#
#                 except FileNotFoundError:
#                     print(prot_dir)
#                     continue
#                 if (meraged["rec_rmsd"]<10).any() or (meraged["lig_rmsd"]<4).any():
#                     # mid = meraged[(meraged["rec_rmsd"] < 5) | (meraged["lig_rmsd"] < 2)]
#                     # print(mid.shape[0])
#                     if rates:
#                         labels = (meraged["rec_rmsd"] < 10) | (meraged["lig_rmsd"] < 4)
#                         print(labels[:10])
#                         hit_ratio = (np.cumsum(labels) / np.arange(1, labels.shape[0] + 1))[10]
#                         print(hit_ratio)
#                         hit_rates.append(hit_ratio)
#                     else:
#                         if (meraged["rec_rmsd"] < 10).any():
#                             hit_rates.append(meraged[meraged["rec_rmsd"].le(10)].index[0])
#                         elif (meraged["lig_rmsd"]<4).any():
#                             hit_rates.append(meraged[meraged["lig_rmsd"].le(4)].index[0])
#                 else:
#                     hit_rates.append(-1)
#                     print("non mid")
#     print(hit_rates.count(-1))
#     print(hit_rates)
#


def get_hitraes(model_path,test_pdb):
    test_prot_dict = build_prot_dict(test_pdb,config["arch"]["size_r"],config["arch"]["size_l"],workdir=config["workdir"].split("/")[0],line_len_R=config["line_len_R"],line_len_L=config["line_len_L"],
                                     self_distogram=config["self_distogram_dir"],suffixL=config["suffix_l"],suffixR=config["suffix_r"])
    model = load_modle(model_path)
    model.summary()
    nnm = []
    with open(test_pdb) as f:
        test_files = get_pdb_from_file(test_pdb)
        print(tr_dir)
        for file in os.listdir(tr_dir):
            if file+config["suffix_r"] not in test_prot_dict.keys():
                continue
            if file in test_files:
                try:
                    nnm.append(
                        hit_rate(model, tr_dir, file,  prot_dict=test_prot_dict,ratio=False))
                except OverflowError:
                    print("overflow at : ", file)
                    continue
        print(nnm)

def get_hit_rates_batched(model_path,test_pdb,test_dir,config):
    test_prot_dict = build_prot_dict(test_pdb,config["arch"]["size_r"],config["arch"]["size_l"],
                    workdir=config["workdir"].split("/")[0],line_len_R=config["line_len_R"],
                    line_len_L=config["line_len_L"],self_distogram=config["self_distogram_dir"],
                    suffixL=config["suffix_l"],suffixR=config["suffix_r"])
    model = load_modle(model_path,config)
    model.summary()
    nnm = []
    with open(test_pdb) as f:
        # test_files = get_pdb_from_file(test_pdb)
        print(test_dir)
        for prot_dir in os.listdir(test_dir):
            print(osp.join(test_dir,"midfiles",prot_dir))
            if prot_dir+config["suffix_r"] not in test_prot_dict.keys():
                print(prot_dir)
                continue
            # if not prot_dir in test_files:
            #     print(prot_dir)
            #     continue
            try:
                nnm.append(
                    hit_rate(model, osp.join(test_dir,"midfiles",prot_dir), prot_dir,config,  data_dir=osp.join(test_dir,prot_dir),
                             prot_dict=test_prot_dict,ratio=False))
            except OverflowError:
                print("overflow at : ", file)
                continue
                # except FileNotFoundError:
                #     print(file, " not found")
                #     continue
        # nnm=np.array(nnm)
        print(nnm)

def get_specifisity_batched(model_path,test_pdb,test_dir,config):
    test_prot_dict = build_prot_dict(test_pdb,config["arch"]["size_r"],config["arch"]["size_l"],
                    workdir=config["workdir"].split("/")[0],line_len_R=config["line_len_R"],
                    line_len_L=config["line_len_L"],self_distogram=config["self_distogram_dir"],
                    suffixL=config["suffix_l"],suffixR=config["suffix_r"])
    model = load_modle(model_path, config)
    model.summary()
    nnm = []
    with open(test_pdb) as f:
        test_files = get_pdb_from_file(test_pdb)
        print(test_dir)
        for prot_dir in os.listdir(test_dir):
            # print(osp.join(test_dir,"midfiles",prot_dir))
            # if prot_dir+config["suffix_r"] not in test_prot_dict.keys():
            #     continue
            if (prot_dir[:6]+"_model_0" in test_files) and (prot_dir[7:]+"_model_0" in test_files):
                print(prot_dir)
                try:
                    nnm.append(
                        hit_rate(model, osp.join(test_dir,"midfiles",prot_dir), prot_dir,config , data_dir=osp.join(test_dir,prot_dir),
                                 prot_dict=test_prot_dict, batch_size=Batch_size,ratio=False))
                except OverflowError:
                    print("overflow at : ", file)
                    continue
                # except FileNotFoundError:
                #     print(file, " not found")
                #     continue
            else :
                print("Error at folder "+prot_dir)
        # nnm=np.array(nnm)
        print(nnm)

# tr_dir="dockground/dockground_trans_dir"
# tr_dir="dockground/dockground_trans_dir_oversampled"
# tr_dir="ABDB/AlphaFold_trans_dir_test"
# tr_dir="ABDB/Alphafold_test_patch_data"
# tr_dir="ABDB/specifisity_trans_dir"
# tr_dir="ABDB/ABDB_trans_dir_fine_nano_test"
# seq_dir="dockground"
# structdir="dockground_1"

# tr_dir="AA_data/AA_trans"
# seq_dir="AA_data"
# structdir="AA_trans"

# structdir="ABDB"
# tr_dir="ABDB/docking_trans_fine"
# seq_dir="ABDB"

Batch_size=20
# data_dir="/cs/labs/dina/matanhalfon/patch_data"
# data_dir="/cs/labs/dina/matanhalfon/CAPRI/patch_data_dockground_bound"
# data_dir="/cs/labs/dina/matanhalfon/patch_data_AA"
# data_dir="/cs/labs/dina/matanhalfon/CAPRI/patch_data_ABDB_nano"
# data_dir="/cs/labs/dina/matanhalfon/CAPRI/dock_test/patches_reg"


#small_model
split1_small=[1, 5, 21, 0, 8, 0, 135, 0, 28, 0, 1, 0, 12, 22, 500, 500, 22, 0, 0, 21, 1, 0, 500, 0, 25, 20, 0, 0, 0, 500, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 7, 14, 30, 10, 2]
split2_small=[6, 10, 0, 3, 0, 0, 0, 25, 0, 16, 500, 0, 0, 4, 0, 1, 7, 7, 4, 1, 0, 352, 11, 2, 4, 1, 500, 2, 239, 0, 17, 0, 16, 0, 123, 500, 500, 0, 404, 17, 7, 0, 0, 27, 0]
split3_small=[0, 500, 3, 0, 308, 0, 4, 0, 2, 3, 0, 12, 3, 18, 1, 0, 0, 4, 0, 0, 1, 0, 500, 0, 2, 18, 16, 500, 2, 8, 0, 1, 59, 2, 449, 66, 500, 634, 0, 1, 5, 53, 0, 500, 66]
split4_small=[0, 0, 69, 15, 500, 100, 346, 0, 0, 4, 46, 8, 221, 4, 3, 500, 5, 3, 99, 37, 79, 1, 1, 8, 71, 0, 2, 13, 500, 8, 0, 0, 70, 34, 22, 0, 28, 0, 14, 500, 13, 0, 1, 500, 2]
split5_small=[0, 20, 13, 8, 13, 500, 1, 500, 9, 500, 6, 500, 14, 500, 6, 6, 24, 5, 1, 438, 500, 1, 25, 1, 0, 1, 12, 4, 22, 148, 8, 2, 6, 500, 9, 5, 43, 1, 500, 1, 4, 500, 500, 2, 1]

# split1_rates=[0.36363636363636365, 0.18181818181818182, 0.2727272727272727, 0.9090909090909091, 0.6363636363636364, 1.0, 0.0, 0.36363636363636365, 0.0, 1.0, 0.2727272727272727, 0.8181818181818182, 0.0, 0.0, 0, 0, 0.0, 0.18181818181818182, 0.2727272727272727, 0.09090909090909091, 0.36363636363636365, 1.0, 0, 0.36363636363636365, 0.36363636363636365, 0.0, 0.36363636363636365, 0.9090909090909091, 1.0, 0, 0.5454545454545454, 0.0, 0.36363636363636365, 0.0, 0.2727272727272727, 0.6363636363636364, 0.0, 0.18181818181818182, 0.45454545454545453, 0.45454545454545453, 0.5454545454545454, 0.09090909090909091, 0.09090909090909091, 0.36363636363636365, 0.18181818181818182]
split1_rates=[0.7273, 0.0, 0.0, 0.7273, 0.3636, 0.9091, 0.0, 0.4545, 0.0909, 0.9091, 0.2727, 0.7273, 0.0, 0.0, 0, 0, 0.0, 0.1818, 0.2727, 0.0, 0.1818, 0.9091, 0, 0.0909, 0.0, 0.3636, 0.3636, 0.8182, 0.8182, 0, 0.4545, 0.5455, 0.2727, 0.1818, 0.6364, 0.2727, 0.0909, 0.1818, 0.4545, 0.8182, 0.0909, 0.8182, 0.0, 0.0909, 0.1818]

splits_small=[split1_small,split2_small,split3_small,split4_small,split5_small]

#patch_net
# split1=[0, 0, 1, 0, 0, 0, 168, 0, 74, 0, 1, 0, 41, 18, 500, 500, 13, 2, 2, 10, 1, 0, 500, 0, 1, 101, 0, 0, 0, 500, 0, 90, 0, 15, 7, 0, 16, 7, 0, 0, 0, 10, 9, 1, 5]
# split2=[7, 1, 5, 3, 3, 0, 0, 6, 0, 5, 500, 12, 0, 46, 1, 24, 5, 5, 8, 5, 1, 478, 36, 207, 1, 28, 500, 3, 20, 6, 9, 0, 9, 33, 18, 500, 500, 1, 408, 5, 0, 0, 3, 20, 39]
# split3=[4, 500, 6, 0, 57, 0, 0, 1, 6, 2, 0, 37, 1, 13, 5, 0, 3, 2, 0, 0, 2, 3, 500, 0, 0, 9, 26, 500, 0, 13, 0, 10, 55, 1, 578, 1, 500, 633, 0, 1, 4, 1, 0, 500, 34]
# split4=[10, 0, 0, 3, 500, 37, 39, 0, 4, 8, 8, 0, 268, 0, 5, 500, 12, 7, 3, 43, 28, 2, 0, 4, 96, 0, 0, 4, 500, 35, 0, 1, 200, 151, 25, 0, 42, 0, 2, 500, 2, 0, 1, 500, 3]
# split5=[0, 4, 25, 0, 4, 500, 0, 500, 4, 500, 0, 500, 0, 500, 0, 1, 0, 1, 0, 684, 500, 5, 163, 0, 0, 9, 0, 2, 16, 31, 0, 5, 0, 500, 1, 0, 88, 0, 500, 6, 0, 500, 500, 1, 0]

#sacled
split1=[0, 10, 17, 0, 1, 0, 31, 0, 8, 0, 2, 0, 15, 31, 990, 990, 30, 0, 0, 41, 3, 0, 990, 4, 30, 4, 0, 0, 0, 990, 0, 0, 2, 8, 0, 1, 9, 1, 1, 0, 6, 0, 15, 2, 3]
split2=[8, 0, 1, 1, 0, 0, 0, 4, 0, 30, 990, 1, 0, 21, 0, 0, 14, 7, 2, 0, 0, 700, 13, 76, 1, 0, 990, 0, 39, 1, 0, 0, 0, 13, 12, 990, 990, 0, 623, 0, 0, 0, 0, 31, 23]
split3=[0, 990, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 1, 0, 0, 0, 0, 990, 0, 0, 0, 1, 990, 0, 0, 0, 0, 3, 0, 38, 0, 990, 241, 0, 0, 0, 0, 0, 990, 0]
split4=[1, 0, 0, 1, 990, 11, 31, 0, 0, 0, 0, 0, 0, 0, 0, 990, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 990, 0, 0, 0, 2, 13, 0, 0, 0, 0, 2, 990, 1, 0, 0, 990, 0]
split5=[3, 0, 19, 0, 5, 990, 0, 990, 16, 990, 0, 990, 0, 990, 1, 0, 6, 1, 1, 764, 990, 4, 136, 0, 1, 7, 3, 0, 6, 47, 0, 1, 9, 990, 10, 8, 15, 2, 990, 37, 4, 990, 990, 4, 6]


#re sampled
split_5_over=[0, 4480, 5, 0, 45, 23, 0, 31, 497, 2, 0, 109, 0, 0, 0, 0, 17, 0, 0, 2, 18, 0, 0, 9, 11, 0, 77, 35, 17, 1, 8, 0, 75, 11, 139, 0, 19, 0, 1, 4, 3, 5, 185, 11, 4480]
# [0, 4480, 6, 1, 2, 15, 3, 4, 339, 0, 0, 69, 1, 0, 4, 0, 8, 0, 0, 1, 2, 27, 8, 3, 9, 10, 179, 24, 1, 11, 2, 0, 40, 15, 540, 0, 8, 0, 0, 0, 4, 2, 3, 4, 4480]

#bound
# split1=[5, 23, 3, 7, 0, 3, 18, 0, 22, 6, 3, 4, 2, 2, 1, 39, 1, 500, 0, 33, 0, 2, 0, 0, 0, 0, 0, 1, 6, 0, 18, 0, 5, 0, 0, 23, 0, 0, 60, 1, 0, 0, 0, 30, 0]
# split2=[13, 35, 0, 0, 26, 6, 0, 0, 0, 0, 21, 7, 0, 151, 18, 0, 44, 0, 154, 1, 0, 7, 37, 11, 96, 84, 0, 0, 0, 5, 0, 727, 8, 0, 1, 0, 15, 0, 0, 9, 0, 0, 33, 0, 8]
# split3=[34, 1, 0, 3, 0, 8, 0, 0, 0, 115, 53, 25, 0, 0, 0, 3, 0, 10, 1, 0, 0, 9, 0, 500, 2, 0, 9, 500, 5, 11, 1, 0, 45, 0, 0, 97, 29, 14, 1, 3, 0, 0, 0, 3, 1]
# split4=[1, 27, 0, 0, 11, 0, 0, 17, 4, 3, 1, 0, 6, 10, 1, 0, 0, 1, 0, 114, 0, 2, 0, 0, 0, 2, 500, 0, 123, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 1, 3, 1]
# split5=[3, 0, 0, 2, 23, 2, 12, 2, 12, 4, 0, 51, 0, 6, 500, 0, 3, 0, 0, 0, 1, 3, 15, 7, 18, 43, 1, 19, 500, 3, 0, 4, 0, 75, 9, 0, 8, 3, 0, 0, 0, 0, 6, 1, 4]

#SOAP
split1_s=[8, 11800, 15, 45, 0, 90, 1195, 8, 235, 0, 336, 21, 262, 5, 126, 13792, 230, 978, 10347, 1, 0, 51, 29536, 189, 1, 610, 32, 3, 33, 3180, 1, 143, 35, 2, 3365, 4, 500, 45, 168, 5, 70, 4, 233, 87, 871]
split2_s= [1147, 2598, 40, 41, 584, 32, 177, 0, 90, 242, 161, 2, 0, 115, 1158, 471, 137, 230, 10347, 1248, 0, 13450, 1, 6697, 73, 170, 790, 1620, 65, 4, 688, 28, 15, 143, 158, 4265, 8525, 3365, 2215, 4, 119, 37, 2, 1466, 281]
split3_s=[263, 4720, 0, 688, 1195, 33, 423, 8, 2, 2145, 17, 643, 336, 5, 4, 6, 186, 26, 18, 446, 64, 1, 0, 5, 332, 146, 0, 500, 991, 102, 3, 74, 1, 14076, 47, 733, 1327, 2215, 0, 5, 991, 1127, 500, 500, 281]
split4_s=[2598, 40, 41, 1, 833, 1, 7, 32, 0, 0, 235, 110, 603, 336, 260, 126, 6, 137, 186, 26, 230, 978, 1248, 1, 11, 49, 0, 2, 0, 24, 102, 688, 0, 12, 3835, 2182, 2499, 33, 143, 500, 30, 64, 1, 500, 402]
split5_s=[584, 11800, 1, 32, 10622, 523, 0, 14856, 4861, 2788, 2145, 6464, 92, 40285, 0, 6, 3, 1, 0, 13450, 500, 49, 20, 1750, 55, 74, 14076, 1, 4, 437, 61, 208, 500, 1327, 1, 168, 5369, 5, 38903, 21, 37, 404, 500, 105, 8]

split_1_abb=[2, 15, 7, 1, 1, 2, 31, 1, 26, 2, 25, 1, 90, 6, 500, 500, 5, 3, 10, 35, 1, 1, 500, 4, 2, 18, 6, 1, 1, 500, 2, 7, 2, 6, 1, 1, 8, 2, 4, 1, 9, 1, 4, 41, 21]
split_2_abb=[15, 9, 10, 37, 7, 2, 5, 6, 2, 29, 500, 6, 6, 10, 14, 2, 5, 15, 36, 1, 1, 451, 27, 602, 6, 3, 500, 21, 28, 3, 1, 2, 8, 3, 83, 500, 500, 1, 880, 5, 8, 4, 3, 2, 0]
split_3_abb=[12, 500, 2, 7, 88, 1, 1, 1, 15, 2, 1, 51, 19, 11, 3, 1, 1, 4, 1, 1, 1, 4, 500, 2, 1, 16, 115, 500, 2, 8, 1, 1, 211, 1, 596, 89, 500, 696, 7, 3, 6, 12, 5, 500, 28]

split1_s_rates=[0.18181818181818182, 0.0, 0.0, 0.0, 0.5454545454545454, 0.0, 0.0, 0.09090909090909091, 0.0, 0.9090909090909091, 0.0, 0.0, 0.0, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2727272727272727, 0.18181818181818182, 0.18181818181818182, 0.0, 0.0, 0.2727272727272727, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.2727272727272727, 0.0, 0.0, 0.09090909090909091, 0.0, 0.36363636363636365, 0.0, 0.0, 0.0, 0.18181818181818182, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0]

split_1=[7, 55, 8, 146, 500, 270, 32, 0, 148, 377, 500, 5, 4, 76, 178, 1, 13, 500, 500, 500, 188, 5, 3, 25, 500, 8, 12, 6, 3, 500, 9, 10, 2, 0, 500, 32, 500, 38, 46, 1065, 500, 27, 9, 500]
split_2=[9, 985, 0, 0, 85, 4, 63, 500, 97, 7, 4, 8, 45, 1, 11, 500, 14, 2, 4, 25, 0, 11, 2, 500, 500, 1, 8, 4, 1, 104, 8, 0, 22, 40, 0, 3, 1, 110, 17, 500, 7, 3, 1, 19]
split_3=[2, 2, 5, 12, 53, 4, 455, 1, 16, 66, 1, 8, 3, 148, 500, 500, 534, 2, 500, 20, 8, 500, 1, 106, 175, 1, 1, 40, 357, 51, 11, 0, 116, 9, 50, 19, 28, 1, 10, 8, 209, 8, 4, 500]
split_4=[1, 14, 2, 1, 25, 33, 62, 327, 28, 41, 0, 663, 30, 108, 278, 1, 23, 12, 20, 0, 55, 22, 1, 0, 13, 4, 170, 45, 521, 8, 32, 23, 29, 27, 13, 15, 92, 2, 500, 22, 81, 500, 500, 119]
split_5=[0, 500, 2, 6, 11, 500, 9, 112, 561, 5, 12, 454, 0, 3, 6, 1, 16, 3, 1, 2, 4, 10, 0, 1, 2, 59, 148, 8, 500, 3, 9, 1, 12, 1, 1379, 2, 1, 29, 1, 67, 6, 159, 403, 208, 500]

splits=[split_1,split_2,split_3,split_4,split_5]
split_abblations=[split_1_abb,split_2_abb,split_3_abb]
splits_soap=[split1_s,split2_s,split3_s,split4_s,split5_s]


#mid score
#small_patch_net
mid_split1_small=[31, 1020, 1050, 18, 893, 2, 178, 20, 450, 3, 287, 2, 140, 1020, 500, 500, 887, 163, 3, 142, 212, 0, 500, 1050, 206, 193, 1020, 5, 43, 500, 26, 44, 61, 0, 4, 15, 1020, 54, 2, 78, 1020, 125, 104, 1020, 12]
mid_split2_small=[23, 49, 5, 11, 40, 5, 4, 24, 4, 1020, 500, 42, 0, 126, 53, 156, 1020, 702, 48, 12, 8, 990, 272, 990, 6, 1020, 500, 1020, 145, 0, 15, 2, 28, 170, 9, 500, 500, 0, 990, 62, 14, 2, 17, 17, 62]
mid_split3_small=[173, 99, 69, 109, 500, 974, 990, 23, 157, 17, 162, 80, 1020, 25, 120, 500, 124, 1020, 102, 128, 373, 28, 88, 386, 221, 154, 524, 28, 500, 279, 0, 195, 128, 1020, 190, 2, 1020, 24, 465, 500, 445, 139, 12, 500, 286]
mid_split4_small=[587, 500, 40, 7, 813, 11, 66, 142, 68, 11, 77, 419, 7, 1020, 45, 32, 23, 469, 5, 36, 169, 0, 500, 227, 49, 597, 287, 500, 1020, 101, 2, 11, 990, 1050, 990, 1007, 500, 990, 59, 1, 24, 1020, 1050, 500, 298]
mid_split5_small=[15, 1020, 656, 75, 1020, 500, 12, 500, 77, 500, 20, 500, 18, 500, 26, 37, 24, 11, 15, 990, 500, 81, 1020, 19, 114, 51, 1050, 7, 42, 568, 29, 9, 1020, 500, 38, 5, 467, 17, 500, 412, 6, 500, 500, 29, 1]

#patch net
mid_split1=[16, 1020, 1050, 150, 162, 11, 369, 222, 199, 6, 15, 3, 409, 1020, 500, 500, 179, 64, 7, 327, 209, 5, 500, 1050, 26, 309, 1020, 17, 31, 500, 27, 458, 20, 125, 8, 154, 1020, 224, 82, 68, 1020, 18, 21, 1020, 31]
mid_split2=[335, 66, 220, 39, 39, 83, 0, 303, 8, 1020, 500, 30, 3, 763, 263, 665, 1020, 617, 56, 73, 26, 990, 307, 990, 18, 1020, 500, 1020, 622, 6, 113, 51, 83, 378, 233, 500, 500, 4, 990, 21, 59, 0, 51, 184, 133]
mid_split3=[692, 500, 127, 9, 168, 13, 2, 56, 645, 14, 16, 270, 167, 1020, 52, 8, 90, 131, 19, 92, 151, 224, 500, 68, 59, 127, 107, 500, 1020, 218, 3, 51, 990, 1050, 990, 530, 500, 990, 42, 23, 35, 1020, 1050, 500, 158]
mid_split4=[313, 5, 108, 145, 500, 667, 990, 41, 73, 324, 330, 8, 1020, 11, 338, 500, 72, 1020, 55, 525, 829, 81, 81, 150, 403, 35, 58, 4, 500, 240, 0, 169, 773, 1020, 65, 24, 1020, 51, 180, 500, 159, 9, 23, 500, 340]
mid_split5=[12, 1020, 788, 13, 1020, 500, 11, 500, 38, 500, 1, 500, 3, 500, 65, 12, 1, 17, 24, 990, 500, 69, 1020, 68, 250, 110, 1050, 34, 104, 367, 2, 45, 1020, 500, 123, 30, 480, 7, 500, 180, 0, 500, 500, 17, 2]

#scaled
# split1_mid=[9, 1020, 0, 990, 1020, 8, 8, 1020, 11, 990, 21, 126, 5, 15, 7, 41, 17, 500, 990, 16, 8, 1, 1, 500, 1020, 10, 35, 29, 28, 1, 336, 500, 23, 1020, 27, 1020, 4, 13, 6, 30, 500, 500, 5, 500, 14]
# split2_mid=[1, 6, 5, 1020, 5, 58, 105, 6, 6, 3, 1020, 24, 10, 65, 49, 500, 1020, 12, 209, 500, 5, 15, 3, 6, 1020, 357, 0, 29, 31, 64, 65, 500, 24, 9, 13, 23, 13, 500, 212, 25, 14, 9, 324, 1, 3]
# split3_mid=[61, 4, 1020, 1050, 6, 1, 12, 4, 2, 10, 1, 93, 2, 500, 1020, 4, 4, 0, 5, 4, 500, 9, 4, 500, 8, 990, 1020, 500, 8, 6, 0, 34, 5, 17, 11, 9, 2, 500, 990, 32, 19, 1020, 4, 8, 3]
# split4_mid=[1020, 514, 1020, 4, 60, 500, 8, 1020, 34, 3, 38, 22, 15, 500, 34, 16, 1020, 15, 48, 500, 1020, 26, 0, 33, 15, 34, 500, 500, 1, 1020, 9, 31, 9, 500, 228, 18, 7, 35, 1020, 82, 9, 17, 42, 1050, 5]
# split5_mid=[27, 500, 14, 14, 11, 3, 10, 280, 8, 13, 2, 1020, 68, 19, 990, 500, 1020, 500, 23, 0, 57, 500, 1, 500, 1020, 1020, 63, 3, 500, 142, 28, 329, 1020, 6, 5, 500, 16, 4, 8, 15, 1050, 500, 0, 16, 7]

split1_mid=[4, 43, 18, 3, 19, 1, 213, 2, 219, 2, 84, 7, 18, 45, 500, 500, 212, 1, 1, 57, 22, 1, 500, 23, 60, 5, 311, 4, 19, 500, 1, 26, 29, 9, 6, 10, 13, 2, 3, 9, 41, 3, 16, 16, 19]
split2_mid=[305, 52, 18, 100, 31, 39, 7, 16, 1, 500, 500, 38, 18, 316, 632, 21, 500, 210, 35, 18, 19, 500, 81, 500, 80, 500, 500, 500, 369, 3, 65, 29, 150, 319, 14, 500, 500, 1, 500, 1, 2, 1, 8, 324, 250]
split3_mid=[30, 500, 20, 15, 8, 5, 12, 26, 5, 1, 11, 9, 204, 500, 13, 1, 1, 59, 12, 19, 56, 24, 500, 13, 2, 22, 5, 500, 500, 3, 9, 8, 500, 500, 500, 3, 500, 500, 21, 6, 1, 500, 500, 500, 107]
split4_mid=[258, 2, 11, 11, 500, 515, 500, 28, 6, 19, 3, 9, 500, 18, 7, 500, 25, 500, 1, 48, 92, 23, 18, 81, 103, 111, 57, 2, 500, 12, 1, 4, 44, 500, 19, 5, 500, 10, 113, 500, 106, 8, 4, 500, 10]
split5_mid=[50, 500, 186, 22, 500, 500, 18, 500, 101, 500, 2, 500, 9, 500, 206, 2, 9, 2, 50, 500, 500, 93, 500, 24, 90, 143, 500, 15, 330, 369, 16, 38, 500, 500, 116, 9, 270, 14, 500, 515, 5, 500, 500, 41, 13]
#SOAP
mid_split1_s=[306, 500, 500, 150, 0, 11369, 6689, 8, 2371, 0, 34075, 34, 574, 500, 500, 500, 230, 7587, 16870, 48, 0, 8897, 500, 500, 1, 17555, 500, 66, 362, 500, 1, 143, 36, 33, 4835, 4, 500, 194, 8718, 244, 500, 4, 25422, 500, 5005]
mid_split2_s=[27791, 9872, 1586, 1568, 500, 6289, 214, 21, 11369, 500, 6067, 6, 0, 20800, 500, 662, 500, 230, 16870, 2451, 0, 500, 48, 500, 231, 5323, 500, 500, 500, 335, 1874, 279, 2645, 143, 1923, 500, 500, 4835, 500, 4, 119, 255, 2, 8514, 4595]
mid_split3_s=[23055, 500, 0, 11185, 6689, 2394, 500, 8, 225, 51236, 17, 1803, 34075, 500, 19, 1531, 1256, 26, 592, 24333, 3431, 104, 500, 14555, 984, 21231, 28, 500, 500, 102, 66, 190, 500, 500, 500, 500, 34592, 500, 0, 244, 2686, 11300, 500, 500, 4595]
mid_split4_s=[9872, 1586, 1568, 249, 500, 5, 500, 6289, 21, 0, 2371, 321, 500, 34075, 1061, 500, 2228, 500, 1256, 26, 230, 7587, 2451, 48, 11, 766, 0, 192, 500, 231, 228, 1874, 2, 500, 16363, 3687, 500, 362, 143, 500, 500, 278, 5, 500, 17072]
mid_split5_s=[500, 500, 5, 6289, 19572, 500, 0, 500, 40077, 500, 51236, 500, 500, 500, 741, 1531, 10, 130, 0, 500, 500, 766, 500, 500, 4330, 190, 500, 1, 7767, 6787, 498, 1928, 500, 34592, 1, 8718, 22521, 244, 500, 72, 255, 500, 500, 12642, 11]


splits_mid=[split1_mid,split2_mid,split3_mid,split4_mid,split5_mid]
splits_mid_s=[mid_split1_small,mid_split2_small,mid_split3_small,mid_split4_small,mid_split5_small]
splits_soap_mid=[mid_split1_s,mid_split2_s,mid_split3_s,mid_split4_s,mid_split5_s]

ls=[14, 33, 4, 3480, 29, 83, 36, 333, 135, 11, 20, 121, 186, 115, 668, 761, 11, 435, 3480, 200, 3, 212, 3480, 3480, 253, 55, 39, 136, 10, 22, 444, 918, 5, 278, 231, 201, 3, 3480, 72, 262, 1795, 3480, 1824, 174, 5, 2933, 22, 3480, 10, 158, 212, 23, 3480, 86, 16, 101, 170, 731, 97, 3480, 162, 144, 327, 146, 3480, 524, 3, 127, 143, 32, 3367, 195, 105, 52, 100, 105, 82, 696, 258, 125]

def get_ys(nnf):
    nnf.sort()
    xs = np.arange(500)
    ys = [bisect.bisect_right(nnf, val) / len(nnf) for val in xs]
    return ys

def mean_neg(ls):
    num_of_neg=[]
    for split in ls:
        negs=0
        for res in split:
            if res==500:
                negs+=1
        num_of_neg.append(negs)
    num_of_neg.sort()
    return num_of_neg[0]
    # return sum(num_of_neg)/len(num_of_neg)


def get_info(splits,label):
    values=[]
    for s in splits:
        ys=np.array(get_ys(s))
        values.append(ys)
    values=np.array(values)
    mean=np.mean(values,axis=0)
    std=np.std(values,axis=0)
    xs = np.arange(500)
    ax.plot(xs,mean,label=label)
    ax.set_xscale('log')
    # plt.tick_params(axis='y',which='minor')
    ax.fill_between(xs,mean-std,mean+std,alpha=0.3)

    # plt.axvline(x=1, ymin=0, ymax=1, linestyle='--', color='r')
    # plt.axvline(x=1.17, ymin=0, ymax=1, linestyle='--', color='b')
    # plt.axvline(x=2, ymin=0, ymax=1, linestyle='--', color='g')
    # plt.text(0.1, 0.3, "hit rate at 1: %.3f" % mean[2])
    # plt.text(0.28, 0.7, "hit rate at 10: %.3f" % mean[11])
    # plt.text(1.2, 0.9, "hit rate at 15: %.3f" % mean[16])
    # plt.text(2.1, 0.9, "hit rate at 100: %.3f" % mean[100])
    # plt.xlabel("log(N)")
    # plt.ylabel("protein with acceptable solution %")
    # plt.title("success rate")
    # plt.show()
    # plt.savefig("mean_success_rate_acc_patchNet new.png")

def add_meta(file_name,title,splits):
    num_of_neg = mean_neg(splits)
    # plt.axhline(y=1 - (num_of_neg / len(splits[0])), color='r', linestyle='-')
    plt.xlabel("Number of predictons (log scales)")
    plt.ylabel("% complexes with acceptable solution")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0,xmax=500)
    # plt.text(1,0.95,"max acceptable %.2f" %(1 - (num_of_neg / len(splits[0])+0.01)))
    plt.grid()
    # plt.show()
    plt.savefig(file_name)

def plot_compare(split1,split2):
    fig, ax = plt.subplots()
    get_info(splits, "PatchNet acc  unbounded")
    get_info(splits_soap_mid, "SOAP")
    add_meta("success_rate_acc_score_new_b.png", "success rate acc", splits_mid)

if __name__ == '__main__':
    model_path=sys.argv[1]
    test_pdb=sys.argv[2]
    test_dir=sys.argv[3]
    # model_path = "NNscripts/lr-0.003_5_train_transformer_ABDBADAMW_nano_soap_rank_no_bias_transformer_3_3_heads_4_5_60_35_1e4_lr_weights_1.17_0.11_more_kernal_1_3_3_wd_8e-3/mymodel_110"
    # test_pdb = osp.join(wordkdir,"test_pdbs")
    # test_pdb = "ABDB/test_pdbs"
    # test_pdb="dockground/test_pdbs"
    # get_specifisity_batched(model_path,test_pdb,test_dir,config_spec)s
    get_hit_rates_batched(model_path,test_pdb,test_dir,config_evaluate)
    # get_hitraes(model_path,test_pdb)
