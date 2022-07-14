from preprocessing import preprosser
import os.path as osp
import tensorflow as tf
from zipfile import BadZipFile
import numpy as np
import time
import pdb
import gc
from config import config4 as config

seqDir = "dssp"

bins=[0.23,0.35,0.49,0.8,1]
def to_one_hot(x):
    x=tf.stack(x)
    discretisize=tf.keras.layers.experimental.preprocessing.Discretization(bins)(x)
    return   tf.keras.utils.to_categorical(discretisize, num_classes=len(bins))


def build_prot_dict(prot1, prot2, size_r, size_l):
    dict = {}
    self1 = np.load("self-distograms/" + prot1.split(".pdb")[0] + ".npy")
    self2 = np.load("self-distograms/" + prot2.split(".pdb")[0] + ".npy")
    self1 = preprosser.padTo(self1, (size_r, size_r))
    self2 = preprosser.padTo(self2, (size_l, size_l))
    dssp_file1 = "dssp/" + prot1.split(".pdb")[0] + ".dssp"
    dssp_file2 = "dssp/" + prot2.split(".pdb")[0] + ".dssp"
    dssp1 = preprosser.getOneHotMatrix(prot1, size_r)
    dssp2 = preprosser.getOneHotMatrix(prot2, size_l)
    dssp_1 = preprosser.padTo(dssp1,(size_r, 25))
    dssp_2 = preprosser.padTo(dssp2,(size_l, 25))
    dict[prot1[:6]]=(dssp1, self1)
    dict["Ag"] = (dssp2, self2)
    print(prot1)
    print(prot2)
    print(dict)
    return dict


def get_new_batch(data_dir, line_number, batch_size=2500, mini_file_size=10000,single_batched =False):
    # try:
    batch_dir = line_number // mini_file_size
    batch_file = (line_number % mini_file_size) // batch_size
    # print("getting batch: batch folder "+str(batch_dir)+" batch_file "+str(batch_file))
    if single_batched:
        file_path_geo = osp.join(data_dir,  "batch_geo_" + str(batch_file + 1) + ".npz")
        file_path_cord = osp.join(data_dir,"batch_cords_" + str(batch_file + 1) + ".npz")
    else:
        file_path_geo=osp.join(data_dir, "batch" + str(batch_dir), "batch_geo_" + str(batch_file + 1) + ".npz")
        file_path_cord = osp.join(data_dir, "batch" + str(batch_dir), "batch_cords_" + str(batch_file + 1) + ".npz")
    geo_batch = dict(np.load(file_path_geo,
                             allow_pickle=True))["arr_0"]
    cords_batch = dict(
        np.load(file_path_cord,
                allow_pickle=True))["arr_0"]
    if geo_batch is None:
        raise FileNotFoundError
    return geo_batch, cords_batch

def get_cur_batch_index(line_number, batch_size=2500, mini_file_size=10000):
    batch_folder = line_number // mini_file_size
    batch_file = (line_number % mini_file_size) // batch_size
    return batch_folder, batch_file


def draw_from_batch(line_number, geo_batch, cords_batch, batch_size=2500, mini_file_size=10000):
    batch_index = line_number % batch_size
    return geo_batch[batch_index], cords_batch[batch_index]


def single_file_genarator_pre_batched(prot_ag,prot_ab, size_r=1000, size_l=1000,
                                      patch_size=20, data_dir="",files_in_batch=2500,
                                      single_batched=False):
    cur_batch_number = 0
    cur_batch_folder = 0
    geo_batch, cords_batch = get_new_batch(data_dir, 0,single_batched=single_batched)
    while True:
        if not line:
            raise EOFError
        try:
            new_batch_folder, new_batch_number = get_cur_batch_index(line_number,batch_size=files_in_batch)
            if new_batch_number != cur_batch_number or cur_batch_folder != new_batch_folder:
                geo_batch, cords_batch = get_new_batch(data_dir, line_number,single_batched=single_batched,batch_size=files_in_batch)
            cur_batch_folder, cur_batch_number = new_batch_folder, new_batch_number
            geo_patches, patches = draw_from_batch(line_number, geo_batch, cords_batch,batch_size=files_in_batch)

            seq1, self_1 = preprosser.padTo(preprosser.getOneHotMatrix("dssp/" + prot_ag.split(".pdb")[0] + ".dssp", size_r),
                                            (size_r, 25)),preprosser.padTo(np.load("self-distograms/" + prot_ag.split(".pdb")[0] + ".npy"),(size_r,size_r))
            seq2, self_2 = preprosser.padTo(preprosser.getOneHotMatrix("dssp/" + prot_ab.split(".pdb")[0] + ".dssp", size_l),
                                            (size_, 25)),preprosser.padTo(np.load("self-distograms/" + prot_ab.split(".pdb")[0] + ".npy"),(size_l,size_l))
        except OverflowError:
            print("overflow  at ",name)
            continue
        except FileNotFoundError:
            print("file not found: ", file_path)
            print("at "+data_dir," at batch folder - "+str(cur_batch_folder)+" at batch: "+str(cur_batch_number))
            continue
        tf.debugging.assert_shapes(
            [(seq1, (size_r, 25)), (seq2, (size_l, 25)), (self_1, (size_r, size_r)),
             (self_2, (size_l, size_l)), (geo_patches, (8, patch_size, patch_size)),
             (patches, (8, 2))])
        yield seq1, self_1, seq2, self_2, geo_patches, patches


def file_genarator_pre_batched(prot_ag,prot_ab, batch_size=10, size_r=1000, size_l=1000,
                      patch_size=20, data_dir="",file_in_batch=2500,single_batched=False):
    line_gen = single_file_genarator_pre_batched(prot_ag,prot_ab, size_r=size_r,
                                                size_l=size_l,patch_size=patch_size,data_dir=data_dir,files_in_batch=file_in_batch,
                                                single_batched=single_batched)
    cur_batch_folder = 0
    line_number = 0
    i=0
    while True:
        seqs1 = []
        distograms1 = []
        seqs2 = []
        distograms2 = []
        geoMats = []
        patches_batch = []
        try:
            while len(seqs1) < batch_size:
                seq1, self_1, seq2, self_2, geo_patches, patches = line_gen.__next__()
                seqs1.append(seq1)
                distograms1.append(self_1)
                seqs2.append(seq2)
                distograms2.append(self_2)
                patches_batch.append(patches)
                geoMats.append(geo_patches)
                line_number += 1

        except EOFError:
            print("trans file finished at file_genarator_pre_batched")
            raise EOFError

        batch = {"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                          "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                          "input_6": tf.stack(patches_batch)}
        i+=1
        yield batch
