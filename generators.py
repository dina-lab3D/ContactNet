import os.path as osp
import tensorflow as tf
from zipfile import BadZipFile
import numpy as np
import time
import pdb
import gc

# seqDir = "dssp"
AAs = "ARNDCQEGHILKMFPSTWYVX"
Symbols = {c: i for i, c in enumerate(AAs)}

def padTo(mat, dim, val=0, is_np=False):
    s = mat.shape
    padding = [[0, m - s[i]] for (i, m) in enumerate(dim)]

    if is_np:
        mat = np.pad(mat, padding, constant_values=val)
    else:
        mat = tf.pad(mat, padding, 'CONSTANT', constant_values=val)
    return mat


def get_sequences(dir, prot):
    '''
    :param structures: a list of structures (can be made with get_structures())
    :return: a list of the sequences of the structures
    '''
    file = osp.join(dir, prot)
    seq = ""
    with open(file) as seqFile:
        lines = seqFile.readlines()
        for line in lines:
            if line.startswith(">"):
                continue
            seq += line.strip()
    return seq


def find_ss(c):
    if c == "G" or c == "H" or c == "I":
        return "1"
    elif c == "E" or c == "B":
        return "2"
    else:
        return "0"


def get_dssp(dssp_file, get_seq=True):
    flag = False
    seq = ""
    dssp = []
    accs = []
    with open(dssp_file) as f:
        lines = f.readlines()
        for line in lines:
            if flag:
                if line[13] == "!":
                    continue
                elif 91 < ord(line[13]):
                    seq += "C"
                else:
                    seq += line[13]
                c = line[16]
                dssp.append(find_ss(c))
                acc = int(line[35:38]) / 260
                accs.append(acc)
            if line.strip().startswith("#"):
                flag = True
    one_hot_seq = tf.keras.utils.to_categorical(dssp, num_classes=3)
    accs = np.asarray(accs).reshape((-1, 1))
    if get_seq:
        return one_hot_seq, accs, seq
    return one_hot_seq, accs


def getNumrical(seq):
    return [Symbols[c] for c in seq]


def getOneHot(seq, seqss, accs):
    lst = getNumrical(seq)
    seq_one_hot = tf.keras.utils.to_categorical(lst, num_classes=21)

    conected = np.concatenate([seq_one_hot, seqss, accs], axis=1)
    return tf.constant(conected, shape=[len(lst), 25])


def getOneHotMatrix(dssp_file, size, fasta=False):
    if fasta:
        seq = get_sequences(dssp_file.split(".")[0].split("_")[0] + ".seq")
        seqss, accs = get_dssp(dssp_file, get_seq=False)
    else:
        seqss, accs, seq = get_dssp(dssp_file)

    if len(seq) > size:
        print("size over th:", len(seq), prot)
        raise OverflowError
    if len(seq) != len(seqss):
        print("WTF")
    dssp = getOneHot(seq, seqss, accs)
    return dssp


bins = [0.23, 0.35, 0.49, 0.8, 1]


def to_one_hot(x):
    x = tf.stack(x)
    discretisize = tf.keras.layers.experimental.preprocessing.Discretization(bins)(x)
    return tf.keras.utils.to_categorical(discretisize, num_classes=len(bins))


def get_new_batch(data_dir, line_number, batch_size=2500, mini_file_size=10000, single_batched=False):
    # try:
    batch_dir = line_number // mini_file_size
    batch_file = (line_number % mini_file_size) // batch_size
    # print("getting batch: batch folder "+str(batch_dir)+" batch_file "+str(batch_file))
    if single_batched:
        file_path_geo = osp.join(data_dir, "batch_geo_" + str(batch_file + 1) + ".npz")
        file_path_cord = osp.join(data_dir, "batch_cords_" + str(batch_file + 1) + ".npz")
    else:
        file_path_geo = osp.join(data_dir, "batch" + str(batch_dir), "batch_geo_" + str(batch_file + 1) + ".npz")
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


def single_file_genarator_pre_batched(prot_ag, prot_ab, size_r=1000, size_l=1000,
                                      patch_size=20, data_dir="", files_in_batch=2500,
                                      single_batched=False, trans_num=0):
    cur_batch_number = 0
    cur_batch_folder = 0
    geo_batch, cords_batch = get_new_batch(data_dir, 0, single_batched=single_batched)
    for line_number in range(trans_num):
        try:
            new_batch_folder, new_batch_number = get_cur_batch_index(line_number, batch_size=files_in_batch)
            if new_batch_number != cur_batch_number or cur_batch_folder != new_batch_folder:
                geo_batch, cords_batch = get_new_batch(data_dir, line_number, single_batched=single_batched,
                                                       batch_size=files_in_batch)
            cur_batch_folder, cur_batch_number = new_batch_folder, new_batch_number
            geo_patches, patches = draw_from_batch(line_number, geo_batch, cords_batch, batch_size=files_in_batch)

            seq1, self_1 = padTo(getOneHotMatrix(prot_ag.split(".pdb")[0] + ".dssp", size_r),
                                 (size_r, 25)), padTo(np.load(prot_ag.split(".pdb")[0] + "_self_distogram.npy"),
                                                      (size_r, size_r))
            seq2, self_2 = padTo(getOneHotMatrix(prot_ab.split(".pdb")[0] + ".dssp", size_l),
                                 (size_l, 25)), padTo(np.load(prot_ab.split(".pdb")[0] + "_self_distogram.npy"),
                                                      (size_l, size_l))
        except OverflowError:
            print("overflow  at ", name)
            continue
        except FileNotFoundError:
            print("file not found: ", file_path)
            print("at " + data_dir,
                  " at batch folder - " + str(cur_batch_folder) + " at batch: " + str(cur_batch_number))
            continue
        tf.debugging.assert_shapes(
            [(seq1, (size_r, 25)), (seq2, (size_l, 25)), (self_1, (size_r, size_r)),
             (self_2, (size_l, size_l)), (geo_patches, (8, patch_size, patch_size)),
             (patches, (8, 2))])
        yield seq1, self_1, seq2, self_2, geo_patches, patches


def file_genarator_pre_batched(prot_ag, prot_ab, batch_size=10, size_r=1000, size_l=1000,
                               patch_size=20, data_dir="", file_in_batch=2500, single_batched=False, trans_num=0):
    line_gen = single_file_genarator_pre_batched(prot_ag, prot_ab, size_r=size_r, trans_num=trans_num,
                                                 size_l=size_l, patch_size=patch_size, data_dir=data_dir,
                                                 files_in_batch=file_in_batch,
                                                 single_batched=single_batched)
    cur_batch_folder = 0
    line_number = 0
    i = 0
    for line_number in range(trans_num):
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

        except EOFError:
            print("trans file finished at file_genarator_pre_batched")
            raise EOFError

        batch = {"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                 "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                 "input_6": tf.stack(patches_batch)}
        i += 1
        yield batch
