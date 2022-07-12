import numpy as np
import pandas as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import Bio.PDB as pdb
from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
import os.path as osp
import sys
import os
import seaborn as sn
import matplotlib.pyplot as plt
import time
import time
import pickle
from Bio.PDB.Polypeptide import PPBuilder
import csv

# from patch_extractor import find_centers_by_hyrstic ,patch_to_tensors,get_cords

structs = ""
seqDir = "dssp"
structDir = "/cs/labs/dina/matanhalfon/structs/"
# structDir = "/cs/labs/dina/matanhalfon/CAPRI/tomer_models"


# self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/dock_test/self_distograms"
self_distogram_dir = "/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distograms"
# self_distogram_dir="/cs/labs/dina/matanhalfon/self_distograms"


SIZE = 1000

AAs = "ARNDCQEGHILKMFPSTWYVX"
Symbols = {c: i for i, c in enumerate(AAs)}


class preprosser:

    @staticmethod
    def getNumrical(seq):
        # print(seq)
        return [Symbols[c] for c in seq]

    @staticmethod
    def getOneHot(seq, seqss, accs):
        lst = preprosser.getNumrical(seq)
        seq_one_hot = tf.keras.utils.to_categorical(lst, num_classes=21)

        conected = np.concatenate([seq_one_hot, seqss, accs], axis=1)
        return tf.constant(conected, shape=[len(lst), 25])

    @staticmethod
    def get_structures():
        '''
        :return: a list of the structures of all the loops.
        '''
        structures = []
        for pdb_code in os.listdir(structs):
            try:
                pdb_filename = osp.join(structs, pdb_code, pdb_code + '.pdb')
                structure = pdb.PDBParser().get_structure(pdb_code, pdb_filename)
                structures.append(structure)
            except Exception:
                print(pdb_code)

        return structures

    @staticmethod
    def find_ss(c):
        if c == "G" or c == "H" or c == "I":
            return "1"
        elif c == "E" or c == "B":
            return "2"
        else:
            return "0"

    @staticmethod
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
                    dssp.append(preprosser.find_ss(c))
                    acc = int(line[35:38]) / 260
                    accs.append(acc)
                if line.strip().startswith("#"):
                    flag = True
        one_hot_seq = tf.keras.utils.to_categorical(dssp, num_classes=3)
        accs = np.asarray(accs).reshape((-1, 1))
        if get_seq:
            return one_hot_seq, accs, seq
        return one_hot_seq, accs

    @staticmethod
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

    @staticmethod
    def padTo(mat, dim, val=0, is_np=False):
        s = mat.shape
        padding = [[0, m - s[i]] for (i, m) in enumerate(dim)]
        # print(padding)
        # print(mat.shape)
        if is_np:
            mat = np.pad(mat, padding, constant_values=val)
        else:
            mat = tf.pad(mat, padding, 'CONSTANT', constant_values=val)
        return mat

    @staticmethod
    def getOneHotMatrix(dssp_file, size, fasta=False):
        if fasta:
            seq = preprosser.get_sequences(dssp_file.split(".")[0].split("_")[0] + ".seq")
            seqss, accs = preprosser.get_dssp(dssp_file, get_seq=False)
        else:
            seqss, accs, seq = preprosser.get_dssp(dssp_file)

        if len(seq) > size:
            print("size over th:", len(seq), prot)
            raise OverflowError
        if len(seq) != len(seqss):
            print("WTF")
        dssp = preprosser.getOneHot(seq, seqss, accs)
        # print("len of "+prot,len(dssp))
        return dssp
        # return preprosser.padTo(dssp, (size,25))

    @staticmethod
    def get_matrixs(path, size_r, size_l):
        # for file in os.listdir(dir):
        mat = np.load(path)
        shape = mat.shape
        mat = np.swapaxes(np.swapaxes(mat.flatten().reshape(np.roll(shape, 1)), 0, 2), 0, 1)
        if shape[0] > size_r or shape[1] > size_l:
            print("XXXX", path)
            print("size over th:", shape)
            raise OverflowError
        # mat=mat[:,:,0]
        # mat=np.expand_dims(mat,axis=-1)
        # print(np.max(mat))
        # mat=mat/100
        # mat[:,:,0]=mat[:,:,0]/100
        # mat[:,:,1:]=np.sin(mat[:,:,1:]/2)
        # dims=file.split('.')[1].split('X')
        # mat.reshape((int(dims[0]),int(dims[1]),int(dims[2])))
        return preprosser.padTo(mat, (size_r, size_l, 3))

    @staticmethod
    def get_distograms(path, size_r, size_l, prot1, prot2):
        # for file in os.listdir(dir):
        mat = np.load(path,allow_pickle=True)
        shape = mat.shape
        # mat = np.swapaxes(np.swapaxes(mat.flatten().reshape(np.roll(shape, 1)), 0, 2), 0, 1)
        if shape[0] > size_r or shape[1] > size_l:
            print("XXXX", path)
            print("size_r ",size_r," size_l ",size_l)
            print("size over th:", shape)
            raise OverflowError
        pair_distograms = mat
        # self_distogram_1=preprosser.padTo(np.load(osp.join(self_distogram_dir,prot1+"_distogram")),(size_r,size_r))
        # self_distogram_2=preprosser.padTo(np.load(osp.join(self_distogram_dir,prot2+"_distogram")),(size_l,size_l))
        # self_distogram_1 = np.load(osp.join(self_distogram_dir, prot1 + "_distogram"))
        # self_distogram_2 = np.load(osp.join(self_distogram_dir, prot2 + "_distogram"))
        # return pair_distograms,self_distogram_1,self_distogram_2
        return pair_distograms
        # mat=np.expand_dims(mat,axis=-1)
        # print(np.max(mat))
        # mat=mat/100
        # mat[:,:,0]=mat[:,:,0]/100
        # mat[:,:,1:]=np.sin(mat[:,:,1:]/2)
        # dims=file.split('.')[1].split('X')
        # mat.reshape((int(dims[0]),int(dims[1]),int(dims[2])))
        # return preprosser.padTo(pair_distograms, (size_r, size_l,3)),preprosser.padTo(self_distogram_1,(size_r,size_r)),preprosser.padTo(self_distogram_2,(size_l,size_l))

    @staticmethod
    def get_prot_len(dir="dssp", is_fasta=False,test_list=None,line_len=7):
        protains = []
        max = 0
        for seq_file in os.listdir(dir):
            if not test_list or seq_file[:line_len] in test_list:
                # print(seq_file)
                # print(seq_file[:line_len])
                if is_fasta:
                    seqss, accs = preprosser.get_dssp(dir, seq_file, get_seq=False)
                    seq = preprosser.get_sequences(dir, seq_file + ".seq")
                else:
                    seqss, accs, seq = preprosser.get_dssp(dir, seq_file)

                if len(seq) != len(seqss):
                    print("fucked at: ", seq_file)
                    raise KeyError
                if len(seq) > max:
                    max = len(seq)
                protains.append((seq_file, len(seq)))
        print(max)
        return protains


    @staticmethod
    def show_mat(mat, vmax=0.8, save=False, title=""):
        sn.heatmap(mat, cmap="Reds", vmax=vmax)
        plt.title(title)
        if save:
            plt.savefig(title)
        plt.show()

    @staticmethod
    def get_to_long(name=None, line_end = 4, suffix_r = "_u1", suffix_l = "_u2",len_r=250,len_l=700):
        protains = preprosser.get_prot_len(name,line_len=line_end)
        black_list = []
        for p in protains:
            if p[1] < 35 :
                print(p[0], " ", p[1])
                black_list.append(p[0][:line_end])

            elif p[1] > len_r and p[0].endswith(suffix_r+".dssp"):
                print(p[0], " ", p[1])
                black_list.append(p[0][:line_end])
            elif p[1] > len_l and p[0].endswith(suffix_l+".dssp"):
                print(p[0], " ", p[1])
                black_list.append(p[0][:line_end])
            # print(suffix_r + ".dssp")
            # print(p)
            # exit(0)
        print("black listed: ", len(black_list))
        black_list=set(black_list)
        # print(black_list)
        return black_list

    @staticmethod
    def write_small_data(origin, short, workdir):
        black_list = preprosser.get_to_long()
        print(black_list)
        with open(osp.join(workdir, origin)) as ori:
            with open(osp.join(workdir, short), 'w+') as s:
                short_lines = []
                for i, line in enumerate(ori):
                    split = line.split(" ")
                    name = split[1][:4]
                    if name not in black_list:
                        short_lines.append(line)
                    else:
                        print(line)
                s.writelines(short_lines)

    @staticmethod
    def get_shortData():
        with open("short_pos") as pos:
            with open("short_neg") as neg:
                pos_lines = pos.readlines()
                neg_lines = neg.readlines()
                label = np.concatenate([np.ones(500), np.zeros(500)])
                data = np.concatenate([pos_lines, neg_lines])
                data = np.concatenate([data.reshape((1, -1)), label.reshape((1, -1))], axis=0).T
                np.random.shuffle(data)
                return data[:, 0].reshape(1, -1), data[:, 1].reshape(-1, 1)

    @staticmethod
    def get_name_from_line(line):
        line = line.strip()
        splitted = line.split("\t")
        index = splitted[0]
        prot1 = splitted[1].split(" ")[0]
        prot2 = splitted[1].split(" ")[1]
        name = osp.join(prot1 + "X" + prot2 + "transform_number_" + index)  # switch prot1-prot2

        return name, prot1.split(".")[0], prot2.split(".")[0]

    @staticmethod
    def parse_line(line,  get_label=False, get_rmsd=False,get_dockq=False):
        line = line.strip()
        splitted = line.split("\t")
        # index = splitted[0]
        # prot1 = (splitted[1].split(".")[0])[:11]##change to 9
        # prot2 = (splitted[2].split(".")[0])[:11]
        # prot1 = splitted[1].split(".")[0]
        # prot2 = splitted[2].split(".")[0]
        # name = osp.join(prot2[0:4], splitted[1] + "X" + splitted[2] + "transform_number_" + index)#switch prot1-prot2
        name, prot1, prot2 = preprosser.get_name_from_line(line)
        # dir = osp.join(structDir, "struct",structdir)
        if get_dockq:
            label = 1 if splitted[-1] == "True" else 0
            rec_rmsd = float(splitted[-4])
            lig_rmsd = float(splitted[-3])
            dockq=float(splitted[-2])
            return name, prot1, prot2,  label, rec_rmsd, lig_rmsd,dockq

        elif get_rmsd:
            label = 1 if splitted[-1] == "True" else 0
            rec_rmsd = float(splitted[-3])
            lig_rmsd = float(splitted[-2])
            return name, prot1, prot2,  label, rec_rmsd, lig_rmsd
        elif get_label:
            label = int(splitted[-1])
            return name, prot1, prot2,  label, None, None
        return name, prot1, prot2,  None, None, None

    @staticmethod
    def get_data(file_path, size_r, size_l, prot1, prot2, workdir):
        geoMat = preprosser.get_matrixs(file_path, size_r, size_l)
        seq1 = preprosser.getOneHotMatrix(osp.join(workdir, seqDir), prot1, size_r)
        seq2 = preprosser.getOneHotMatrix(osp.join(workdir, seqDir), prot2, size_l)
        return geoMat, seq1, seq2

    @staticmethod
    def get_data_transformer(file_path, size_r, size_l, prot1, prot2):
        # geoMat,self_1,self_2= preprosser.get_distograms(file_path, size_r, size_l,prot1,prot2)
        geoMat = preprosser.get_distograms(file_path, size_r, size_l, prot1, prot2)
        # seq1 = preprosser.getOneHotMatrix(osp.join(workdir, seqDir), prot1, size_r)
        # seq2 = preprosser.getOneHotMatrix(osp.join(workdir, seqDir), prot2, size_l)
        # return geoMat,self_1,self_2,seq1, seq2
        return geoMat

    #
    @staticmethod
    def line_gen(workdir, data_file, suffix, finate=False, header=0):
        # print((osp.join(workdir, data_file) + suffix))
        while True:
            with open(osp.join(workdir, data_file) + suffix) as f:
                reader = csv.reader(f)
                for i, line in enumerate(reader):
                    if i < header:
                        continue
                    yield i,line[0]
                print("XXXXXXX balanced gen finished " + data_file)
            if finate:
                raise EOFError
                # with open(osp.join(workdir, data_file) + suffix) as reader:
            #     line=reader.readline()
            #     while line:
            #         yield line
            #         line=reader.readline()
            #     print("XXXXXXX balanced gen finished "+suffix)
            # if finate:
            #     raise EOFError


def extract_patchs(im, patchs, patch_size=30):
    out = []
    for p in patchs:
        extracted = im[p[0]:p[0] + patch_size, p[1]:p[1] + patch_size]
        out.append(preprosser.padTo(extracted, (patch_size, patch_size)))
    return tf.stack(out)

# if __name__ == '__main__':
#     workdir=sys.argv[1]
# print(preprosser.get_to_long())
#     preprosser.write_small_data("short_neg","short_neg1","cur_data")
#     for batch in complex_genarator("InterfaceFeatures/protainTrans",6):
#         print("gen works")
#  dir="InterfaceFeatures/seqs"
#  prot= "2nn6.pdb"
#  seq=preprosser.getOneHotMatrix(dir,prot,2500)
#  print("padded",seq.shape)
#  # for geo in preprosser.get_matrixs("InterfaceFeatures/structs",(2500,300,6)):
#  #     print(geo)
#  # parser=pdb.PDBParser()
#  # struct= parser.get_structure("2nn6","InterfaceFeatures/example/2nn6.pdb")
#  # dict=preprosser.getOneHot(preprosser.get_sequences(struct))
#  # print(preprosser.padSeq(dict,350,0))
#  # mol1 =pdb.get()
# # preprosser.get_matrixs("InterfaceFeatures/structs")
