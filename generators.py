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
# structDir = "/cs/labs/dina/matanhalfon/structs/"
# data_dir="/cs/labs/dina/matanhalfon/CAPRI/patch_data_ABDB"
# data_dir="/cs/labs/dina/matanhalfon/patch_data"
# data_dir="/cs/labs/dina/matanhalfon/patch_data_dockground"
# data_dir="/cs/labs/dina/matanhalfon/CAPRI/patch_data_ABDB_nano"




# data_dir="/cs/labs/dina/matanhalfon/CAPRI/dock_test/patches"

# self_distogram_dir = "/cs/labs/dina/matanhalfon/self_distograms"

# self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/dockground/self_distogram_bound"

# self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distograms"
# self_distogram_dir = "/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distogram_nano"


# self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/dock_test/self_distogra

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


def balanced_genarator(data_file, prot_dict, data_postfix="", batch_size=10, size_r=1000, size_l=1000, patch_size=20,
                       sample_rate=4, data_dir="", title_len=4, workdir="", get_rmsd=True):
    pos_gen = preprosser.line_gen(workdir, data_file, "_pos_" + data_postfix)
    neg_gen = preprosser.line_gen(workdir, data_file, "_neg_" + data_postfix)
    line_number = 0
    while True:
        seqs1 = []
        distograms1 = []
        seqs2 = []
        distograms2 = []
        geoMats = []
        labels = []
        rmsds_rec = []
        rmsds_lig = []
        patches_batch = []
        names=[]
        while len(seqs1) < batch_size:
            # gc.collect()
            if line_number % sample_rate == 0:
                i,line = pos_gen.__next__()
                label = 1
            else:
                i,line = neg_gen.__next__()
                label = 0
            if not line:
                raise EOFError

            name, prot1, prot2, label, rec_rmsd, lig_rmsd = preprosser.parse_line(line, get_label=True,
                                                                                  get_rmsd=get_rmsd)
            file_path = osp.join(data_dir, prot1[:title_len], name + ".npz")
            try:
                compressed = np.load(file_path, allow_pickle=True)
                # print(file_path)
                compressed_dict = dict(compressed)
                seq1, self_1, seq2, self_2 = prot_dict[prot1]
                geo_patches = compressed_dict["arr_0"]
                patches = compressed_dict["arr_1"]
                print(seq1.shape,self1.shape,seq2.shape,self2.shape,geo_patches.shape)
                if np.any(patches[:, 0] + patch_size > size_l) or np.any(patches[:, 1] + patch_size > size_r):
                    print("overflow at " + file_path)
                    print(patches)
                    print(size_r, " ", size_l)
                    raise IndexError
            except FileNotFoundError:
                print("file not found: ", file_path)
                print("at balanced")
                continue
            except OverflowError:
                print(name)
                continue
            patches_batch.append(patches)
            geoMats.append(geo_patches)
            seqs1.append(seq1)
            distograms1.append(self_1)
            seqs2.append(seq2)
            distograms2.append(self_2)
            labels.append(label)
            rmsds_rec.append(rec_rmsd)
            rmsds_lig.append(lig_rmsd)
            names.append(name)
            line_number += 1
        if get_rmsd:
            batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                      "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                      "input_6": tf.stack(patches_batch),
                      },
                     {"classification": tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (batch_size, 1)),
                      "rec_rmsd": tf.stack(rmsds_rec),
                      "lig_rmsd": tf.stack(rmsds_lig),
                      "names":tf.convert_to_tensor(names)
                      }
                     )
        else:
            batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                      "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                      "input_6": tf.stack(patches_batch)},
                     tf.convert_to_tensor(labels, dtype=tf.float32))
        yield batch


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


def single_file_genarator_pre_batched(data_file, prot_dict, size_r=1000, size_l=1000,
                                      patch_size=20, data_dir="", title_len=4, workdir="",files_in_batch=2500,
                                      get_rmsd=True,get_dockq=True,single_batched=False,finite=False):
    line_gen = preprosser.line_gen(workdir, data_file, "",finate=finite)
    cur_batch_number = 0
    cur_batch_folder = 0
    geo_batch, cords_batch = get_new_batch(data_dir, 0,single_batched=single_batched)
    while True:
        line_number,line = line_gen.__next__()
        if not line:
            raise EOFError
        name, prot1, prot2, label, rec_rmsd, lig_rmsd,dockq = preprosser.parse_line(line, get_label=True,
                                                                              get_rmsd=get_rmsd,get_dockq=get_dockq)
        file_path = osp.join(data_dir, prot1[:title_len], name + ".npz")
        try:
            new_batch_folder, new_batch_number = get_cur_batch_index(line_number,batch_size=files_in_batch)
            if new_batch_number != cur_batch_number or cur_batch_folder != new_batch_folder:
                geo_batch, cords_batch = get_new_batch(data_dir, line_number,single_batched=single_batched,batch_size=files_in_batch)
            cur_batch_folder, cur_batch_number = new_batch_folder, new_batch_number
            geo_patches, patches = draw_from_batch(line_number, geo_batch, cords_batch,batch_size=files_in_batch)
            seq1, self_1 = prot_dict[prot1[:6]]
            seq2, self_2 = prot_dict["Ag"]
            # print(cords_batch)
            if np.any(patches[:, 0] + patch_size > size_l) or np.any(patches[:, 1] + patch_size > size_r):
                print("overflow at " + file_path)
                print(patches)
                print(size_r, " ", size_l)
                raise IndexError
            if np.any(patches[:, 0] + patch_size >= size_r) or np.any(patches[:, 1] + patch_size >= size_l):
                print("overflow at " + file_path)
                print(patches)
                print(size_r, " ", size_l)
                raise IndexError
        except OverflowError:
            print("overflow  at ",name)
            continue
        # except KeyError:
        #     print(prot1 +" not folded")
        #     continue
        except FileNotFoundError:
            print("file not found: ", file_path)
            print("at "+data_dir," at batch folder - "+str(cur_batch_folder)+" at batch: "+str(cur_batch_number))
            continue
        tf.debugging.assert_shapes(
            [(seq1, (size_r, 25)), (seq2, (size_l, 25)), (self_1, (size_r, size_r)),
             (self_2, (size_l, size_l)), (geo_patches, (8, patch_size, patch_size)),
             (patches, (8, 2))])
        yield seq1, self_1, seq2, self_2, geo_patches, patches, label, rec_rmsd, lig_rmsd,dockq, name


def balanced_genarator_pre_batched(data_file, prot_dict, data_postfix="", batch_size=50, size_r=1000, size_l=1000,
                                   patch_size=20, sample_rate=4, data_dir="", title_len=4, workdir="", get_rmsd=True):
    pos_gen = single_file_genarator_pre_batched(data_file + "_pos_" + data_postfix, prot_dict, size_r=size_r,
                                                size_l=size_l,patch_size=patch_size, data_dir=osp.join(data_dir, "train_pos"),
                                                title_len=title_len, workdir=workdir, get_rmsd=get_rmsd)
    neg_gen = single_file_genarator_pre_batched(data_file + "_neg_" + data_postfix, prot_dict, size_r=size_r,
                                                size_l=size_l, patch_size=patch_size,data_dir=osp.join(data_dir, "train_neg"),
                                                title_len=title_len,workdir=workdir, get_rmsd=get_rmsd)
    # cur_batch_number = 0
    # cur_batch_folder = 0
    line_number = 0
    while True:
        seqs1 = []
        distograms1 = []
        seqs2 = []
        distograms2 = []
        geoMats = []
        labels = []
        dockqs = []
        rmsds_lig = []
        patches_batch = []
        # names = []
        try:
            while len(seqs1) < batch_size:
                # gc.collect()
                if line_number % sample_rate == 0:
                    seq1, self_1, seq2, self_2, geo_patches, patches, label, rec_rmsd, lig_rmsd,dockq, name = pos_gen.__next__()
                else:
                    seq1, self_1, seq2, self_2, geo_patches, patches, label, rec_rmsd, lig_rmsd,dockq, name = neg_gen.__next__()
                patches_batch.append(patches)
                geoMats.append(geo_patches)
                seqs1.append(seq1)
                distograms1.append(self_1)
                seqs2.append(seq2)
                distograms2.append(self_2)
                labels.append(label)
                dockqs.append(dockq)
                rmsds_lig.append(lig_rmsd)
                # names.append(name)
                line_number += 1
        except EOFError :
            print("trans file ended at balanced generator ")
            if len(patches_batch) == 0:
                raise EOFError
        if get_rmsd:
            batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                      "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                      "input_6": tf.stack(patches_batch),
                      },
                     {"classification": tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (batch_size, 1)),
                      # "rec_rmsd": tf.stack(rmsds_rec),
                      "dockQ":to_one_hot(dockqs),
                      "lig_rmsd": tf.stack(rmsds_lig),
                      # "names":tf.convert_to_tensor(names)
                      }
                     )
        else:
            batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                      "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                      "input_6": tf.stack(patches_batch)},
                     tf.convert_to_tensor(labels, dtype=tf.float32))
        # layer(tf.stack(dockqs))
        # to_one_hot(dockqs)
        yield batch


def complex_genarator(data_file, prot_dict=None, batch_size=10, size_r=1000, size_l=1000, data_postfix="",
                      patch_size=20, data_dir="", title_len=4, workdir="", finite=False, header=0):
    data_gen = preprosser.line_gen(workdir, data_file, data_postfix, finate=finite, header=header)
    batch_number = 0
    while True:
        seqs1 = []
        distograms1 = []
        seqs2 = []
        distograms2 = []
        geoMats = []
        labels = []
        rmsds_rec = []
        rmsds_lig = []
        patches_batch = []
        names = []
        while len(seqs1) < batch_size:
            i,data_line = data_gen.__next__()

            name, prot1, prot2, label, rec_rmsd, lig_rmsd = preprosser.parse_line(data_line,
                                                                                  get_label=True,
                                                                                  get_rmsd=True)
            file_path = osp.join(data_dir, prot1[:title_len], name + ".npz")
            try:
                compressed = np.load(file_path, allow_pickle=True)
                compressed_dict = dict(compressed)
                geo_patches = compressed_dict["arr_0"]
                patches = compressed_dict["arr_1"]
                seq1, self_1, seq2, self_2 = prot_dict[prot1]
                # print(patches)
                # print("seq1: ",seq1.shape)
                # print("seq2: ", seq2.shape)
                if np.any(patches[:, 0] + patch_size > size_l) or np.any(patches[:, 1] + patch_size > size_r):
                    print("overflow at " + file_path)
                    print(patches)
                    print(size_r, " ", size_l)
                    raise IndexError
                tf.debugging.assert_shapes(
                    [(seq1, (size_r, 25)), (seq2, (size_l, 25)), (self_1, (size_r, size_r)),
                     (self_2, (size_l, size_l)), (geo_patches, (8, patch_size, patch_size)),
                     (patches, (8, 2))])
                patches_batch.append(patches)
                geoMats.append(geo_patches)
                seqs1.append(seq1)
                distograms1.append(self_1)
                seqs2.append(seq2)
                distograms2.append(self_2)
                labels.append(label)
                rmsds_rec.append(rec_rmsd)
                rmsds_lig.append(lig_rmsd)
                names.append(name)

            except EOFError:
                print("finish with : ", data_file)
                batch = (
                    {"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                     "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                     "input_6": tf.stack(patches_batch)},
                    {"classification": tf.reshape(tf.convert_to_tensor(labels), (batch_size, 1)),
                     "rec_rmsd": tf.reshape(tf.convert_to_tensor(rmsds_rec), (batch_size, 1)),
                     "lig_rmsd": tf.reshape(tf.convert_to_tensor(rmsds_lig), (batch_size, 1)),
                     "names":tf.convert_to_tensor(names)
                     })
                return batch
            except FileNotFoundError:
                print("not found: ", file_path)

        batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                  "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                  "input_6": tf.stack(patches_batch)
                  },
                 {"classification": tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (batch_size, 1)),
                  "rec_rmsd": tf.reshape(tf.convert_to_tensor(rmsds_rec), (batch_size, 1)),
                  "lig_rmsd": tf.reshape(tf.convert_to_tensor(rmsds_lig), (batch_size, 1)),
                  "names": tf.reshape(tf.convert_to_tensor(names), (batch_size, 1))
                  })
        # batch=([tf.stack(seqs1),tf.stack(seqs2),tf.stack(geoMats)],tf.convert_to_tensor(labels))
        # line_number+=batch_size
        # print("num batch: ",batch_number)
        yield batch



def file_genarator_pre_batched(data_file, prot_dict=None, batch_size=10, size_r=1000, size_l=1000, data_postfix="",
                      patch_size=20, data_dir="", title_len=4, workdir="", finite=False,evaluate=False,file_in_batch=2500,
                    single_batched=False,get_rmsd=True,get_dockq=True):
    line_gen = single_file_genarator_pre_batched(data_file + data_postfix, prot_dict, size_r=size_r,
                                                size_l=size_l,patch_size=patch_size,data_dir=data_dir,files_in_batch=file_in_batch,
                                                title_len=title_len, workdir=workdir, get_dockq=get_dockq,
                                                single_batched=single_batched,finite=finite)
    cur_batch_folder = 0
    line_number = 0
    i=0
    while True:
        seqs1 = []
        distograms1 = []
        seqs2 = []
        distograms2 = []
        geoMats = []
        labels = []
        rmsds_rec = []
        dockqs=[]
        rmsds_lig = []
        patches_batch = []
        names = []
        try:
            while len(seqs1) < batch_size:

                seq1, self_1, seq2, self_2, geo_patches, patches, label, rec_rmsd, lig_rmsd, dockq,name = line_gen.__next__()

                patches_batch.append(patches)
                geoMats.append(geo_patches)
                seqs1.append(seq1)
                distograms1.append(self_1)
                seqs2.append(seq2)
                distograms2.append(self_2)
                labels.append(label)
                rmsds_lig.append(lig_rmsd)
                # rmsds_rec.append(rec_rmsd)
                dockqs.append(dockq)
                if evaluate:
                    rmsds_rec.append(rec_rmsd)
                    names.append(name)
                line_number += 1
        except EOFError:
            print("trans file finished at file_genarator_pre_batched")
            # if len(patches_batch)==0:
            raise EOFError
        if get_rmsd:
            if evaluate:
                batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                          "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                          "input_6": tf.stack(patches_batch),
                          },
                         # {"classification": tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (batch_size, 1)),
                          {"classification": tf.stack(labels),
                           "rec_rmsd": tf.stack(rmsds_rec),
                          "lig_rmsd": tf.stack(rmsds_lig),
                          "dockQ":tf.stack(dockqs),
                          "names":tf.convert_to_tensor(names)
                          })
            else:
                batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                          "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                          "input_6": tf.stack(patches_batch),
                          },
                         {"classification": tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (batch_size, 1)),
                          "lig_rmsd": tf.stack(rmsds_lig),
                          "dockQ":to_one_hot(dockqs)
                          })
        else:
            batch = ({"input_1": tf.stack(seqs1), "input_2": tf.stack(distograms1), "input_3": tf.stack(seqs2),
                      "input_4": tf.stack(distograms2), "input_5": tf.stack(geoMats),
                      "input_6": tf.stack(patches_batch)},
                     tf.convert_to_tensor(labels, dtype=tf.float32))
        # dockqs=tf.convert_to_tensor(dockqs)
        # print(dockqs.shape)
        # if not layer.is_adapted:
        #     layer.adapt(dockqs)
        # layer(dockqs)
        # print(i,patches_batch)
        i+=1
        yield batch


def create_gen(type, data_file, workdir, data_postfix, sample_rate=4, prot_dict=None, batch_size=8,
               patch_size=30, size_r=1000, size_l=1000, data_dir="", title_len=4, batched=False):
    if type == "balanced":
        print("Here")
        if batched:
            gen = balanced_genarator_pre_batched(data_file, prot_dict=prot_dict, batch_size=batch_size,
                                                 sample_rate=sample_rate,
                                                 size_r=size_r, size_l=size_l, data_dir=data_dir, title_len=title_len,
                                                 workdir=workdir, data_postfix=data_postfix)
        else:
            gen = balanced_genarator(data_file, prot_dict=prot_dict, batch_size=batch_size, sample_rate=sample_rate,
                                     size_r=size_r, size_l=size_l, data_dir=data_dir, title_len=title_len,
                                     workdir=workdir, data_postfix=data_postfix)
    else:
        print("Here2")
        if batched:
            gen = file_genarator_pre_batched(data_file, data_postfix=data_postfix, prot_dict=prot_dict, batch_size=batch_size,
                                             patch_size=patch_size, size_r=size_r, size_l=size_l, data_dir=data_dir,
                                             title_len=title_len, workdir=workdir)
        else:
            gen = complex_genarator(data_file, data_postfix=data_postfix, prot_dict=prot_dict, batch_size=batch_size,
                                patch_size=patch_size, size_r=size_r, size_l=size_l, data_dir=data_dir,
                                title_len=title_len, workdir=workdir)
    return gen


def gen_cycler(type, data_file, prot_dict=None, workdir=None, sample_rate=4, batch_size=8,
               patch_size=30, data_dir="", data_postfix="oversample", title_len=4, size_r=1000, size_l=1000,
               batched=False):
    while True:
        print(data_postfix)
        gen = create_gen(type, data_file, workdir, prot_dict=prot_dict, sample_rate=sample_rate,
                         batch_size=batch_size,
                         patch_size=patch_size, data_dir=data_dir, title_len=title_len, size_r=size_r, size_l=size_l,
                         data_postfix=data_postfix, batched=batched)
        try:
            for batch in gen:
                yield batch
        except EOFError:
            print("XXXXXXXXX restarted gen of type " + type)

#
if __name__ == '__main__':
    train_prot = "/cs/labs/dina/matanhalfon/CAPRI/" + config["workdir"] + "/train_pdbs" + config["pdb_file_suffix"]

    # test_prot = "/cs/labs/dina/matanhalfon/CAPRI/" + config["workdir"] + "/test_pdbs" + config["pdb_file_suffix"]

    train_prot_dict = build_prot_dict(train_prot, config["arch"]["size_r"], config["arch"]["size_l"],
                                      workdir=config["workdir"].split("/")[0], line_len_R=config["line_len_R"],
                                      line_len_L=config["line_len_L"],
                                      suffixL=config["suffix_l"], suffixR=config["suffix_r"],
                                      self_distogram=config["self_distogram_dir"])
    # test_prot_dict = build_prot_dict(test_prot, config["arch"]["size_r"], config["arch"]["size_l"],
    #                                  workdir=config["workdir"].split("/")[0], line_len_R=config["line_len_R"],
    #                                  line_len_L=config["line_len_L"]
    #                                  , suffixL=config["suffix_l"], suffixR=config["suffix_r"],
    #                                  self_distogram=config["self_distogram_dir"])
    # gen_batched=gen_cycler("comp","test_trans",prot_dict=test_prot_dict,workdir=config["workdir"],
    #                        batch_size=config["hyper"]["batch_size"],data_postfix="_"+config["data_type_test"],patch_size=config["arch"]["patch_size"],
    #                        size_r=config["arch"]["size_r"], size_l=config["arch"]["size_l"],data_dir=config["patch_dir_complex"]
    #                        ,title_len=config["line_len_R"],batched=True)

    gen_batched = gen_cycler("balanced", config["data_file"],prot_dict=train_prot_dict,sample_rate=config["hyper"]["sample_rate"], workdir=config["workdir"],
                                      batch_size=config["hyper"]["batch_size"],
                                    size_r=config["arch"]["size_r"], size_l=config["arch"]["size_l"],patch_size=config["arch"]["patch_size"],data_dir=config["patch_dir"],
                                    title_len=config["line_len_R"],data_postfix=config["data_type_train"],batched=True)


    # print(batch_1["input_5"])
    # print("     XXXXX       ")
    # print(batch_2["input_5"])
    i=0
    for batch,label in gen_batched:
        if i%2000==0:
            print(i)
        total = tf.reduce_sum(tf.abs(batch["input_5"]))
        tf.debugging.assert_none_equal(total,0.0)
        i+=1
        # batch_1, label_1 = gen_reg.__next__()
        # for key  in batch_1.keys():
        #     try:
        #         tf.debugging.assert_equal(batch_1[key],batch_2[key],message="error at key="+key)
        #     except:
        #         print("failed at key "+key)
        #         tf.debugging.assert_equal(label_1["classification"],label_2["classification"])
        #         for  i in range (batch_1[key].shape[0]):
        #             tf.debugging.assert_equal(batch_1[key][i], batch_2[key][i], message="error at index=" + str(i))
