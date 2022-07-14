import os
import os.path as osp
import pandas as pd
import numpy as np
from termcolor import colored
import random as rd
from preprocessing import preprosser
from Bio import SeqIO, pairwise2
from Bio.PDB import Polypeptide, is_aa, PDBParser
import pathlib
import re
import seaborn as sns
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import subprocess

get_frag_chain = "~dina/utils/get_frag_chain.Linux"
get_name_chain = "/cs/staff/dina/scripts/namechain.pl"
comndSeq = "~dina/utils/pdb2fasta"
comdDssp = "/cs/staff/dina/software/Staccato/mkdssps"
NanoNet_location = "/cs/labs/dina/tomer.cohen13/matan/NanoNet/"
comdNanoNet = "python3 NanoNet.py "
nanoNetSuffix = "-r -p /cs/labs/dina/tomer.cohen13/pulchra_306/pulchra"
getChains = " /cs/staff/dina/scripts/chainSelector.pl"
extractChains = "/cs/staff/dina/utils/getChain.Linux"
alignComd = "/cs/staff/dina/scripts/alignRMSD.pl"
nameComd = "~dina/scripts/namechain.pl"
rmsd_s = "~dina/utils/rmsd -t "
rmsd_alighn = " ~dina/projects/rmsd/rmsd3.linux "
Alphafold = "/cs/staff/dina/projects/collabFold_script.py"
clustring = "/cs/labs/dina/dina/projects/InterfaceClustering/interface_cluster.linux "
renumber = "/cs/staff/dina/utils/srcs/renumber/renumber"

iplddt="~dina/utils//srcs/interface/interface "
predict = "/cs/labs/dina/tomer.cohen13/matan/NanoNet/NanoNet.py"

HEAVY_SEQ = "QVQLKESGPGLVAPSQSLSITCTVSGFSLTDYGVDWVRQPPGKGLEWLGMIWGDGSTDYNSALKSRLSITKDNSKSQVFLKMNSLQTDDTARYYCVRDPADYGNYDYALDYWGQGTSVTVSS"
LIGHT_SEQ = "DIELTQSPDSLAVSLGQRATISCRASESVDSYGNSFMQWYQQKPGQPPKLLIYRASNLESGIPARFSGTGSRTDFTLTINPVEADDVATYYCQQSDEYPYMYTFGGGTKLEIKR"

UNIQE_AA = {"UNK": "X", "TYS": "Y", "FME": "M", "PCA": "Q", "CSD": "C", "MLY": "K", "SEP": "S", "YCM": "C", "CSX": "C",
            "NEP": "H", "IAS": "D"}

def get_seq(sourceDir, out_dir="seqs"):
    i = 0
    for dir in os.listdir(sourceDir):
        if osp.isdir(osp.join(sourceDir, dir)):
            for file in os.listdir(osp.join(sourceDir, dir)):
                if file.endswith("Ab.pdb"):
                    name = file.split(".")[0]
                    comd = comndSeq + " -s " + osp.join(sourceDir, dir, file) + " >> " + osp.join(out_dir,
                                                                                                  name + ".fasta")
                    os.system(comd)
                    print(i, " : ", file)
                    i += 1


def get_comp_seq(source_dir, out_dir,pdbs,suffix_ab="_Ab.pdb",suffix_ag="_Ag.pdb"):
    i = 0
    for dir in os.listdir(source_dir):
        if dir in pdbs:
            if osp.isdir(osp.join(source_dir, dir)):
                header = ">" + osp.join(source_dir, dir) + ".pdb chain A"
                comd_ab = comndSeq + " -s " + osp.join(source_dir, dir, dir + suffix_ab)
                comd_ag = comndSeq + " -s " + osp.join(source_dir, dir, dir + suffix_ag)
                fasta_ag = subprocess.check_output(comd_ag, shell=True).decode("utf-8").strip().split("\n")
                seq_ag = "\n".join(fasta_ag[1:])
                fasta_ab = subprocess.check_output(comd_ab, shell=True).decode("utf-8").strip().split("\n")
                seq_ab = "\n".join(fasta_ab[1:])
                seq = header + "\n" + seq_ab + ":" + seq_ag
                print(seq)
                verifyDir(osp.join(out_dir, dir))
                with open(osp.join(out_dir, dir, dir + "_seq.fasta"), 'w+') as f:
                    f.writelines(seq)
                print(i, " : ", dir)
                i += 1


def check_for_pdb(dir):
    for file in os.listdir(dir):
        if file.endswith(".pdb"):
            return True
    return False


def fold_fasta(dir,pdbs):
    i = 0
    pdbs_test = {"1IQD", "2W9E", "1KXQ"}
    for prot_dir in pdbs_test:
        if check_for_pdb( osp.join(dir,prot_dir)):
            i += 1
            print(osp.join(dir, prot_dir))
            file = prot_dir + "_seq.fasta"
            os.chdir(osp.join(dir,prot_dir))
            subprocess.run("python3 " + Alphafold + " " + file, shell=True)
            os.chdir("../../..")
    print(i)

def pdb_to_dssp(workdir,file,out_dir):
        name = file.split(".pdb")[0]
        comd = comdDssp + " " + osp.join(workdir, file)
        os.system(comd)
        print(osp.join(workdir, file + ".dssp "))
        print(osp.join(out_dir, name + ".dssp"))
        mv_comnd = "mv " + osp.join( workdir, file + ".dssp ") + osp.join(out_dir, name + ".dssp")
        os.system(mv_comnd)


def write_dssp(source_dir, out_dir,prot_suffix=".pdb"):
    i = 0
    print("here")
    for dir in os.listdir(source_dir):
        if osp.isdir(osp.join(source_dir,dir)):
            for file  in os.listdir(osp.join(source_dir,dir)):
                if file.endswith(prot_suffix) or file.endswith("_Ag.pdb") :
                    pdb_to_dssp(osp.join(source_dir,dir), file)
                    # name = file.split(".pdb")[0]
                    # comd = comdDssp + " " + osp.join(source_dir,dir, file)
                    # os.system(comd)
                    # print(osp.join(source_dir, file + ".dssp "))
                    # print(osp.join(out_dir, name + ".dssp"))
                    # mv_comnd = "mv " + osp.join(source_dir,dir, file + ".dssp ") + osp.join(out_dir, name + ".dssp")
                    # os.system(mv_comnd)
                    print(i, " : ", file)
                    i += 1


def get_name(sourceDir):
    for file in os.listdir(sourceDir):
        if (file.endswith("l_u.pdb")):
            ligand = file
        elif (file.endswith("r_u.pdb")):
            rec = file
    return rec, ligand


def get_label(recRmsd, ligRmsd):
    if (recRmsd < 10 or ligRmsd < 4):  # 5,2
        return 1
    return 0


def get_easy_label(recRmsd, ligRmsd):
    if (recRmsd < 8 or ligRmsd < 4):
        return 1
    elif recRmsd > 25 or ligRmsd > 10:
        return 0
    return -1


def write_trans_line(line, writer, label_func, numOfNeg, rec, lig, i, is_aa=False):
    splited = line.split('|')
    if len(splited) < 13:
        return 0, i
    trans = splited[14].strip()
    rmsd = splited[1]
    rmsds = rmsd.split('(')
    rec_RMSD = float(rmsds[0])
    lig_RMSD = float(rmsds[1].split(')')[0])
    label = label_func(rec_RMSD, lig_RMSD)
    if label == 0:
        numOfNeg += 1
        if numOfNeg > 20500:  # change to 1200
            if numOfNeg == 20501:
                print("over 20500 neg at ", i)
            return 1, i
    if is_aa:
        line = str(i) + " " + lig + " " + rec + " " + trans + " " + str(label) + " " + str(rec_RMSD) + " " + str(
            lig_RMSD) + "\n"
    else:
        line = str(i) + " " + rec + " " + lig + " " + trans + " " + str(label) + " " + str(rec_RMSD) + " " + str(
            lig_RMSD) + "\n"

    writer.write((line))
    if label == 0:
        return 1, i + 1
    else:
        return 0, i + 1


def write_lines(lines, writer):
    header = True
    pos_count = 0
    numOfNeg = 0
    last_writen = 30000
    for i, line in enumerate(lines):
        if (line.find('#') != -1):
            header = False
            continue
        if not header:
            label, last_writen = write_trans_line(line, writer, get_label, numOfNeg, osp.basename(rec),
                                                  osp.basename(lig), last_writen)
            if label == 0:
                pos_count += 1
                if pos_count > 350:
                    print("over 300 pos")
                    break
            numOfNeg += label

            # splited=line.split('|')
            # if len(splited)<13:
            #     continue
            # trans=splited[14].strip()
            # rmsd=splited[1]
            # rmsds=rmsd.split('(')
            # rec_RMSD=float(rmsds[0])
            # lig_RMSD=float(rmsds[1].split(')')[0])
            # if get_label(rec_RMSD, lig_RMSD)== "0":
            #     numOfNeg+=1
            #     if numOfNeg>1300 :
            #         if numOfNeg==1301:
            #             print("over 1200 neg at ",i)
            #         continue
            # line= str(i) +" " + rec +" " + lig +" " + trans +" " + get_label(rec_RMSD, lig_RMSD) + "\n"
            # writer.write((line))
        elif (line.startswith("ligandPdb")):
            l_split = line.split("\t")
            lig = l_split[-1].strip()
        elif (line.startswith("receptorPdb")):
            r_split = line.split("\t")
            rec = r_split[-1].strip()
    print("num of  pos- ", pos_count)
    print("num of neg- ", numOfNeg)


def getTransFile(sourceDir, name):
    with open(name, 'w') as t_file:
        # with open(name+"_easy",'w')as te_file:
        for file in os.listdir(sourceDir):
            if (file == dockingFIle):
                # if(file.endswith(".res")):
                with open(osp.join(sourceDir, file)) as res:
                    lines = res.readlines()
                    return write_lines(lines, t_file)


def over_sampleList(lst, factor):
    while (1 < factor):
        lst[0:0] = lst
        factor /= 2
    return lst


def aggregate_by_batch(dir, writer):
    for i, file in enumerate(os.listdir(dir)):
        pos = []
        neg = []
        with open(osp.join(dir, file)) as trFile:
            lines = trFile.readlines()
            print(file, "  number of lines- ", len(lines))
            for line in lines:
                line = line.strip()
                label = line.split(" ")[-3]
                if label == "1":
                    pos.append(line)
                elif label == "0":
                    neg.append(line)
                writer.write(line)
                writer.write("\n")
        print("pos num - ", len(pos))
        print("neg num - ", len(neg))


def aggregat_by_label(dir, pdb_file):
    with open(pdb_file) as pd:
        pdbs = {prot.strip() for prot in pd}
        # print(pdb_file)
        # print(pdbs)
    pos = []
    neg = []
    for i, file in enumerate(os.listdir(dir)):
        if file not in pdbs:
            continue
        with open(osp.join(dir, file)) as trFile:
            lines = trFile.readlines()
            for line in lines:
                line = line.strip()
                label = line.split(" ")[-3]
                if label == "1":
                    pos.append(line)
                elif label == "0":
                    # continue
                    neg.append(line)
                else:
                    print("WTF:", line)
            # file_pos=over_sampleList(file_pos,4)
            # if len(pos)!=0:
            #     pos[0:0]=file_pos
            #     neg[0:0]=file_neg
    rd.shuffle(neg)
    rd.shuffle(pos)
    # neg=neg[:int(len(neg)/100)] ##under sample
    pos_size = len(pos)
    neg_size = len(neg)
    print("pos_size: ", pos_size, "neg_size: ", neg_size)
    # neg[0:0]=pos
    return pos, neg


def writeFile(lines, name):
    with open(name, 'w') as file:
        for line in lines:
            file.write(line.strip())
            file.write("\n")


def split_data(data, outdir, name):
    trans, labels = [], []
    for i, line in enumerate(data):
        line = line.strip()
        splited = line.split(" ")
        label = splited[-3]
        transnformation = " ".join(splited[:-3])
        trans.append(transnformation)
        labels.append(label)
    # X_train, X_test, y_train, y_test= train_test_split(trans,labels,test_size=0.2)
    writeFile(trans, osp.join(outdir, name + "_data"))
    writeFile(labels, osp.join(outdir, name + "_label"))
    # writeFile(y_test,osp.join(outdir,"test_label"))
    # writeFile(y_train, osp.join(outdir,"train_label"))


def get_train_by_label(trainX, train_label, workdir):
    pos = []
    neg = []
    with open(osp.join(workdir, trainX)) as data:
        with open(osp.join(workdir, train_label)) as label:
            d_lines = data.readlines()
            l_lines = label.readlines()
            if len(d_lines) != len(l_lines):
                print("not same length")

            for i, d in enumerate(d_lines):

                label = l_lines[i].strip()
                d_line = d.strip()
                if label.strip() == '0':
                    neg.append(d_line)
                else:
                    pos.append(d_line)
    writeFile(pos, osp.join(workdir, "train_pos"))
    writeFile(neg, osp.join(workdir, "train_neg"))


def remove_missing_batches(lines, struct_dir, out_name, split=False):
    count = 0
    with open(out_name, 'w') as out:
        for i, line in enumerate(lines):
            if i % 5000 == 0:
                print("index: ", i, " writen: ", count)
            splited = line.split(" ")
            index = splited[0]
            rec = splited[1]
            lig = splited[2]
            transname = rec + "X" + lig + "transform_number_" + index
            prot = rec[:4]
            path = osp.join(struct_dir, prot, transname)
            if osp.isfile(path):
                count += 1
                out.write(line + "\n")
                # out_lines.append(line)
    # with open(data_file+"fixed",'w') as out:
    #     out.writelines(out_lines)


def write_small_data(origin, short, workdir, sort=False):
    black_list = preprosser.get_to_long(workdir + "/dssp")
    print(black_list)
    with open(osp.join(workdir, origin)) as ori:
        with open(osp.join(workdir, short), 'w+') as s:
            short_lines = []
            for i, line in enumerate(ori):
                split = line.split(" ")[0]
                name = split[1][:4]
                if name not in black_list:
                    short_lines.append(line)
                # else:
                #     print(line)
            if sort:
                short_lines = sorted(short_lines, key=lambda x: x.split(" ")[-2])
            s.writelines(short_lines)


def get_prots_name_from_soap(path):
    with open(path) as f:
        line_1 = f.readline().strip()
        line_2 = f.readline().strip()
        receptor = line_1.split(" ")[2]
        ligand = line_2.split(" ")[2]
    return receptor, ligand


def soap_to_pandas(path):
    receptor, ligand = get_prots_name_from_soap(path)
    df = pd.read_csv(path, header=3)
    if not len(df):
        raise pd.errors.EmptyDataError
    s = df.iloc[:, 0]
    splited = s.str.split("|")
    dict = {"soap": splited.str[1],
            "transformation": splited.str[6]}
    df = pd.DataFrame(dict)
    df["soap"] = df["soap"].astype('float')
    # df["Zscore"] = df["Zscore"].astype('float')
    # df=df.set_index(splited.str[0])
    sorted = df.sort_values("soap")
    # sorted.index
    return sorted, receptor, ligand


def dock_to_pandas(path):
    df = pd.read_csv(path, header=21, skipfooter=1)  # 23
    s = df.iloc[:, 0]
    splited = s.str.split("|")
    rmsds = splited.str[1]
    rec_rmsds = rmsds.str.split("(").str[0].astype("float")
    lig_rmsds = rmsds.str.split("(").str[1].str.split(")").str[0].astype("float")
    labels = (rec_rmsds < 10) | (lig_rmsds < 4)
    dockQ = splited.str[2]
    # trans=splited.str[14]
    dict = {"rec_rmsd": rec_rmsds,
            "lig_rmsd": lig_rmsds, "dockQ": dockQ, "labels": labels}
    df = pd.DataFrame(dict)
    df["rec_rmsd"] = df["rec_rmsd"].astype('float')
    df["lig_rmsd"] = df["lig_rmsd"].astype('float')
    # df=df.set_index(splited.str[14])
    # df=df.sort_values("rec_rmsd")
    return df


def extract_from_rms(df):
    s = df.iloc[:, 0]
    splited = s.str.split("|")
    rmsds = splited.str[1]
    rec_rmsds = rmsds.str.split("(").str[0].astype("float")
    lig_rmsds = rmsds.str.split("(").str[1].str.split(")").str[0].astype("float")
    labels = (rec_rmsds < 10) | (lig_rmsds < 4)
    dockQ = splited.str[2]

    return rec_rmsds, lig_rmsds, labels, dockQ


def rmsd_to_pandas(path):
    df = pd.read_csv(path, skipfooter=2, header=None)  # 23
    rec_rmsds, lig_rmsds, labels, dockQ = extract_from_rms(df)
    # trans=splited.str[14]
    dict = {"rec_rmsd": rec_rmsds,
            "lig_rmsd": lig_rmsds, "dockQ": dockQ, "labels": labels}
    df = pd.DataFrame(dict)
    df["rec_rmsd"] = df["rec_rmsd"].astype('float')
    df["lig_rmsd"] = df["lig_rmsd"].astype('float')
    # df=df.set_index(splited.str[0])
    # df=df.sort_values("rec_rmsd")
    return df


def merage_tables(dock_path, soap_path, take_top=0, out_dir=""):
    try:
        df, receptor, ligand = soap_to_pandas(soap_path)
        df1 = dock_to_pandas(dock_path)
        meraged = df.join(df1)
        if (meraged.isnull().values.any()):
            raise ArithmeticError
        # meraged = meraged.reset_index(drop=True)
        meraged.insert(0, '   ,mn of complex', ligand + " " + receptor)
        if take_top:
            meraged = meraged.head(take_top)
        meraged.to_csv(out_dir, sep="\t", quoting=csv.QUOTE_NONE, quotechar="",
                       escapechar="\\",header=None)
        print(out_dir)
    except pd.errors.EmptyDataError:
        print("At file " + dock_path + " the csv is empty ")


def preduce_tran_dir(workdir="", struct_dir="", trans_dir="", docking_file=dockingFile, soap_file=soapFile, take_top=0):
    transDir_cur = osp.join(workdir, trans_dir)
    verifyDir(transDir_cur)
    structDir_cur = osp.join(workdir, struct_dir)
    for i, dir in enumerate(os.listdir(structDir_cur)):
        if osp.isdir(osp.join(structDir_cur, dir)):
            print("file num: ", i, "dir name: ", dir)
            file_name = osp.joicdcn(transDir_cur, dir)
            # if osp.exists(file_name):
            #     print("file - "+file_name+ " exists ")
            #     continue
            try:
                merage_tables(osp.join(structDir_cur, dir, docking_file), osp.join(structDir_cur, dir, soap_file),
                              out_dir=osp.join(transDir_cur, dir), take_top=take_top)
            except FileNotFoundError:
                print("file not found at: " + dir)
            # getTransFile(osp.join(structDir_cur, dir), file)


def preduce_spesifisity_dir(workdir=""):
    spec_data_cur = osp.join(workdir, spec_dir)
    verifyDir(spec_data_cur)
    structDir_cur = osp.join(workdir, structDir)
    for i, dir in enumerate(os.listdir(structDir_cur)):
        if osp.isdir(osp.join(structDir_cur, dir)):
            print("file num: ", i, "dir name: ", dir)
            # if osp.exists(file_name):
            #     print("file - "+file_name+ " exists ")
            #     continue
            try:
                merage_tables(osp.join(structDir, dir, dockingFile), osp.join(structDir, dir, soapFile), take_top=100,
                              out_dir=osp.join(spec_data_cur, dir))
            except FileNotFoundError:
                print("file not found at: " + dir)


def get_pdb_from_file(file, line_len=0):
    pdbs = set()
    with open(file) as ls:
        for line in ls:
            if line_len:
                pdbs.add((line.split(" ")[0][:line_len]).strip())
            else:
                pdbs.add((line.split(" ")[0]).strip())
    return pdbs


def del_abcent(set, dir):
    for file in os.listdir(dir):
        if file not in set:
            os.remove(osp.join(dir, file))


def get_specifisity_prots(path='ABDB/cd_hit_50.1', workdir="ABDB"):
    with open(path) as file:
        lines = file.readlines()
    unique = set([line.split("/")[-2] for line in lines if line.startswith(">")])
    black_list = set(
        preprosser.get_to_long(osp.join(workdir, "dssp"), line_end=6, suffix_r="_Ab_nano", suffix_l="_Ag", len_r=250,
                               len_l=700))
    cleaned = unique - black_list
    from_test = get_pdb_from_file(osp.join("ABDB/AlphaFold_data_dir/AlphaFold_data", "test_pdbs"))
    final = set()
    for p in from_test:
        if osp.exists(osp.join(workdir, "AlphaFold_docking", p, p + "_Ab.pdb")) and (p[:6] in cleaned):
            final.add(p[:6])
    # from_test -= to_remove
    return final


def write_specifisity_dir(data_dir, out_dir):
    final = get_pdb_from_file("bench5AA/db5_docking/prots", 6)
    # final = get_specifisity_prots()
    # with open("ABDB/specifisity_prots.txt", "w+") as f:
    #     for p in final:
    #         f.write(p)
    #         f.write("\n")
    print(final)
    print(len(final))
    for p1 in final:
        for p2 in final:
            dest = osp.join(out_dir, p1 + "_" + p2)
            if osp.exists(dest):
                continue
            verifyDir(dest)
            # os.system("cp " + osp.join(data_dir, p1 + "_model_0", p1 + "_model_0_Ab.pdb") + " " + dest)
            os.system("cp " + osp.join(data_dir, p1 + "_model_0", p1 + "_Ab.pdb") + " " + dest)
            os.system("cp " + osp.join(data_dir, p1 + "_model_0", p1 + "_l_u.pdb") + " " + dest)
            # os.system("cp " + osp.join(data_dir, p2 + "_model_0", p2 + "_model_0_Ag.pdb") + " " + dest)


def bad_folded(path):
    df = pd.read_csv(path)
    high_rmsd = set(df[df["rmsd"] > 3]["name"])
    return high_rmsd


def split_pdbs(workdir, out_dir, trans_dir, test_size=45, line_end=4, suffix_r="_u1", suffix_l="_u2", size_r=250,
               size_l=700):
    pdbs = set()
    proteins = set()
    black_list = preprosser.get_to_long(workdir + "/dssp", line_end=line_end, suffix_r=suffix_r, suffix_l=suffix_l,
                                        len_r=size_r, len_l=size_l)
    # bf = bad_folded("scores.csv")
    # print()
    # with open("dockground/LIST_noAApdbs.txt") as no_aa:
    #     lines=no_aa.readlines()
    #     no_aa_list=[line.strip() for line in lines]
    #     print(len(no_aa_list))
    for file in os.listdir(osp.join(workdir, trans_dir)):
        if (file[:line_end] not in black_list):
            # if (file not in bf):
            pdbs.add(file)
            proteins.add(file[:line_end])
        # else:
        #     print(file, " ", (file.split("_")[0] not in black_list))
        # , "  ", ((file.split("_")[0] in no_aa_list)))
    # pdbs_list = list(pdbs)
    # print(pdbs_list)
    # rd.shuffle(pdbs_list)
    test_prot = set(rd.sample(proteins, test_size))
    train_prot = proteins - test_prot
    print("num prot test :", len(test_prot))
    print("num prot train :", len(train_prot))
    # print(train_prot)
    test = [p for p in pdbs if p[:line_end] in test_prot]
    train = [p for p in pdbs if p[:line_end] in train_prot]
    print("num models test :", len(test))
    print("num models train :", len(train))
    # train=remove_seq_identity(test,train,osp.join(workdir,"seqs"),th=0.97)
    with open(out_dir + "/train_pdbs", 'w') as train_f:
        for line in train:
            train_f.write(line + "\n")
    with open(out_dir + "/test_pdbs", 'w') as test_f:
        for line in test:
            test_f.write(line + "\n")
    with open(out_dir + "/all_pdbs", 'w') as train_f:
        for line in pdbs:
            train_f.write(line + "\n")


def set_data(workdir,struct_dir,trans_dir,docking_file=dockingFile, soap_file=soapFile,prot_suffix=".pdb"
             ,suffix_l="_Ag.pdb",suffix_r="_Ab.pdb",line_len=6 ,size_r=250,size_l=700):
    # write_dssp(osp.join(workdir, struct_dir), osp.join(workdir, "dssp"),prot_suffix=prot_suffix)  ###create the dssp files
    # preduce_tran_dir(workdir=workdir,struct_dir=struct_dir,trans_dir=trans_dir,docking_file=docking_file, soap_file=soap_file)
    split_pdbs(workdir,workdir, trans_dir, line_end=line_len, suffix_r=suffix_r, suffix_l=suffix_l, size_r=size_r,
               size_l=size_l)


def plot_metrics(history, title):
    """
    plot the metrics of the train and validation after each epoch
    :param history: the model history object
    :param title: the figure title
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history["val_" + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
    plt.savefig(str(title) + ".png")

def parse_alphafold_log_line(line,prot):
    model_id=prot[:6] + "_model_" + str(int(line[30])-1)
    plddt = re.findall('\d+\.\d+|\d+', line)[-2]
    ptmscore = re.findall('\d+\.\d+| \d+ ', line)[-1]
    return model_id,plddt,ptmscore


def Alphafold_score(file, prot,splited_modeles="ABDB/AlphaFold_splited_modeled",suffix_r="_Ab.pdb",suffix_l="_Ag.pdb"):
    with open(file) as f:
        lines_relevant = [l for l in f.readlines() if "pLDDT" in l]
    models=[]
    plddts=[]
    ptmscores=[]
    try:
        for line in lines_relevant:
            model_name, plddt, ptmscore=parse_alphafold_log_line(line,prot)
            if model_name not in models:
                models.append(model_name)
                plddts.append(plddt)
                ptmscores.append(ptmscore)
    except ValueError :
        print("crashed at")
    # models = [prot[:6] + "_model_" + str(int(l[30])-1) for l in lines_relevant]
    # l[24:31]
    iplddts=get_ipLDDT(splited_modeles,models,suffix_r=suffix_r,suffix_l=suffix_l)
    # plddt = [re.findall('\d+\.\d+|\d+', line)[-2] for line in lines_relevant]
    # ptmscore = [re.findall('\d+\.\d+| \d+ ', line)[-1] for line in lines_relevant]
    return pd.DataFrame({"name": models, "iplddt":iplddts,"pLDDT": plddts, "ptmscore": ptmscores})


def get_modeled_rmsd(dir, prot, orig_dir):
    origin_pdb = osp.join(orig_dir, prot[:6], prot + ".pdb")
    prots = []
    model = []
    rmsds = []
    for i in range(5):
        modeled = osp.join(dir,
                           prot,
                           "ABDB_splited_pdb_" + prot[:6] + "_" + prot + ".pdb_chain_L*model_" + str(i + 1) + ".pdb")
        comd = rmsd_s + origin_pdb + " " + modeled
        rmsd = subprocess.check_output(comd, shell=True).decode("utf-8").strip().split(" ")[-1].split("\n")[0]
        prots.append(prot[:6] + "_model_" + str(i ))
        rmsds.append(float(rmsd))
        model.append(i + 1)
    return pd.DataFrame({"name": prots, "rmsd": rmsds, "model": model})


def avg_tables_by_col(tables, col_name):
    orig=tables[0]
    for table in tables[1:]:

        orig[col_name]+=table[col_name]
    orig[col_name]=orig[col_name]/len(tables)
    re_sorted=orig.sort_values(col_name,ascending=False)
    return re_sorted

def sort_by_trans(df, orig_col, target_col, sort_by="score", ascending=True):
    if sort_by=="soap":
        df[sort_by]=(-df[sort_by])
    df["score"] = (df[sort_by] - df[sort_by].mean()) / df[sort_by].std()
    sorted = df.sort_values(orig_col, ascending=ascending)
    renamed=sorted.rename(columns={orig_col:target_col})
    return renamed


def get_ensambel(dir_models,test_pdbs,ensambel_name,suffixes,use_Soap=False):
    test_list = get_pdb_from_file(osp.join(test_pdbs, "test_pdbs"))
    for prot in test_list:
        print(prot)
        normaelized_taels = []
        for s in suffixes:
            eval_name="evaluation"+s+prot
            try:
                df=pd.read_csv(osp.join(dir_models,prot[:6],eval_name),sep="\t",index_col=False)
            except pd.errors.EmptyDataError:
                print(eval_name+" is empty")
                continue
            df_sorted=sort_by_trans(df, "trans", "trans")
            # df["score"]=(df["score"]-df["score"].mean())/df["score"].std()
            # sorted = df.sort_values("trans")
            normaelized_taels.append(df_sorted)
        if use_Soap:
            df_soap=pd.read_csv(osp.join(dir_models,prot[:6],"soap_results_"+prot),sep="\t",index_col=False)
            df_soap_sorted=sort_by_trans(df_soap, "transformation", "trans", sort_by="soap", ascending=True)
            normaelized_taels.append(df_soap_sorted)
        ensembel_df=avg_tables_by_col(normaelized_taels, "score")
        ensembel_df.to_csv(osp.join(dir_models,prot[:6],"evaluation_"+ensambel_name+prot),sep="\t",index=False)


def get_ipLDDT(dir,models,suffix_r="_Ab.pdb",suffix_l="_Ag.pdb"):
    ipLDDTs=[]
    for model in models:
        comd=iplddt+osp.join(dir,model[:4],model+suffix_r+" ")+osp.join(dir,model[:4],model+suffix_l+" ")+" 6 -i"
        score = str(subprocess.check_output(comd, shell=True).decode("utf-8").strip().split("\n")[-1].split(" = ")[1])
        ipLDDTs.append(score)
    return ipLDDTs


def get_consencse(dir,data_file,workdir,alphaFold_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/AlphaFold_splited_modeled",
                  eval_dir="ABDB/AlphaFold_docking_patch_data",save_dir="",model_suffix="sw130_",res_file="results.csv"):
    pdbs=get_pdb_from_file(osp.join(workdir,data_file),line_len=6)
    output = pd.DataFrame(columns=["name", "pLDDT", "ptmscore", "rmsd"])
    bad_ones = []
    # black_list = set(
    #     preprosser.get_to_long(osp.join(workdir, "dssp"), line_end=6, suffix_r="_Ab_nano", suffix_l="_Ag", len_r=250,
    #                            len_l=700))
    for prot in os.listdir(dir):
        # if prot[:6] in black_list or (prot not in pdbs):
        if prot not in pdbs:
            # print("skipping")
            continue
        try:
            print(prot)
            df1 = Alphafold_score(osp.join(dir, prot, "log.txt"), prot)
            df2 = pd.read_csv(osp.join(alphaFold_dir, prot, res_file), sep="\t", index_col=0)
            df2["name"]=[prot+"_model_"+str(i) for i in range(5)]
            eval_df=pd.read_csv(osp.join(eval_dir,"midfiles",prot,"evaluation"+model_suffix+prot),sep="\t",index_col=0)
            eval_df["name"]=eval_df["name"].str[2:16]
            eval_df=eval_df[["name","score",'lig_rmsd',"rec_rmsd","dockQ",'predicted_dockQ']]
            meraged = df1.merge(df2,on="name")
            meraged = meraged.merge(eval_df,on="name")
            meraged.to_csv(osp.join(save_dir,prot+model_suffix+"consensus.csv"),sep="\t",index=False)
        except subprocess.CalledProcessError:
            bad_ones.append(prot)
        except ValueError:
            print("error at prot")
    print(bad_ones)

def build_rmsd_Alphafold_table(dir, workdir="ABDB",alphaFold_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/AlphaFold_splited_modeled",res_file="results.csv"):
    output = pd.DataFrame(columns=["name", "pLDDT", "ptmscore", "rmsd"])
    bad_ones = []
    black_list = set(
        preprosser.get_to_long(osp.join(workdir, "dssp"), line_end=6, suffix_r="_Ab_nano", suffix_l="_Ag", len_r=250,
                               len_l=700))
    for prot in os.listdir(dir):
        if prot[:6] in black_list:
            print("skipping")
            continue
        try:
            df1 = Alphafold_score(osp.join(dir, prot, "log.txt"), prot)
            df2 = get_modeled_rmsd(dir, prot, "/cs/labs/dina/matanhalfon/CAPRI/ABDB/AlphaFold_splited_modeled")
            # df2=pd.read_csv(osp.join(alphaFold_dir,prot,res_file),sep="\t",index_col=0)
            meraged = df1.join(df2)
            output = pd.concat([meraged, output])
        except subprocess.CalledProcessError:
            bad_ones.append(prot)
    print(bad_ones)
    output.to_csv("scores_folded.csv", index=False)


def merage_by_label(dir, pdb_file, take_top=2000, add_pos=0, sort_rmsd=False, sort_soap=False):
    with open(pdb_file) as prot:
        pdbs = {prot.strip() for prot in prot}
    pos = None
    neg = None
    for i, file in enumerate(os.listdir(dir)):
        # print(i, end=" ")
        if file not in pdbs:
            # print(file, sep=" ")
            continue
        with open(osp.join(dir, file)) as trFile:
            tables = pd.read_csv(trFile, sep="\t", index_col=0)
            if sort_rmsd:
                tables = tables.sort_values(by="rec_rmsd")
            if sort_soap:
                tables = tables.sort_values(by="soap")
            top_scores = tables.head(take_top)
            extra_pos = tables[tables["labels"]].head(add_pos)
            top_scores_pos = top_scores[top_scores["labels"]]
            prot_pos = pd.concat([top_scores_pos, extra_pos])
            prot_neg = top_scores[~top_scores["labels"]]
            if pos is None:
                pos = prot_pos
                neg = prot_neg
            else:
                pos = pd.concat([pos, prot_pos])
                neg = pd.concat([neg, prot_neg])
        if i % 300 == 0:
            print("At step " + str(i) + " neg:" + str(len(neg)) + " pos: " + str(len(pos)))
    pos = pos.sample(frac=1)
    neg = neg.sample(frac=1)
    # neg=neg[:int(len(neg)/100)] ##under sample
    pos_size = len(pos)
    neg_size = len(neg)
    print("pos_size: ", pos_size, "neg_size: ", neg_size)
    print(pos["rec_rmsd"].mean())
    print(pos["rec_rmsd"].median())
    print(neg["rec_rmsd"].mean())
    print(neg["rec_rmsd"].median())
    return pos, neg


def write_train(workdir, outdir, trans_dir="", sort=False, suffix="_full", add_pos=0, sort_rmsd=False, sort_soap=False):
    transDir_cur = osp.join(workdir, trans_dir)
    pos, neg = merage_by_label(transDir_cur, osp.join(outdir, "train_pdbs"), add_pos=add_pos, sort_rmsd=sort_rmsd,
                               sort_soap=sort_soap)
    if sort:
        pos = pos.sort_values(by=["soap"])
        neg = neg.sort_values(by=["soap"], ascending=False)
        pos.to_csv(osp.join(outdir, "sorted_pos" + suffix), sep="\t", quoting=csv.QUOTE_NONE, quotechar="",
                   escapechar="\\", header=None)
        neg.to_csv(osp.join(outdir, "sorted_neg" + suffix), sep="\t", header=None, quoting=csv.QUOTE_NONE,
                   quotechar="", escapechar="\\")

    else:
        pos.to_csv(osp.join(outdir, "train_pos" + suffix), sep="\t", quoting=csv.QUOTE_NONE, quotechar="",
                   escapechar="\\", header=None)
        neg.to_csv(osp.join(outdir, "train_neg" + suffix), sep="\t", header=None, quoting=csv.QUOTE_NONE,
                   quotechar="", escapechar="\\")

    all = pd.concat([pos, neg])
    all.to_csv(osp.join(outdir, "train_all_trans" + suffix), sep="\t", header=None, quoting=csv.QUOTE_NONE,
               quotechar="", escapechar="\\")


def write_test(workdir, outdir, trans_dir="", suffix="", add_pos=5, sort_soap=False):
    transDir_cur = osp.join(workdir, trans_dir)
    pos, neg = merage_by_label(transDir_cur, osp.join(outdir, "test_pdbs"), add_pos=add_pos, sort_soap=sort_soap)
    # pos[0:0] = neg
    all = pd.concat([pos, neg])
    all = all.sample(frac=1)
    all.to_csv(osp.join(outdir, "test_trans" + suffix), header=None, sep="\t", quoting=csv.QUOTE_NONE, quotechar="",
               escapechar="\\")
    # rd.shuffle(pos)
    # writeFile(pos, osp.join(workdir, "test_trans"))
    # remove_missing_batches(pos,geo_matrix_dir, osp.join(workdir, "test_trans"))
    # with open(osp.join(workdir, "test_trans")) as data:
    #     lines = data.readlines()
    #     split_data(lines, workdir,"test")


def verifyDir(path):
    if not osp.isdir(path):
        os.mkdir(path)


def prepare_data(outdir, workdir="", trans_dir=transDir, add_pos_to_train=100, test_size=45, line_end=4, suffix_r="_u1",
                 suffix_l="_u2"):
    split_pdbs(workdir, outdir, trans_dir=trans_dir, test_size=test_size, line_end=line_end, suffix_r=suffix_r,
               suffix_l=suffix_l)
    write_train(workdir, outdir, trans_dir=trans_dir, sort=False, suffix="_AlphaFold", add_pos=add_pos_to_train,
                sort_rmsd=False, sort_soap=False)
    # write_test(workdir, outdir, trans_dir=trans_dir, suffix="_AlphaFold", sort_soap=False)


def scrape_data():
    verifyDir(transDir)
    with open(osp.join("dockground_fine", "dockground_trans_batched"), 'w') as writer:
        preduce_tran_dir()
        aggregate_by_batch(transDir, writer)


def get_chain_from_fasta(pdb_dir, out_folder):
    for pdb in os.listdir(pdb_dir):
        prot_name = pdb.split(".")[0]
        # comd = extractChains+" L "+ osp.join(pdb_dir,pdb)+" > "+osp.join(out_folder,prot_name+"_L.pdb")
        # os.system(comd)
        # comd = extractChains + " H " + osp.join(pdb_dir, pdb) + " > " + osp.join(out_folder, prot_name + "_H.pdb")
        # os.system(comd)
        # to_fast_comd=comndSeq +" "+osp.join(out_folder,prot_name+"_H.pdb") +" >> "+osp.join(out_folder,prot_name)+".seq"
        # os.system
        # os.chdir(NanoNet_location)
        # nanocomd=comdNanoNet+" "+osp.join(out_folder,prot_name)+".seq "+nanoNetSuffix
        # os.system(nanocomd)
        # align=alignComd+" "+osp.join(out_folder,prot_name+"_H.pdb")+" "+osp.join(out_folder,prot_name+"_nanonet_ca.rebuilt.pdb")
        # os.system(align)
        # name=nameComd+" "+ osp.join(out_folder,prot_name+"_nanonet_ca.rebuilt_tr.pdb") +" H"
        # os.system(name)
        ab_dir = osp.join("/cs/labs/dina/matanhalfon/CAPRI/ABDB/nanoAb", prot_name)
        verifyDir(ab_dir)
        # merageComd="cat "+osp.join(out_folder,prot_name+"_nanonet_ca.rebuilt_tr.pdb")+" "+osp.join(out_folder,prot_name+"_L")+" > "+osp.join(ab_dir,prot_name+"_Ab_nano.pdb")
        os.system("cp " + osp.join("/cs/labs/dina/matanhalfon/CAPRI/ABDB/splited_pdb", prot_name,
                                   prot_name + "_Ag.pdb") + " " + ab_dir)
        print(prot_name)


def get_chains(pdbFile):
    comd = getChains + " " + osp.join(pdbFile)
    chains = subprocess.check_output(comd, shell=True).decode("utf-8").strip().split(" ")
    AB_count = 0
    Antigen_chains = []
    for c in chains:
        if c == "H" or c == "L":
            AB_count += 1
        else:
            Antigen_chains.append(c)
    if AB_count != 2:
        print("XXXXXXXX not all AB chains are exists")
        raise IndexError
    if len(Antigen_chains) == 0:
        print("XXX there is no antigen chains")
    #     raise IndexError
    return Antigen_chains


def extract_chains(pdbFile, antigenChains, workdir=""):
    # ab_name=osp.basename(pdbFile).split(".")[0]
    # veifyDir(osp.join(workdir,ab_name))
    ab_filename = osp.basename(pdbFile).split(".")[0] + "_Ab.pdb"
    ag_filename = osp.basename(pdbFile).split(".")[0] + "_Ag.pdb"
    # comdAb = extractChains+ " " +"LH "+ osp.join(pdbFile)+" > "+osp.join(workdir,b_filename)
    # comdAg = extractChains+ " " +"LH "+ osp.join(pdbFile)+" > "+osp.join(workdir,d_filename)
    comdAb = extractChains + " " + "LH " + osp.join(pdbFile) + " > " + osp.join(workdir, ab_filename)
    comdAg = extractChains + " " + "".join(antigenChains) + " " + osp.join(pdbFile) + " > " + osp.join(workdir,
                                                                                                       ag_filename)
    print(comdAb)
    print(comdAg)
    os.system(comdAb)
    os.system(comdAg)


def split_all_antibody(complexDir, outDir):
    for prot in os.listdir(complexDir):
        if prot.endswith(".pdb"):
            pdbFile = osp.join(complexDir, prot)
            # chains=get_chains(pdbFile)
            extract_chains(pdbFile, chains, workdir=outDir)


def get_seq_from_fasta(path):
    seq = ""
    for rec in SeqIO.parse(path, format="fasta"):
        seq += str(rec.seq)
    return seq


def remove_seq_identity(test_pdbs, trains_pdbs, seq_dir, th=0.94, suffix_l="_Ab.seq", suffix_r="_Ag.seq"):
    new_train = []
    i = 0
    for prot_train in trains_pdbs:
        train_prot_seq_ab = get_seq_from_fasta(osp.join(seq_dir, prot_train + suffix_l))
        flag = True
        for prot_test in test_pdbs:
            test_prot_seq_ab = get_seq_from_fasta(osp.join(seq_dir, prot_test + suffix_l))
            ab_identity = pairwise2.align.globalxx(train_prot_seq_ab, test_prot_seq_ab, score_only=True) / min(
                len(test_prot_seq_ab), len(train_prot_seq_ab))
            if ab_identity > th:
                flag = False
                print(prot_train)
                i += 1
                break
        if flag:
            new_train.append(prot_train)
    print(i)
    return new_train


def merage_df(dir, out_name, pdb_file=""):
    if pdb_file:
        with open(pdb_file) as prots:
            pdbs = {prot.strip() for prot in prots}
    table = None
    added = 1
    for i, file in enumerate(os.listdir(dir)):
        if not pdb_file or file in pdbs:
            added += 1
            print(file, i, added)
            df = pd.read_csv(osp.join(dir, file), sep="\t", index_col=0)
            if table is None:
                table = df
            else:
                table = pd.concat([table, df])
            if added % 1000 == 0:
                table.to_csv(out_name + "_part_" + str(added // 1000), sep="\t", quoting=csv.QUOTE_NONE, quotechar="",
                             escapechar="\\", header=None)
                table = None

    table.to_csv(out_name + "_part_" + str((added // 1000) + 1), sep="\t", quoting=csv.QUOTE_NONE, quotechar="",
                 escapechar="\\", header=None)
    # table.to_csv(out_name, sep="\t", quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\",header=None)


def create_test_dir(train_dir, test_dir, take_top=6000):
    for file in os.listdir(train_dir):
        df = pd.read_csv(osp.join(train_dir, file), sep="\t", index_col=0)
        sorted = df.sort_values("soap").head(take_top)
        sorted.to_csv(osp.join(test_dir, file), sep="\t", quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
        print(file)


def create_target_dirs(dir, Ag_file, docking_dir):
    for i in range(5):
        # os.system("cp " + Ag_file + "_Ag.pdb " + osp.join(docking_dir, dir + "_model_" + str(i),
        #                                                   dir + "_model_" + str(i) + "_Ag.pdb"))
        # os.system("cp " + Ag_file + "_Ab.pdb " + osp.join(docking_dir, dir + "_model_" + str(i),
        #                                                   Ag_file + "_Ab.pdb"))

        # if not osp.exists(osp.join(docking_dir, dir)+"_model_"+str(i)):
        #     print(osp.join(docking_dir, dir)+"_model_"+str(i))
            os.system("mkdir " + osp.join(docking_dir, dir)+"_model_"+str(i))
        #     os.system("cp "+ Ag_file+"_Ag.pdb " +osp.join(docking_dir, dir+"_model_"+str(i),Ag_file+"_model_"+str(i)+"_Ag.pdb"))
            os.system("cp "+ Ag_file+"_Ab.pdb " +osp.join(docking_dir, dir+"_model_"+str(i),osp.basename(Ag_file)+"_Ab.pdb"))


def create_model_dir_name(Alphafold_dir, dir, i):
    prot = dir[:6]
    return osp.join(Alphafold_dir, dir + "_Ab",
                    "ABDB_splited_pdb_" + prot + "_" + dir + "_Ab.pdb" + "_chain_L_unrelaxed_rank_*" + str(
                        i + 1) + ".pdb")


def copy_Alphafold_models(Alphafold_dir, dock_dir, dir, unfolded):
    for i in range(5):
        file_name = create_model_dir_name(Alphafold_dir, dir, i)
        if (not glob.glob(file_name)):
            unfolded.add(dir)
        os.system(
            "cp " + create_model_dir_name(Alphafold_dir, dir, i) + " " + osp.join(dock_dir, dir + "_model_" + str(i),
                                                                                  dir + "_model_" + str(i) + "_Ab.pdb"))


def split_modeled(Alphafold_dir, out_dir,pdbs,suffix_r="_Ab.pdb",suffix_l="_Ag.pdb"):
    for dir in os.listdir(Alphafold_dir):
        if dir in pdbs:
            for file in os.listdir(osp.join(Alphafold_dir, dir)):
                # if not check_for_pdb(osp.join(target_dir)):
                try:
                    if file.endswith(".pdb"):
                        # target_dir = osp.join(out_dir, dir+"_model_" +str(int(file[-5])-1))
                        target_dir = osp.join(out_dir, dir)
                        os.system("mkdir " + target_dir)
                        print(file)
                        comd_ab = get_frag_chain + " " + osp.join(Alphafold_dir, dir, file) + " BC >" + osp.join(
                            target_dir , dir + "_model_" +str(int(file[-5])-1) + suffix_r)
                        os.system(comd_ab)
                        comd_ag = get_frag_chain + " " + osp.join(Alphafold_dir, dir,file) + " DEFGHIJKLMNOPQRSTUVWXYZ >" + osp.join(
                            target_dir,dir + "_model_" +str(int(file[-5]) - 1) + suffix_l)
                        os.system(comd_ag)
                except ValueError:
                    print("XXXXXXXXXXX ",file)

def merage_AlphaFold_dock_results(workdir, prot):
    meraged = pd.DataFrame(columns=["rec_rmsd", "lig_rmsd", "dockQ", "labels"])
    for i in range(5):
        file = osp.join(workdir, prot, "rmsd_" + prot + "_model_" + str(i) + ".res")
        df = pd.read_csv(file, skipfooter=2)
        rec_rmsds, lig_rmsds, labels, dockQ = extract_from_rms(df)
        meraged = meraged.append(
            {"rec_rmsd": rec_rmsds[0], "lig_rmsd": lig_rmsds[0], "dockQ": dockQ[0], "labels": labels[0]},
            ignore_index=True)
    meraged.to_csv(osp.join(workdir, prot, "results.csv"), sep="\t")
    return meraged


def get_dock_file(workdir, prot,suffix_r="_Ab.pdb",suffix_l="_Ag.pdb"):
    os.chdir(osp.join(workdir, prot))
    try:
        for i in range(5):
            # comd = rmsd_alighn + "-c  " + prot + suffix_l+" " + prot + "_model_" + str(i) + suffix_l+" " + prot + \
            #        suffix_r+" " + prot + "_model_" + str(i) + suffix_r+" trans -o " + "rmsd_" + prot + "_model_" + str(
            #     i) + ".res"
            comd = rmsd_alighn + "-c  " + prot  + suffix_l + " " + prot + "_model_" + str(i) + suffix_l + " " + prot + \
                   suffix_r + " " + prot + "_model_" + str(
                i) + suffix_r + " trans -o " + "rmsd_" + prot + "_model_" + str(
                i) + ".res"
            print(comd)
            subprocess.check_call(comd, shell=True)
    except subprocess.CalledProcessError:
        # with open("/cs/labs/dina/matanhalfon/CAPRI/bad_folded_AlphaFold_db5", "a") as file:
        #     file.write("failed at " + prot + '\n')
            print(prot)
    os.chdir("../../..")


def create_Alphafold_dock_dirs(Ag_dir, Alphafold_dir, docking_dir,test_pdbs):
    unfolded = set()
    for dir in os.listdir(Ag_dir):
        if (not test_pdbs) or dir in test_pdbs:
            if osp.isdir(osp.join(Ag_dir, dir)):
                # print(dir)
                # create_target_dirs(dir,osp.join(Ag_dir,dir,dir),docking_dir)
                copy_Alphafold_models(Alphafold_dir, docking_dir, dir, unfolded)
    print(unfolded)


def aggreagate_evaluations(pdbs_test, evaluation_dir, struct_dir, out_dir, eval_file="evaluation_dockQ_3_",
                           suffix_l="_Ag.pdb", suffix_r="_Ab.pdb",
                           line_len=14):
    for pdb in pdbs_test:
        print(pdb)
        # os.system("mkdir " + osp.join(out_dir, pdb[:6]))
        os.system("cp " + osp.join(struct_dir, pdb, pdb + "_Ab.pdb") + " " + osp.join(out_dir, pdb[:6]))
        os.system("cp " + osp.join(struct_dir, pdb, pdb + "_Ag.pdb") + " " + osp.join(out_dir, pdb[:6]))
        os.system("cp " + osp.join(evaluation_dir, "midfiles",pdb, eval_file + pdb) + " " + osp.join(out_dir, pdb[:6],
                                                                                          eval_file + pdb))


def write_clustring_comd(results_dir, dir, cluster_size, file_type="evaluation_", reverse=True):
    comd = clustring + osp.join(results_dir, dir, dir + "_Ag.pdb")
    for i in range(5):
        if osp.isfile(osp.join(results_dir, dir, dir + "_model_" + str(i) + "_Ab.pdb")):
            comd += " " + osp.join(results_dir, dir, dir + "_model_" + str(i) + "_Ab.pdb") + " " + \
                    osp.join(results_dir, dir, file_type + dir + "_model_" + str(i))
            # osp.join(results_dir, dir, "soap_results_" + dir + "_model_" + str(i))
    comd += " " + str(cluster_size)
    if reverse:
        comd += " -s "
    comd += " -o " + osp.join(results_dir, dir, dir + "_" + file_type + "_clusterd_" + str(cluster_size) + ".results")
    return comd


def cluster_results(results_dir, test_pdbs,eval_file="evaluation_dockQ_3_", cluster_size=4):
    # pdbs=["5VL3_4"]
    for dir in test_pdbs:
        comd = write_clustring_comd(results_dir, dir, cluster_size, file_type=eval_file,reverse=False)
        os.system(comd)
        # comd_soap=write_clustring_comd(results_dir,dir,cluster_size,file_type="soap_results_",reverse=False)
        # os.system(comd_soap)


def get_seq_aa_from_chain_object(chain):
    aa_residues = []
    seq = ""
    for residue in chain:
        aa = residue.get_resname()
        # if is_aa(aa) and not residue.has_id('CA'):
        #
        if not is_aa(aa) or not residue.has_id('CA'):
            continue
        elif aa in UNIQE_AA:
            seq += UNIQE_AA[aa]
        else:
            seq += Polypeptide.three_to_one(residue.get_resname())
        aa_residues.append(residue)
    return seq, aa_residues


def get_seq_aa(pdb, chain_letter):
    try:
        chain = PDBParser(QUIET=True).get_structure(pdb, pdb)[0][chain_letter]
    except KeyError:
        chain = PDBParser(QUIET=True).get_structure(pdb, pdb)[0][" "]
    return get_seq_aa_from_chain_object(chain)

def create_pdb_dir(dock_dir,pdb_dir,suffix_r,suffix_l):
    for dir in os.listdir(dock_dir):
        if osp.isdir(osp.join(dock_dir,dir)):
            for file in os.listdir(osp.join(dock_dir,dir)):
                if file.endswith(suffix_r) or file.endswith(suffix_l):
                    os.system("cp "+osp.join(dock_dir,dir,file)+" "+pdb_dir)
                    os.system("cp "+osp.join(dock_dir,dir,file)+" "+pdb_dir)

def get_h_l_by_seq_alignment(pdb, chain_h, chain_l, out_dir, suffix):
    try:
        h_seq, h_aa = get_seq_aa(pdb, chain_h)
        l_seq, l_aa = get_seq_aa(pdb, chain_l)

        for ref_seq, get_func, chain, h_l_seq, aa in zip([HEAVY_SEQ, LIGHT_SEQ], [get_h_chain, get_l_chain], [chain_h, chain_l], [h_seq, l_seq], [h_aa, l_aa]):
            alignments = pairwise2.align.globalxd(ref_seq, h_l_seq, -3, 0, -0.1, 0, penalize_end_gaps=(False, True),
                                                  one_alignment_only=True)

            start, end = alignments[0].seqA.index(ref_seq[0]), alignments[0].seqA.rindex(ref_seq[-1])
            chain_seq = alignments[0].seqB[start:end + 1].replace("-", "")
            print("@Seq@", chain_seq)
            start_aa = h_l_seq.find(chain_seq)
            end_aa = start_aa + len(chain_seq) - 1
            pdb_start = str(aa[start_aa].get_id()[1]) + aa[start_aa].get_id()[2]
            pdb_end = str(aa[end_aa].get_id()[1]) + aa[end_aa].get_id()[2]
            print(pdb_start, pdb_end)
            get_func(pdb, chain, out_dir, start=pdb_start, end=pdb_end)
        subprocess.run(f"cat {pdb.split('.')[0]}_heavy.pdb {pdb.split('.')[0]}_light.pdb > {out_dir}/{osp.basename(pdb)[:4]}{suffix}",
                       shell=True)
    except KeyError:
        print(f"key error at {pdb}")

# def get_h_l_by_seq_alignment(pdb, chain_h, chain_l, out_dir, suffix):
#     try:
#         h_seq, h_aa = get_seq_aa(pdb, chain_h)
#         l_seq, l_aa = get_seq_aa(pdb, chain_l)
#
#         h_l_seq = h_seq + l_seq
#         aa = h_aa + l_aa
#
#         for ref_seq, get_func, chain in zip([HEAVY_SEQ, LIGHT_SEQ], [get_h_chain, get_l_chain], [chain_h, chain_l]):
#             alignments = pairwise2.align.globalxd(ref_seq, h_l_seq, -3, 0, -0.1, 0, penalize_end_gaps=(False, True),
#                                                   one_alignment_only=True)
#
#             start, end = alignments[0].seqA.index(ref_seq[0]), alignments[0].seqA.rindex(ref_seq[-1])
#             chain_seq = alignments[0].seqB[start:end + 1].replace("-", "")
#             print("@Seq@", chain_seq)
#             start_aa = h_l_seq.find(chain_seq)
#             end_aa = start_aa + len(chain_seq) - 1
#             pdb_start = str(aa[start_aa].get_id()[1]) + aa[start_aa].get_id()[2]
#             pdb_end = str(aa[end_aa].get_id()[1]) + aa[end_aa].get_id()[2]
#             get_func(pdb, chain, out_dir, start=pdb_start, end=pdb_end)
#         subprocess.run(f"cat {pdb.split('.')[0]}_heavy.pdb {pdb.split('.')[0]}_light.pdb > {out_dir}/{osp.basename(pdb)[:4]},_{suffix}",
#                        shell=True)
#     except KeyError:
#         print(f"key error at {pdb}")


# def copy_files_to_target(test_pdbs,data_dir="ABDB/AlphaFold_splited_modeled",dest="ABDB/AlphaFold_splited_modeled"):

def get_AlphaFold_results(data_dir="ABDB/AlphaFold_splited_modeled",dest="ABDB/AlphaFold_splited_modeled",
    workdir="ABDB/AlphaFold_data_dir/AlphaFold_data",suffix_r="_Ab.pdb",suffix_l="_Ag.pdb"):
    # pdbs_test = get_pdb_from_file(osp.join(workdir, "test_pdbs"),line_len=6)
    pdbs_test={"1IQD","2W9E","1KXQ"}
    names=[]
    results=[]
    for i,file in enumerate(pdbs_test):
        print(file)
        os.system("cp " + osp.join(data_dir, file,file+suffix_r) + " " + osp.join(dest,file))
        os.system("cp " + osp.join(data_dir, file,file+suffix_l) + " " + osp.join(dest,file))
        with open(osp.join(dest,file,"trans"),"w+") as t_file:
            t_file.write("1 0.0 0.0 0.0 0.0 0.0 0.0")
        if not osp.isdir(osp.join(data_dir, file)):
            continue
        get_dock_file(data_dir, file,suffix_l=suffix_l,suffix_r=suffix_r)
        try:
            prot_res=merage_AlphaFold_dock_results(data_dir,file)
        except FileNotFoundError :
            print("not res in "+file)
            names.append(file)
            results.append(False)
            continue
        res=prot_res["labels"].any()
        names.append(file)
        results.append(res)
        # print(file+" "+str(res))
    print(len(names))
    print(sum(results))
    res_dict={"names":names,"results":results}
    df=pd.DataFrame(res_dict)
    df.to_csv("AlphaFold_results_db5.csv",sep="\t")

def get_AlphaFold_transformation(pdb_test,workdir="ABDB/AlphaFold_splited_modeled"):
    index=[]
    prots=[]
    soap=[]
    trans=[]
    rec_rmsd=[]
    lig_rmsd=[]
    dockQ=[]
    labels=[]
    test_prots = []
    for i,dir in enumerate(os.listdir(workdir)):
        if dir in pdbs_test:
            index_cur = []
            prots_cur = []
            soap_cur = []
            trans_cur = []
            rec_rmsd_cur = []
            lig_rmsd_cur = []
            dockQ_cur = []
            labels_cur = []
            for file in os.listdir(osp.join(workdir,dir)):
                res=pd.read_csv(osp.join(workdir,dir,"results.csv"),sep="\t")
                if file.endswith("Ab.pdb") and "model" in file:
                        print(file)
                        index_cur.append("1111")
                        prots_cur.append(dir+"_model_" +str(int(file[-8])-1)+"_Ab.pdb "+dir+"_model_" +str(int(file[-8])-1)+"_Ag.pdb")
                        soap_cur.append(6666)
                        trans_cur.append("0.0 0.0 0.0 0.0 0.0 0.0")
                        rec_rmsd_cur.append(res["rec_rmsd"][int(file[-8])-1])
                        lig_rmsd_cur.append(res["lig_rmsd"][int(file[-8])-1])
                        dockQ_cur.append(res["dockQ"][int(file[-8])-1])
                        labels_cur.append(res["labels"][int(file[-8])-1])
                        test_prots.append(dir+"_model_" +str(int(file[-8])-1))
            df_cur = pd.DataFrame({"index": index_cur, "names": prots_cur, "soap": soap_cur, "trans": trans_cur,
                               "rec_rmsd": rec_rmsd_cur,"lig_rmsd": lig_rmsd_cur, "dockQ": dockQ_cur, "labels": labels_cur})
            verifyDir(osp.join("ABDB","AlphaFold_docking_trans"))
            df_cur.to_csv("ABDB/AlphaFold_docking_trans/"+dir, index=False, sep="\t")
            index +=index_cur
            prots +=prots_cur
            soap +=soap_cur
            trans +=trans_cur
            rec_rmsd +=rec_rmsd_cur
            lig_rmsd +=lig_rmsd_cur
            dockQ +=dockQ_cur
            labels +=labels_cur
    df=pd.DataFrame({"index":index,"names":prots,"soap":soap,"trans":trans,"rec_rmsd":rec_rmsd,"lig_rmsd":lig_rmsd,"dockQ":dockQ,"labels":labels})
    df.to_csv("ABDB/AlphaFold_splited_modeled/AlphaFold_trans",header=False,index=False,sep="\t")

def merage_AlphaFold_patchDock_results(alphaFold_res_dir,patchdock_res_dir,save_dir,test_proteins,model_suffix_patchDock,model_suffix_alphaFold,cluster):
    pdbs_test = get_pdb_from_file(test_proteins,line_len=6)
    for prot in pdbs_test:
        patchdock_cluster_eval=pd.read_csv(osp.join(patchdock_res_dir,prot,prot+"_evaluation"+model_suffix_patchDock+"_clusterd_"+str(cluster)+".results"),sep="|", header=2)[['score', 'lrmsd', 'irmsd', 'dockQ', 'transformation']]
        if patchdock_cluster_eval[patchdock_cluster_eval["transformation"]=="0.0 0.0 0.0 0.0 0.0 0.0"].shape[0]>0:
            print(colored("patchdock preduced 0 transform",'red'))
        alphaFold_eval=pd.read_csv(osp.join(alphaFold_res_dir,prot,"evaluation"+model_suffix_alphaFold+prot),sep="\t")[["score","rec_rmsd","lig_rmsd","dockQ","trans"]]
        alphaFold_eval=alphaFold_eval.rename(columns={"rec_rmsd":"lrmsd","lig_rmsd":"irmsd","trans":"transformation"})
        combined=pd.concat([patchdock_cluster_eval,alphaFold_eval])
        combined.sort_values("score")
        combined=combined.reset_index(drop=True)
        alphafold_hit=combined[combined["transformation"]=="0.0 0.0 0.0 0.0 0.0 0.0"].index[0]
        print(alphafold_hit)
        alphaFold_eval.to_csv(osp.join(save_dir,prot+model_suffix_patchDock+"_clusterd_"+str(cluster)+"_consensus"))

def get_h_chain(pdb, heavy, out_dir, start=0, end=113, ):
    subprocess.run(f"{get_frag_chain} {pdb} {heavy} {start} {end} > {pdb.split('.')[0]}_heavy.pdb", shell=True)
    subprocess.run(f"{get_name_chain} heavy.pdb H", shell=True)
    # create_fasta_from_chain("heavy.pdb", "H")


def get_l_chain(pdb, light, out_dir, start=0, end=109):
    subprocess.run(f"{get_frag_chain} {pdb} {light} {start} {end} > {pdb.split('.')[0]}_light.pdb", shell=True)
    subprocess.run(f"{get_name_chain} light.pdb L", shell=True)
    # create_fasta_from_chain("light.pdb", "L")

def extract_trans_col(dir,model_suffix,test_pdbs):
    pdbs=get_pdb_from_file(test_pdbs)
    for prot in pdbs:
        df=pd.read_csv(osp.join(dir,prot[:6],model_suffix+prot),sep="\t")
        trans=df["transformation"][:15]
        trans.to_csv(osp.join(dir,prot[:6],model_suffix+prot+".trans" ),sep="\t",index=False,header=False)



def parse_interface_line(line,scores,line_number=0):
    if not line_number:
        line_number = line[0]
    line = line[2:].split("=")
    metric = line[0]
    score = float(line[1].split(" ")[1])
    if line_number not in scores.keys():
        scores[line_number] = dict()
    scores[line_number][metric] = score


def get_interface_lable(lines,line_number=0):
    scores = dict()
    for line in lines:
        if line.startswith("ASA"):
            continue
        parse_interface_line(line,scores,line_number=line_number)

    recall=[scores[line]["Recall"] for line in scores.keys()]
    precision=[ scores[line]["Precision"] for line in scores.keys()]
    labels = [1 if (scores[line]["Recall"] >= 0.5 and scores[line]["Precision"] >= 0.5) else 0 for line
              in scores.keys()]
    return labels,recall,precision

def get_interface_results_alphaFold(dir,test_pdbs):
    for prot in os.listdir(dir):
        try:
            if prot in test_pdbs:
                lines=[]
                labels=[]
                recalls=[]
                precisions=[]
                print(prot)
                workdir=osp.join(dir,prot)
                for i in range(1,6):
                    comd=rmsd_alighn+"-i "+osp.join(workdir,prot+"_Ag.pdb")+" "+osp.join(workdir,prot+"_model_"+str(i)+"_Ag.pdb")\
                         +" "+osp.join(workdir,prot+"_Ab.pdb")+" "+osp.join(workdir,prot+"_model_"+str(i)+"_Ab.pdb")+" "+osp.join(workdir,"trans")
                    model_lines = subprocess.run(comd, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,universal_newlines=True).stderr.split("\n")[7:-1]
                    lable,recall,precition=get_interface_lable(model_lines,i)
                    labels.append(lable)
                    recalls.append(recall)
                    precisions.append(precition)
                if labels:
                    df = pd.read_csv(osp.join(workdir,"results.csv"), sep="\t")
                    df["interface_label"] = labels
                    df["Recall"] = recalls
                    df["precision"] = precisions

                    df.to_csv(osp.join(workdir,prot+".interface"),sep="\t",index=False)
                else:
                    print("error at "+prot)
                    print(lines)
        except FileNotFoundError:
            print("missing somthing  at "+prot)

def get_interface_results_alphaFold(dir,test_pdbs,folding_foldr="ABDB/meraged",line_len=6,suffix_r="_Ab.pdb",suffix_l="_Ag.pdb"):
    for prot in test_pdbs:
        labels,recalls,precisions=[],[],[]
        workdir = osp.join(dir, prot[:line_len])
        trans_file=osp.join(workdir,"trans" )
        for model in range(5):
            try:
                print(prot,model)
                comd=rmsd_alighn+"-i "+osp.join(workdir,prot[:line_len]+suffix_l)+" "+osp.join(workdir,prot[:line_len]+"_model_"+str(model)+suffix_l) \
                     +" "+osp.join(workdir,prot[:line_len]+suffix_r)+" "+osp.join(workdir,prot[:line_len]+"_model_"+str(model)+suffix_r)+" "+trans_file
                lines = subprocess.run(comd, shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,universal_newlines=True).stderr.split("\n")[7:-1]
                label, recall, precision = get_interface_lable(lines)
                labels.append(label[0])
                recalls.append(recall[0])
                precisions.append(precision[0])
            except IndexError:
                print("no lines "+prot[:line_len])
                continue
        if labels:
            df=Alphafold_score(osp.join(folding_foldr,prot[:line_len],"log.txt"), prot[:line_len], dir,suffix_r=suffix_r,suffix_l=suffix_l)
            d={"interface_label":labels, "recall_interface":recalls, "precision_interface":precisions}
            interface_df = pd.DataFrame(d)
            rmsd = pd.read_csv(osp.join(workdir, "results.csv"),sep="\t",index_col=0)
            meraged = pd.concat([df, interface_df, rmsd], axis=1)
            meraged.to_csv(osp.join(workdir,prot[:6]+"_interface.csv"),sep="\t")
        else:
            print("error at "+prot)
            print(lines)

def get_interface_results(dir,model_suffix,test_pdbs):
    for prot in test_pdbs:
            print(prot)
            workdir=osp.join(dir,prot[:6])
            trans_file=osp.join(workdir,model_suffix+prot+".trans" )

            comd=rmsd_alighn+"-i "+osp.join(workdir,prot[:6]+"_Ag.pdb")+" "+osp.join(workdir,prot[:6]+"_Ag.pdb")\
                 +" "+osp.join(workdir,prot[:6]+"_Ab.pdb")+" "+osp.join(workdir,prot+"_Ab.pdb")+" "+trans_file
            lines = subprocess.run(comd, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,universal_newlines=True).stderr.split("\n")[7:-1]
            labels, recall, precision = get_interface_lable(lines)
            if labels:
                df = pd.read_csv(osp.join(workdir,model_suffix+prot), sep="\t",index_col=0).head(10)
                df.insert(6,"interface_label",labels)
                df.insert(7,"recall_interface",recall)
                df.insert(8,"precision_interface",precision)
                df=df.rename(columns={"soap":"score"})
                columns = list(df.columns)
                orig_index=columns.index("transformation")
                orig = columns[orig_index]
                columns = columns[:orig_index] + columns[orig_index + 1:]
                columns.append( orig)
                fixed = df[columns]
                fixed.to_csv(osp.join(workdir,model_suffix+"interface_"+prot),sep="\t")
            else:
                print("error at "+prot)
                print(lines)

def get_variable_Ab_variable(workdir, prots_path, base_suffix, out_suffix, heavy="H", light="L"):
    # pdbs = get_pdb_from_file(osp.join(workdir, prots_path))"2W9E","1KXQ",
    pdbs={"1IQD","2W9E","1KXQ"}
    for dir in os.listdir(workdir):
        if dir in pdbs:
            for file in os.listdir(osp.join(workdir, dir)):
                if file.endswith(base_suffix):
                    try:
                        get_h_l_by_seq_alignment(osp.join(workdir, dir, file), heavy, light, osp.join(workdir, dir),
                                                 suffix=out_suffix)
                    except:
                        print(f"crashed at {file}")

def fold_complexes(source_dir,target_dir,pdbs):
    # get_comp_seq("/cs/labs/dina/matanhalfon/CAPRI/ABDB/splited_pdb", "/cs/labs/dina/matanhalfon/CAPRI/ABDB/meraged")
    # fold_fasta("/cs/labs/dina/matanhalfon/CAPRI/ABDB/meraged")
    # get_comp_seq(source_dir, target_dir,pdbs,suffix_ag="_l_u.pdb")
    fold_fasta(target_dir,pdbs)

def fix_interface_file(path,index):
    with open(path) as f:
        lines = f.readlines()
        splited = lines[2].split("|")
        splited.insert(index, "dockQ")
        line2 = "|".join(splited)
        lines[2]= line2
    with open(path, "w") as f:
        for line in lines:
            f.write(line)


def reorder_columns(workdir, suffix, test_pdbs,orig_index,new_index):
    for prot in test_pdbs:
        df=pd.read_csv(osp.join(workdir, prot[:6],suffix + prot), sep = "\t",index_col=0)
        columns=list(df.columns)
        orig=columns[orig_index]
        columns=columns[:orig_index]+columns[orig_index+1:]
        columns.insert(new_index,orig)
        fixed=df[columns]
        fixed.to_csv(osp.join(workdir, prot[:6],suffix + prot), sep = "\t")

def insert_col_name(workdir, suffix, test_pdbs,index,col_name,cluster_range):
    for prot in test_pdbs:
        for c in cluster_range:
            df=pd.read_csv(osp.join(workdir, prot[:6],prot[:6]+suffix+str(c)+".results" ), sep = "|",header=2)
            columns=list(df.columns)
            columns.insert(index,col_name)
            fixed=df[columns]
            fixed.to_csv(osp.join(workdir, prot[:6],suffix + prot), sep = "\t")


def cluster_eval(dir_eval,test_pdbs,eval_file):
    for i in range(2, 9):
        cluster_results(dir_eval,
                        test_pdbs
                        , cluster_size=i, eval_file=eval_file)

def create_interface_files(workdir,suffix,test_pdbs):
    # extract_trans_col(workdir, suffix, osp.join("ABDB/AlphaFold_data_dir/AlphaFold_data", "test_pdbs"))
    test_pdb=osp.join("ABDB/AlphaFold_data_dir/AlphaFold_data", "test_pdbs")
    pdbs=get_pdb_from_file(test_pdb)
    get_interface_results(workdir, suffix,pdbs )
    reorder_columns(workdir,suffix+ "interface_",pdbs,5,9)
    cluster_eval(workdir,test_pdbs,suffix+ "interface_")
    # for i in range(2, 6):
    #     cluster_results(workdir,
    #                     test_pdbs,cluster_size=i, eval_file=suffix+ "interface_")

def delete_sub_folders(dir):
    print(dir)
    os.chdir(dir)
    for sub_dir in os.listdir(dir):
        os.chdir(sub_dir)
        print(pathlib.Path.cwd())
        subprocess.run("find * -type f -delete", shell=True)
        os.chdir("..")
        print(pathlib.Path.cwd())


def insert_dockQ_column(net_suffix = "_evaluation_Ha_101_test"):
    test_prot="ABDB/AlphaFold_data_dir/AlphaFold_data"
    dir_all="ABDB/AlphaFold_all_results"
    test_pdbs = get_pdb_from_file(osp.join(test_prot, "test_pdbs"),line_len=6)
    seen=set()
    # print("XXXfixing clustersXXX")
    for prot in test_pdbs:
        if prot not in seen:
            print("prot: "+prot)
            for cluster in range(2, 9):
                path = osp.join(dir_all, prot[:6], prot[:6] + "_"+net_suffix + "__clusterd_" +
                                str(cluster) + ".results")
                fix_interface_file(path,7)


def aggreate_results_from_models(suffix,pdbs_test, evaluation_dir, struct_dir, out_dir,line_len=6):
    pdbs_test = get_pdb_from_file(pdbs_test, line_len=line_len)
    aggreagate_evaluations(pdbs_test,evaluation_dir,struct_dir,out_dir,eval_file="evaluation_"+suffix+"_")
    cluster_eval(out_dir, pdbs_test, "evaluation_"+suffix+"_")
    insert_dockQ_column(net_suffix="evaluation_"+suffix+"_")
