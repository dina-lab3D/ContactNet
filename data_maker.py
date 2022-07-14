import sys
import os.path as osp
import os
import argparse
import subprocess

#script_intro="#!/bin/csh \n#SBATCH --mem=20g \n#SBATCH -c6 \n#SBATCH --time=18:0:0\
#\n#SBATCH --gres=gpu:1,vmem:14g\
#\nmodule load cuda/11.3\
#\nmodule load cudnn/8.2.1\
#\nmodule load nccl\
#\nsource  /cs/labs/dina/matanhalfon/mypython2/bin/activate.csh\
#\ncd /cs/labs/dina/matanhalfon/CAPRI\
#\npython3 model_evaluate.py NNscripts/lr-0.003_5_train_transformer_ABDBADAMW_Alphfold_transformer_5_4_heads_4_5_70_44_5e4_lr_kernal_1_3_3_wd_5e-3_sample4_slower_decay/mymodel_106 "
#patch_extractor="python3 ../../get_patches.py "

def verify_dir(path):
    if not osp.isdir(path):
        os.mkdir(path)

def pdb_to_dssp(exe_dir, pdb_file):
    dssp_name = "dssp/" + pdb_file.split(".pdb")[0] + ".dssp"
    cmd = exe_dir + "/dssp/mkdssps " + pdb_file
    os.system(cmd)
    dssp_out = pdb_file + ".dssp";
    os.rename(dssp_out, dssp_name)

def pdb_to_self_distogram(exe_dir, pdb_file):
    cmd = exe_dir + "/src/SelfDistogram/self_distogram_maker " + pdb_file
    os.system(cmd)
    out_file_name = pdb_file.split(".pdb")[0] + "_self_distogram.npy"
    new_file_name =  pdb_file.split(".pdb")[0] + "_self_distogram.npy"
    os.rename(out_file_name, new_file_name)


def create_predata(exe_dir, antigen_pdb, antibody_pdb):
    pdb_to_dssp(exe_dir, antigen_pdb)
    pdb_to_dssp(exe_dir, antibody_pdb)

    pdb_to_self_distogram(exe_dir, antigen_pdb)
    pdb_to_self_distogram(exe_dir, antibody_pdb)

def create_input_data(exe_dir, antigen_pdb, antibody_pdb, trans_file, trans_num):
    # generate distograms
    verify_dir("PPI")
    os.chdir("PPI");
    cmd = exe_dir + "/src/ComplexDistogram/complex_distogram_maker ../" + antigen_pdb\
        + " ../" + antibody_pdb + " ../" + trans_file + " " + str(trans_num)
    print(cmd)
    os.system(cmd)
#    os.chdir("..")

    # generate patches
#    verify_dir("patch_data")
#    os.chdir("patch_data");

    patch_cmd= "python3 " + exe_dir + "/get_patches.py " + " distograms.txt"
    #args.prot_dir+" trans"+args.suffix_Ab+" "+osp.join(args.prot_dir, "PPI")+" "+osp.join(args.prot_dir, "patch_data")
    print(patch_cmd)
    subprocess.check_output(patch_cmd, shell=True)
    os.system(patch_cmd)
    os.chdir("..")

def main():

    exe_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    curr_dir = os.getcwd()

    parser=argparse.ArgumentParser(description="parser for data preparation for single protein")
    parser.add_argument('antigen_pdb',type=str,help="Antigen PDB file name")
    parser.add_argument('antibody_pdb',type=str,help="Antibody PDB file name")
    parser.add_argument('trans_file',type=str,default="soap_score.res",help="transformation file name")
    parser.add_argument('--trans_num',type=int,default=0,help="# of transform to read from a transformation file, default all")
    args=parser.parse_args()

    if len(sys.argv) < 4:
        print("Usage: python3 data_maker.py <antigen_pdb> <antibody_pdb> <trans_file> [trans_num]")
        return 1

    create_predata(exe_dir, args.antigen_pdb, args.antibody_pdb)
    create_input_data(exe_dir, args.antigen_pdb, args.antibody_pdb, args.trans_file, args.trans_num)



if __name__ == '__main__':
    main()
