import sys
import os.path as osp
import os
import argparse
import subprocess


def verify_dir(path):
    if not osp.isdir(path):
        os.mkdir(path)

def pdb_to_dssp(exe_dir, pdb_file):
    dssp_name = pdb_file.split(".pdb")[0] + ".dssp"
    cmd = exe_dir + "/dssp/mkdssps " + pdb_file
    os.system(cmd)
    dssp_out = pdb_file + ".dssp";
    os.rename(dssp_out, dssp_name)

def create_predata(exe_dir, antigen_pdb, antibody_pdb):
    pdb_to_dssp(exe_dir, antigen_pdb)
    pdb_to_dssp(exe_dir, antibody_pdb)



def create_input_data(exe_dir, antigen_pdb, antibody_pdb, trans_file, trans_num):
    # generate distograms
    verify_dir("PPI")
    os.chdir("PPI")
    cmd = exe_dir + "/src/ComplexDistogram/complex_distogram_maker ../" + antigen_pdb\
        + " ../" + antibody_pdb + " ../" + trans_file + " " + str(trans_num)
    print(cmd)
    os.system(cmd)

    patch_cmd= "python3 " + exe_dir + "/get_patches.py " + " distograms.txt"
    print(os.system("pwd"))
    print(patch_cmd)
    subprocess.check_output(patch_cmd, shell=True)
    os.system(patch_cmd)

    for n in range(1, trans_num+1):
        dfile = antibody_pdb + "X" + antigen_pdb + "transform_number_" + str(n)
        if os.path.exists(dfile):
            os.remove(dfile)
    os.chdir("..")

def main():

    exe_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    curr_dir = os.getcwd()

    parser=argparse.ArgumentParser(description="parser for data preparation for single protein")
    parser.add_argument('antigen_pdb',type=str,help="Antigen PDB file name")
    parser.add_argument('antibody_pdb',type=str,help="Antibody PDB file name")
    parser.add_argument('trans_file',type=str,default="soap_score.res",help="transformation file name")
    parser.add_argument('--trans_num',type=int,default=6000,help="# of transform to read from a transformation file, default all (6000)")
    args=parser.parse_args()

    if len(sys.argv) < 4:
        print("Usage: python3 data_maker.py <antigen_pdb> <antibody_pdb> <trans_file> [trans_num]")
        return 1

    create_predata(exe_dir, args.antigen_pdb, args.antibody_pdb)
    create_input_data(exe_dir, args.antigen_pdb, args.antibody_pdb, args.trans_file, args.trans_num)



if __name__ == '__main__':
    main()
