import os.path as osp
from  model_evaluate import hit_rate,load_modle
import argparse
import sys

exe_dir = osp.dirname(osp.realpath(sys.argv[0]))
model_path=osp.join(exe_dir,"weights/mymodel_108")
model_path_nano=osp.join(exe_dir,"weights/mymodel_9")




config=dict(
    data_file="train",
    line_len_R=6,
    line_len_L=2,
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=48,  # 65
        graph_latent_dim=72,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=4,  # 7
        number_of_2dtransformer=4,  # 5
        size_l=250,
        size_r=700,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=3,
        heads_2d=6,
        conv_filters=[50, 40, 35, 35, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True,
        class_predictor=[40,30,20],#50
        dockQ_predictor=[40,20,10]
    ),
    hyper=dict(
        dist_alpha=-0.1,
        sample_rate=4,
        batch_size=1,
        num_of_epochs=12,
        steps_per_epoch=1000,
        lr=3 * (1e-3),
        cosine_start=5e-4,
        cosine_steps=14000,
        cosine_alpha=1e-2
    ),
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser for data preparation for single protein")
    parser.add_argument('antigen_pdb',type=str,help="Antigen PDB file name")
    parser.add_argument('antibody_pdb',type=str,help="Antibody PDB file name")
    parser.add_argument("-o", '--output',type=str, default="evaluation",help="Out file")
    parser.add_argument('--trans_num',type=int,default=5000,help="# of transform to read from a transformation file, default all (5000)")
    parser.add_argument('--nanobody', type=int, default=0, help="enter 1 if nanobody to use fine-tuned weights else 0")
    args=parser.parse_args()

    if args.nanobody:
        model = load_modle(model_path_nano, config)
    else:
        model = load_modle(model_path, config)

    hit_rate(model, args.antigen_pdb,args.antibody_pdb, config, trans_num=args.trans_num,data_dir="PPI", out_file=args.output)
