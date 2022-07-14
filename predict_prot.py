import os.path as osp
from  model_evaluate import hit_rate,load_modle
import argparse


model_path="/cs/labs/dina/matanhalfon/CAPRI/NNscripts/lr-0.003_5_train_transformer_ABDBADAMW_Alphfold_transformer_1_3_heads_3_6_72_102_1e3_lr_wd_5e-3_sample3_split_1_dockQ_smallers/mymodel_116"



config=dict(
    data_file="train",
    line_len_R=6,
    line_len_L=2,
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=72,  # 65
        graph_latent_dim=102,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=1,  # 7
        number_of_2dtransformer=3,  # 5
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
        class_predictor=[40,30,20],#50
        dockQ_predictor=[40,30,5]
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
    parser.add_argument('--trans_num',type=int,default=0,help="# of transform to read from a transformation file, default all")
    args=parser.parse_args()
    model = load_modle(model_path, config)
    hit_rate(model, args.antigen_pdb,args.antibody_pdb, config, trans_num=args.trans_num,data_dir="PPI")
