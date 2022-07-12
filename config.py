
import tensorflow as tf
from dock_layers import CloseDist_loss,weighted_BCE

config=dict(
    data_file="train",
    data_type="fine",
    line_len=6,
    patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_fine",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=27,  # 65
        graph_latent_dim=56,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=2,  # 7
        number_of_2dtransformer=4,  # 5
        size_r=830,
        size_l=450,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=3,
        heads_1d=3,
        heads_2d=4,
        conv_filters=[60, 50, 45, 40, 18],
        pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        class_weight={0: 1.145, 1: 0.25},
        batch_size=60,
        num_of_epochs=160,
        steps_per_epoch=330,
        lr=3 * (1e-3),
        cosine_start=8e-5,
        cosine_steps=44800,
        cosine_alpha=5e-6
    ),
    loss_weights=dict(
        classification=1,
        rec_rmsd=0.05,
        lig_rmsd=0.1
    ),
    losses_type={
        "classification": "binary_crossentropy",
        "log_rec_rmsd": CloseDist_loss(),
        "log_lig_rmsd": CloseDist_loss(),
    },
    model_suffix="ADAMW_ABDB_bound_graph_56_27_5e5_lr_weights_1.145_0.25_1wd-2_sample_4_seq_identity<0.94",
    workdir="ABDB",
    suffix_r="_Ab",
    suffix_l="_Ag",
    optimizer=5,
    wd=1e-2
)

config1=dict(
    line_len=4,
    data_file="train",
    data_type_train="fine_soap",
    data_type_test="soap",
    self_distogram_dir="/cs/labs/dina/matanhalfon/self_distograms",
    pdb_file_suffix="",
    patch_dir="/cs/labs/dina/matanhalfon/patch_data_dockground_1",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=40,  # 65
        graph_latent_dim=60,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=3, # 7
        number_of_2dtransformer=4 , # 5
        size_l=720,
        size_r=450,
        drop_rate=0,
        encoder_drop_rate=0.2,
        kernal_size=[1,3,3,3],
        heads_1d=4,
        heads_2d=5,
        conv_filters=[55, 50, 50, 50, 60],
        pooling_layers=[False,True,True,False,False],
        global_pool=True

        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        # class_weight={0: 1.19, 1: 0.3  },
        sample_rate=5,
        batch_size=45,
        num_of_epochs=160,
        steps_per_epoch=340,
        lr=3 * (1e-3),
        cosine_start=2e-4,
        cosine_steps=54400,
        cosine_alpha=1e-7
    ),
    loss_weights=dict(
        classification=1,
        rec_rmsd=0,
        lig_rmsd=0.2
    ),
    losses_type = {
        "classification": "binary_crossentropy",
        "rec_rmsd":CloseDist_loss(-0.1),
        "lig_rmsd": CloseDist_loss(-0.1),
    },
    model_suffix="ADAMW_norm_docking_graph_60_40_heads_4_4_weights_1.19_0.3_wd_1e-3_kernals_1_3_3_sample_4",
    # workdir="dockground/data_split/data_splits_over_neg/split_5",
    workdir="dockground",
    suffix_r="_u1",
    suffix_l="_u2",
    optimizer=5,
    adamW_steps=[8000,12000 ,14000],
    adamW_limits=[2e-4, 8e-5, 5e-5,3e-5],
    wd=5e-3
)

config2=dict(
    data_file="train",
    data_type_train="fine_soap",
    data_type_test="soap",
    pdb_file_suffix="",
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distogram_nano",
    line_len=6,
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_nano",
    patch_dir="/cs/labs/dina/matanhalfon/ABDB_nano_split_1",
    patch_dir_complex="/cs/labs/dina/matanhalfon/ABDB_nano_split_1/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=40,  # 65
        graph_latent_dim=70,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=3,  # 7
        number_of_2dtransformer=3,  # 5
        size_r=700,
        size_l=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=4,
        heads_2d=5,
        conv_filters=[55, 45, 40, 40, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        dist_alpha=-0.1,
        sample_rate=4,
        batch_size=55,
        num_of_epochs=130,
        steps_per_epoch=380,
        lr=3 * (1e-3),
        cosine_start=2e-4,
        cosine_steps=50000,
        cosine_alpha=1e-7
    ),
    loss_weights=dict(
        classification=1,
        rec_rmsd=0,
        lig_rmsd=0.3
    ),
    losses_type={
        "classification": "binary_crossentropy",
        # "classification": weighted_BCE(0.9),#positive  weights
        "rec_rmsd": CloseDist_loss(-0.1),
        "lig_rmsd": CloseDist_loss(-0.1),
    },
    model_suffix="ADAMW_split_2_nano_transformer_3_3_heads_4_5_70_40_2e4_lr_kernal_1_3_3_wd_1e-3_batched_gen_faster",
    workdir="ABDB/splited_data/split_1",
    suffix_r="_Ab_nano",
    suffix_l="_Ag",
    optimizer=5,
    wd=1e-3
)



config3=dict(
    data_file="train",
    data_type_train="AlphaFold",
    data_type_test="AlphaFold",
    pdb_file_suffix="",
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distogram_alphafold",
    line_len_R=14,
    line_len_L=14,
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data_ABDB_nano",
    patch_dir="/cs/labs/dina/matanhalfon/patch_data_Alphafold",
    patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data_Alphafold/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=40,  # 65
        graph_latent_dim=65,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=3,  # 7
        number_of_2dtransformer=4,  # 5
        size_l=750,
        size_r=250,
        drop_rate=0,
        encoder_drop_rate=0.25,
        kernal_size=[1,3,3,3],
        heads_1d=4,
        heads_2d=4,
        conv_filters=[55, 45, 40, 40, 30],
        pooling_layers=[False, True, True, True],
        global_pool=True
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        dist_alpha=-0.1,
        sample_rate=4,
        batch_size=40,
        num_of_epochs=130,
        steps_per_epoch=480,
        lr=3 * (1e-3),
        cosine_start=1e-4,
        cosine_steps=63000,
        cosine_alpha=1e-7
    ),
    loss_weights=dict(
        classification=1,
        rec_rmsd=0,
        lig_rmsd=0.3
    ),
    losses_type={
        "classification": "binary_crossentropy",
        # "classification": weighted_BCE(0.9),#positive  weights
        "rec_rmsd": CloseDist_loss(-0.1),
        "lig_rmsd": CloseDist_loss(-0.1),
    },
    model_suffix="ADAMW_Alphfold_transformer_3_4_heads_4_4_65_40_2e4_lr_kernal_1_3_3_wd_1e-3_sample4",
    workdir="ABDB",
    suffix_r="_Ab",
    suffix_l="_Ag",
    optimizer=5,
    wd=1e-3
)



config4=dict(
    data_file="train",
    data_type_train="AlphaFold",
    data_type_test="AlphaFold",
    pdb_file_suffix="",
    self_distogram_dir="/cs/labs/dina/matanhalfon/CAPRI/ABDB/self_distogram_alphafold",
    line_len_R=14,
    line_len_L=14,
    patch_dir="/cs/labs/dina/matanhalfon/patch_data/patch_data_AlphaFold_big_batch",
    patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data/patch_data_AlphaFold_big_batch/test",
    # patch_dir="/cs/labs/dina/matanhalfon/patch_data/patch_data_AlphaFold_no_db_5",
    # patch_dir_complex="/cs/labs/dina/matanhalfon/patch_data/patch_data_AlphaFold_no_db_5/test",
    arch=dict(
        number_of_patches=8,  # 6
        seq_latent_dim=60,  # 65
        graph_latent_dim=100,  # 80
        patch_size=20,  # 30
        number_of_1dtransformer=2,  # 7
        number_of_2dtransformer=3,  # 5
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
        class_predictor=[40,30,20],#50
        dockQ_predictor=[40,30,5]
        # pooling_layers=[True, False, True, False, False],
        # conv_filters=[75, 65, 56, 32]
    ),
    hyper=dict(
        dist_alpha=-0.1,
        sample_rate=4,
        batch_size=52,
        num_of_epochs=200,
        steps_per_epoch=2000,
        lr=3 * (1e-3),
        cosine_start=1e-4,
        cosine_steps=320000,
        cosine_alpha=1e-2
        # cosine_steps=100000,
        # cosine_alpha=5e-3,
        # t_mul=1.5, #steps_mul
        # m_mul=1.0# lr restart muls

    ),
    loss_weights=dict(
        classification=1,
        rec_rmsd=0,
        dockQ=0,
        lig_rmsd=0.2
    ),
    losses_type={
        "classification": tf.keras.losses.BinaryCrossentropy(),
        "dockQ":tf.keras.losses.CategoricalCrossentropy(),
        # "classification": weighted_BCE(0.95),#positive  weights
        # "rec_rmsd": CloseDist_loss(-0.1),s
        "lig_rmsd": CloseDist_loss(-0.1),
    },
    model_suffix="ADAMW_Alphfold_transformer_2_3_heads_3_5_60_100__1e3_lr_wd_5e-3_sample4_only_dist_losss",
    workdir="ABDB/AlphaFold_data_dir/AlphaFold_data",
    # workdir="ABDB/AlphaFold_data_dir/no_db_5",
    suffix_r="_Ab",
    suffix_l="_Ag",
    optimizer=5,
    wd=5e-3
)