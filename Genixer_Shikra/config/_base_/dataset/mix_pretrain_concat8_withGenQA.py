_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/40,
                do_shuffle=True,
                seed=40,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gc}},
            ),
            dict(
                type='SubSet',
                portion=1/40,
                do_shuffle=True,
                seed=41,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.recvg}},
            ),

            dict(
                type='SubSet',
                portion=1/5,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
            ),

            dict(
                type='SubSet',
                portion=1/5,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.llavalcs}},
            ),

            dict(
                type='SubSet',
                portion=1/5,
                do_shuffle=True,
                seed=44,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.VQAv2_train}}, # 43w
            ),
            
            {{_base_.DEFAULT_TRAIN_DATASET.VQAE_train}},
            {{_base_.DEFAULT_TRAIN_DATASET.VQAX_train}},

            dict(
                type='SubSet',
                portion=1/5,
                do_shuffle=True,
                seed=45,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.caption}},
            ),

            {{_base_.DEFAULT_TRAIN_DATASET.rec}}, # 28w
            {{_base_.DEFAULT_TRAIN_DATASET.reg}}, # 28w

            {{_base_.DEFAULT_TRAIN_DATASET.flickr}},

            ## synthetic data
            {{_base_.DEFAULT_TRAIN_DATASET.syn_lcsrec}},
            {{_base_.DEFAULT_TRAIN_DATASET.syn_sburec}},
            ## synthetic data

            {{_base_.DEFAULT_TRAIN_DATASET.VCR_q_ra}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_rac}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qac_r}},

            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_b}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_oq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_sq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_gq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_b}},

        ],
    ),
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
