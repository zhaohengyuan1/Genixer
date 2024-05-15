_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    train=dict(
        type='ConcatDatasetWithShuffle',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/40,
                do_shuffle=True,
                seed=41,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.genrecvgdata}},
            ),

            dict(
                type='SubSet',
                portion=1/40,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gengcdata}},
            ),

            dict(
                type='SubSet',
                portion=1/4,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.genrecdata}},
            ),

            dict(
                type='SubSet',
                portion=1/4,
                do_shuffle=True,
                seed=44,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.genregdata}},
            ),

            ## GenPointQA
            {{_base_.DEFAULT_TRAIN_DATASET.GenPOINT_LOCAL_b}}, # 27k
            {{_base_.DEFAULT_TRAIN_DATASET.GenPOINT_LOCAL_p}}, # 27k
            {{_base_.DEFAULT_TRAIN_DATASET.GenPOINT_V7W_b}}, # 82k
            {{_base_.DEFAULT_TRAIN_DATASET.GenPOINT_V7W_p}}, # 82k
            
            ## GenQBC
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_QBC}}, # 4k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_QBC}}, # 4k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_QBC}}, # 4k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_QBC}}, # 4k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_QBC}}, # 4k

            ## GenRD
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_RD_QBC}}, # 1.8k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_RD_QBC}}, # 1.8k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_RD_QBC}}, # 1.8k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_RD_QBC}}, # 1.8k
            {{_base_.DEFAULT_TRAIN_DATASET.GenGPT4GEN_RD_QBC}}, # 1.8k
            
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
