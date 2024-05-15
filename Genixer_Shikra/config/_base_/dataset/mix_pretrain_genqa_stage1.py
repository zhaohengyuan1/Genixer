_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    train=dict(
        type='ConcatDatasetWithShuffle',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=41,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.genrecvgdata}},
            ),

            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gengcdata}},
            ),

            dict(
                type='SubSet',
                portion=1,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.genrecdata}},
            ),

            dict(
                type='SubSet',
                portion=1,
                do_shuffle=True,
                seed=44,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.genregdata}},
            ),

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
