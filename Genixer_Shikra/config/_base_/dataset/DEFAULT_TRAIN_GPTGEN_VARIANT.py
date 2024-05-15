GPTGEN_TRAIN_COMMON_CFG = dict(
    type='GPT4Gen',
    filename=r'{{fileDirname}}/../../../data/GPT4GEN_BoxCoT_train.jsonl',
    image_folder=r'/yourpath/flickr30k/flickr30k_images/flickr30k_images',
)

DEFAULT_TRAIN_GPTGEN_VARIANT = dict(
    GPT4GEN_QA=dict(**GPTGEN_TRAIN_COMMON_CFG, version='a', template_file=r"{{fileDirname}}/template/VQA.json"),
    GPT4GEN_QC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='c', template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GPT4GEN_QBC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='bc', template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GPT4GEN_RD_QBC=dict(
        type=GPTGEN_TRAIN_COMMON_CFG['type'],
        image_folder=GPTGEN_TRAIN_COMMON_CFG['image_folder'],
        filename='{{fileDirname}}/../../../data/GPT4GEN_RD_BoxCoT_train.jsonl',
        version='bc',
        template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    
    GenGPT4GEN_QBC=dict(
        type='GenGPT4Gen',
        image_folder=r'/yourpath/flickr30k/flickr30k_images/flickr30k_images',
        filename='{{fileDirname}}/../../../data/GPT4GEN_BoxCoT_train.jsonl',
        version='bc',
        template_file=r"{{fileDirname}}/template/GenQA_general_instructions.json",
        task_tem=' This is a QCA with box task.'
        ),
    GenGPT4GEN_RD_QBC=dict(
        type='GenGPT4Gen',
        image_folder=r'/yourpath/flickr30k/flickr30k_images/flickr30k_images',
        filename='{{fileDirname}}/../../../data/GPT4GEN_RD_BoxCoT_train.jsonl',
        version='bc',
        template_file=r"{{fileDirname}}/template/GenQA_general_instructions.json",
        task_tem=' This is a Referential Dialogue task.'
    ),
)
