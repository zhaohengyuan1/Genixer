Flickr_image_folder = r'/yourpath/flickr30k/flickr30k_images/flickr30k_images'

DEFAULT_TEST_GenQA_VARIANT = dict(
    GENREC_lcs558k=dict(
        type='GenRECEvalDataset',
        filename=r'{{fileDirname}}/../../../data/blip_laion_cc_sbu_558k_imagequery.jsonl',
        image_folder=r'/yourpath/pretrained_data/LLaVA-Pretrain',
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
    ),

    GENREC_sbu=dict(
        type='GenRECEvalDataset',
        filename=r'{{fileDirname}}/../../../data/SBU_830k_imagequery.jsonl',
        image_folder=r'/yourpath/sbu_captions/images',
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
    ),

    GENREC_flickr30k=dict(
        type='GenRECEvalDataset',
        filename=r'{{fileDirname}}/../../../data/flickr30k_images_queryforgenqa.jsonl',
        image_folder=Flickr_image_folder,
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
    ),

    GENREG_flickr30k=dict(
        type='GenREGEvalDataset',
        filename=r'{{fileDirname}}/../../../data/flickr30k_images_queryforgenqa.jsonl',
        image_folder=Flickr_image_folder,
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
    ),

    GENPointQA_flickr30k=dict(
        type='GenPoint_QA_local_Eval',
        filename=r'{{fileDirname}}/../../../data/flickr30k_images_queryforgenqa.jsonl',
        image_folder=Flickr_image_folder,
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
    ),

    GENRD_flickr30k=dict(
        type='GenGPT4Gen_Eval',
        filename=r'{{fileDirname}}/../../../data/flickr30k_images_queryforgenqa.jsonl',
        image_folder=Flickr_image_folder,
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
        version='bc',
        task_tem=' This is a Referential Dialogue task.'
    ),

    GENQCAbox_flickr=dict(
        type='GenGPT4Gen_Eval',
        filename=r'{{fileDirname}}/../../../fast_data/flickr30k_images_queryforgenqa.jsonl',
        image_folder=Flickr_image_folder,
        template_file=r'{{fileDirname}}/template/GenQA_general_instructions.json',
        version='bc',
        task_tem=' This is a QCA with box task.'
    ),
)
