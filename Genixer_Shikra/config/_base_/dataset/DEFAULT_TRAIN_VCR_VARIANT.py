VCR_TRAIN_COMMON_CFG = dict(
    type='VCRDataset',
    filename=r'{{fileDirname}}/../../../data/vcr_train.jsonl',
    image_folder=r'/yourpath/vcr/vcr1images',
)

DEFAULT_TRAIN_VCR_VARIANT = dict(
    VCR_q_a=dict(**VCR_TRAIN_COMMON_CFG, version='q-a', template_file=r"{{fileDirname}}/template/VQA.json",),
    VCR_q_ra=dict(**VCR_TRAIN_COMMON_CFG, version='q-ra', template_file=r"{{fileDirname}}/template/VQA_BCoT.json",),
    VCR_qc_a=dict(**VCR_TRAIN_COMMON_CFG, version='qc-a', template_file=r"{{fileDirname}}/template/VQA.json",),
    VCR_qc_ra=dict(**VCR_TRAIN_COMMON_CFG, version='qc-ra', template_file=r"{{fileDirname}}/template/VQA_BCoT.json",),
    VCR_qc_rac=dict(**VCR_TRAIN_COMMON_CFG, version='qc-rac', template_file=r"{{fileDirname}}/template/VQA_BCoT.json",),
    VCR_qa_r=dict(**VCR_TRAIN_COMMON_CFG, version='qa-r', template_file=r"{{fileDirname}}/template/VQA.json",),
    VCR_q_a_q_r=dict(**VCR_TRAIN_COMMON_CFG, version='q-a-q-r', template_file=r"{{fileDirname}}/template/VQA.json",),
    VCR_qac_r=dict(**VCR_TRAIN_COMMON_CFG, version='qac-r', template_file=r"{{fileDirname}}/template/VQA.json",),
    VCR_qc_a_qc_r=dict(**VCR_TRAIN_COMMON_CFG, version='qc-a-qc-r', template_file=r"{{fileDirname}}/template/VQA.json",),
)

