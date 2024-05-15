_base_ = ['_base_/dataset/mix_pretrain_genqa_stage1.py', '_base_/model/genixer_shikra.py', '_base_/train/train_fsdp.py']

training_args = dict(
    num_train_epochs=5,
    learning_rate=3e-5,
    output_dir=None,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    save_total_limit=1,
)

model_args = dict(
    type='genixer_shikra',
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path=None,
)
