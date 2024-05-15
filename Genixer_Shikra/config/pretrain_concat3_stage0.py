_base_ = ['_base_/dataset/mix_pretrain_concat3.py', '_base_/model/shikra.py', '_base_/train/train_fsdp.py']

training_args = dict(
    num_train_epochs=1,
    output_dir=None,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
)

model_args = dict(
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path=None,
)
