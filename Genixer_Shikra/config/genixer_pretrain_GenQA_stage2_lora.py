_base_ = ['_base_/dataset/mix_pretrain_genqa_stage2.py', '_base_/model/genixer_shikra.py', '_base_/train/train.py']

training_args = dict(
    num_train_epochs=5,
    save_steps=1000000, # no middle saving
    learning_rate=2e-4,
    output_dir=None,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    save_total_limit=1,
)

model_args = dict(
    type='genixer_shikra_lora',
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path=None,

    # lora setting
    lora_enable=True,
    lora_r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    lora_weight_path="",
    lora_bias="none"
)
