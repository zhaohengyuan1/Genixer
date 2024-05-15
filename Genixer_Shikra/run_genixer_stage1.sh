### GenQA with Full LLM

## stage1
accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/genixer_pretrain_GenQA_stage1.py \
        --cfg-options model_args.model_name_or_path=checkpoints/shikra-7b\
        training_args.output_dir=exp/genixer_shikra_pretrain_genQA_stage1
