CHUNKS=8
CKPT=llava-v1.5-7b-Genixer

qfile=data/flickr30k_imagequery.jsonl
imgdir=/yourpath/flickr30k/flickr30k_images/flickr30k_images
datatype=flickr30k_tem0.2
tasktype=Adv

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python -m model_genixer_eval \
        --model-path exp/$CKPT \
        --question-file $qfile \
        --image-folder $imgdir \
        --answers-file ./playground/data/genixer_eval/$datatype/$tasktype/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --task-type $tasktype \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.2 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/genixer_eval/$datatype/$tasktype/answers/$CKPT/merge.jsonl
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/genixer_eval/$datatype/$tasktype/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done