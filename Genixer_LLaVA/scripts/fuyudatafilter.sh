CHUNKS=8
name=Genixer-llava-v1.5-7b
datafile=./playground/data/GenQA_eval/sbu_83w_tem0.2/Common/answers/Genixer-llava-v1.5-7b/vqafile.jsonl
imagedir=/yourpath/sbu_captions/images
datatype=sbu_83w

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${IDX} python -m GenQA_filtering_mp \
        --imagebasedir $imagedir \
        --datafile $datafile \
        --savefile answers/$datatype/$name/${CHUNKS}_${IDX}.jsonl \
        --isfeedback False \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=answers/$datatype/$name/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat answers/$datatype/$name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done