from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

import numpy as np

import argparse
import os
import json
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def getavgjointprob(model, inputs, outputs):
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    all_prob = []
    jointprob = 1

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        # print(f"| {tok:5d} | {processor.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
        all_prob.append(np.exp(score.cpu().numpy()))
        jointprob = jointprob * np.exp(score.cpu().numpy())

    all_prob = all_prob[:-1]
    # print(jointprob)
    # print(np.power(jointprob, 1/len(all_prob)))
    return np.power(jointprob, 1/len(all_prob))



def filtering_data(args):
    model_id = "fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda")

    if args.imagebasedir != None:
        imagebasedir = args.imagebasedir
    else:
        imagebasedir = ''
    
    datafile = args.datafile
    savefile = args.savefile

    Tem = "Here is a question and answer pair. Is '{}' true for this image?\nPlease answer this question with Yes or No.\n"

    Tem_fb = "Here is a question and answer pair. Is '{}' true for this image?\nPlease answer this question with Yes or No and give the reasons or feedback.\n"

    all_data = []
    for q in open(os.path.expanduser(datafile), "r"):
        all_data.append(json.loads(q))

    all_data = get_chunk(all_data, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.savefile)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if os.path.exists(answers_file):
        jsonl_file = open(savefile, 'a+')
    else:
        jsonl_file = open(savefile, 'w')
    
    start_idx = len([json.loads(q) for q in open(savefile, 'r')])

    for idx in range(start_idx, len(all_data)):
        
        # text_prompt = "Generate a coco-style caption.\n"
        if idx % 10 == 0:
            print(idx)

        data = all_data[idx]

        qatext = f"Question: {data['question']} Answer: {data['answer']}"
        image = Image.open(os.path.join(imagebasedir, data['image_path'])).convert('RGB')

        text_prompt = Tem.format(qatext)
        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to('cuda')

        try:
            # generation_output = model.generate(**inputs, max_new_tokens=10, pad_token_id=model.config.eos_token_id, return_dict_in_generate=True, output_scores=True)
            generation_output = model.generate(**inputs, max_new_tokens=10, do_sample=True, top_p=0.95, top_k=10, pad_token_id=model.config.eos_token_id, return_dict_in_generate=True, output_scores=True)
            generation_text = processor.batch_decode(generation_output.sequences[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
            prob = getavgjointprob(model, inputs, generation_output)

            if args.isfeedback == 'True':
                text_prompt2 = Tem_fb.format(qatext)
                inputs2 = processor(text=text_prompt2, images=image, return_tensors="pt").to("cuda")
                generation_output2 = model.generate(**inputs2, max_new_tokens=60, do_sample=True, top_p=0.95, top_k=10, pad_token_id=model.config.eos_token_id)
                generation_text2 = processor.batch_decode(generation_output2[:, inputs2['input_ids'].size(1):], skip_special_tokens=True)
                feedback = generation_text2[0]
            else:
                feedback = ''
        except:
            continue
        
        fuyure = generation_text[0]
        

        data_dict = {
            "question_id": data["question_id"],
            "image_path": data["image_path"],
            "question": data["question"],
            "answer": data["answer"],
            "fuyu8b_results": fuyure,
            "probability": prob,
            "feedback": feedback
        }

        jsonl_file.write(json.dumps(data_dict)+'\n')

        jsonl_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagebasedir", type=str, default=None)
    parser.add_argument("--datafile", type=str, default="")
    parser.add_argument("--savefile", type=str, default="")
    parser.add_argument("--isfeedback", type=str, default="False")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    filtering_data(args)