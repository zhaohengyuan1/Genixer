import multiprocessing
from multiprocessing import Pool, Manager

import os
import json
import re
import logging
import cv2

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and processor globally if they are constant and read-only
model_name = "checkpoints/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)


def de_norm_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def preprocess_image(image_path, corr):
    with Image.open(image_path) as image:
        image = image.crop(corr)
        inputs = processor(text=None, images=image, return_tensors="pt")
        return inputs


def process_image(data):
    pattern = r'\[\d+\.\d+,\d+\.\d+,\d+\.\d+,\d+\.\d+\]'
    image_path, text_data = data

    try:
        if 'coordinates of' in text_data:
            expression = text_data.split('coordinates of')[1].split('Answer:')[0].strip().rstrip('.').strip()
            expmatches = re.findall(pattern, expression)
            if expmatches:
                raise ValueError('Invalid expression format')

            answertext = text_data.split('Answer:')[1].strip()
            matches = re.findall(pattern, answertext)
            if not matches:
                raise ValueError('No coordinates found in answer')

            corr = [float(x) for x in matches[0][1:-1].split(',')]
            if len(corr) != 4:
                raise ValueError('Invalid number of coordinates')

            save_img = cv2.imread(image_path)
            height, width = save_img.shape[:2]
            corr = de_norm_box_xyxy(corr, width, height)
            if corr[2] - corr[0] < 50 or corr[3] - corr[1] < 50:
                raise ValueError('Cropped area is too small')

            text_inputs = processor(expression, images=None, return_tensors="pt").to(device)
            image_inputs = preprocess_image(image_path, corr).to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)
                similarity_score = (image_features @ text_features.T).mean().item()

            if similarity_score > 60:
                data_dict = {
                    "img_path": image_path,
                    "bbox": corr,
                    "expression": expression,
                    "height": height,
                    "width": width,
                    "clip_score": similarity_score
                }
                return data_dict
            else:
                return None

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None


def main():

    baseimgdir = '/yourpath/sbu_captions/images'
    queryfile = 'data/SBU_830k_imagequery.jsonl'
    text_path = '/yourpath/multitest_GENREC_sbu830k_extra_prediction.jsonl'
    savefile = 'clipscores/GENREC_sbu_830k_clipscores_over60.jsonl'

    all_images = [os.path.join(baseimgdir, json.loads(q)['image_path']) for q in open(queryfile, "r")]
    all_data = [json.loads(q)['pred'] for q in open(text_path, "r")]

    logging.info(f"Number of images: {len(all_images)}")
    logging.info(f"Number of data entries: {len(all_data)}")

    manager = Manager()
    counter = manager.Value('i', 0)
    total = len(all_images)  # 处理的总图片数


    with tqdm(total=total) as progress:
        def update(result):
            progress.update()

        # Wrap your data with additional arguments for counter and total
        data_for_processing = [(image_path, text_data) for image_path, text_data in zip(all_images, all_data)]

        # Open the JSONL file once and process images with progress tracking
        with open(savefile, 'w') as jsonl_file, Pool(processes=20, initializer=init, initargs=(counter,)) as pool:
            for result in pool.imap_unordered(process_image, data_for_processing):
                update(result)
                if result is not None:
                    json.dump(result, jsonl_file)
                    jsonl_file.write('\n')

    logging.info("All tasks completed and results written to file.")


def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
