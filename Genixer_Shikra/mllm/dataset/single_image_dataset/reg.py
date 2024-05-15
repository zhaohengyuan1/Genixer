from ..utils import (
    MInstrDataset,
)

from ..root import (
    DATASETS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
)


@DATASETS.register_module()
class REGDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER)
        caption = expr

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'{caption}',
                }
            ]
        }
        return ret


import json

def norm_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box

@DATASETS.register_module()
class GenREGDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.item_question_tem = json.load(open('/yourpath/Genixer/Genixer_Shikra/config/_base_/dataset/template/REG_question.json', 'r', encoding='utf8'))

    def get_item_question_tem(self):
        return self.rng.choice(self.item_question_tem)
    
    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        height = item['height']
        width = item['width']

        image = self.get_image(img_path)


        final_question = self.get_template() + ' This is a Referring Expression Generation (REG) task. The purpose of REG is to generate a unique description for a specified location.'

        bbox = norm_box_xyxy(bbox, w=width, h=height)

        box_strs = []
        box_strs.append(','.join([f"{elem:.{3}f}" for elem in bbox]))
        box_str = '[' + ';'.join(box_strs) + ']'

        Item_question = self.get_item_question_tem().replace(OBJS_PLACEHOLDER, box_str)

        caption = expr

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"Question: {Item_question} Answer: {caption}.",
                }
            ]
        }
        return ret





@DATASETS.register_module()
class GenREGEvalDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)

    def get_item_question_tem(self):
        return self.rng.choice(self.item_question_tem)
    
    def __getitem__(self, index):
        item = self.get_raw_item(index)
        
        if 'img_path' in item:
            img_path = item['img_path']
        elif 'image_path' in item:
            img_path = item['image_path']

        image = self.get_image(img_path)

        final_question = self.get_template() + ' This is a Referring Expression Generation (REG) task. The purpose of REG is to generate a unique description for a specified location.'

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"",
                }
            ]
        }
        return ret


@DATASETS.register_module()
class GCDataset(REGDataset):
    pass
