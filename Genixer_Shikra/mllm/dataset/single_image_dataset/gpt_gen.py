from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
)
from ..utils import MInstrDataset
from ..utils.flickr30k_entities_utils import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER

from typing import List, Union

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]

@DATASETS.register_module()
class GPT4Gen(MInstrDataset):
    def __init__(self, *args, version, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        assert version in ['a', 'c', 'bc']

    def __getitem__(self, item):
        raw = self.get_raw_item(item)
        #
        image = self.get_image(raw['img_path'])
        #
        boxes = raw['boxes']
        #
        question = raw['question']
        question = question.replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        query_boxes_seq = raw['question_boxes_seq']

        if self.version == 'a':
            final_answer = raw['answer']
            answer_boxes_seq = None
        elif self.version == 'c':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, '')
            answer_boxes_seq = None
        elif self.version == 'bc':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            answer_boxes_seq = raw['answer_boxes_seq']
        else:
            assert False

        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': final_answer,
                    'boxes_seq': answer_boxes_seq,
                }
            ]
        }
        return ret




def map_obj(boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    try:
        ret = []
        for boxes in boxes_seq:
            boxes_ret = []
            for box_index in boxes:
                if isinstance(box_index, (list, tuple)):
                    boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                else:
                    boxes_ret.append(boxes_value[box_index])
            ret.append(boxes_ret)
        return ret
    except:
        raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")


def format_box(boxes: Boxes) -> str:
    box_strs = []
    for box in boxes:
        box_strs.append(','.join([f"{elem:.{3}f}" for elem in box]))
    box_str = ';'.join(box_strs)
    return "[" + box_str + "]"

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
class GenGPT4Gen(MInstrDataset):
    def __init__(self, *args, version, task_tem, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        assert version in ['bc']

        self.task_tem = task_tem

    def __getitem__(self, item):
        raw = self.get_raw_item(item)
        image = self.get_image(raw['img_path'])
        
        boxes = raw['boxes']
        question = raw['question']
        question = question.replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        question_boxes_seq = raw['question_boxes_seq']
        

        width = raw['width']
        height = raw['height']

        normed_boxes = [norm_box_xyxy(box, w=width, h=height) for box in boxes]
        q_bboxes_seq = map_obj(normed_boxes, question_boxes_seq)
        q_bboxes_strs = [format_box(bboxes) for bboxes in q_bboxes_seq]
        Item_question = question.replace(BOXES_PLACEHOLDER, '{}').format(*q_bboxes_strs)
        
        ## genqa query tem with <image>
        query_question = self.get_template() + self.task_tem
        
        if self.version == 'bc':
            Item_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            answer_boxes_seq = raw['answer_boxes_seq']

            a_bboxes_seq = map_obj(normed_boxes, answer_boxes_seq)

            a_bboxes_strs = [format_box(bboxes) for bboxes in a_bboxes_seq]
            Item_answer = Item_answer.replace(BOXES_PLACEHOLDER, '{}').format(*a_bboxes_strs)

        else:
            assert False

        # print('query_question', query_question, f'Question: {Item_question} Answer: {Item_answer}')

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': query_question,
                },
                {
                    'from': 'gpt',
                    'value': f'Question: {Item_question} Answer: {Item_answer}',
                }
            ]
        }
        return ret


@DATASETS.register_module()
class GenGPT4Gen_Eval(MInstrDataset):
    def __init__(self, *args, version, task_tem, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        assert version in ['bc']

        self.task_tem = task_tem

    def __getitem__(self, item):
        raw = self.get_raw_item(item)
        image = self.get_image(raw['image_path'])
        
        ## genqa query tem with <image>
        query_question = self.get_template() + self.task_tem
        
        # answer
        Item_answer = ""
        # question
        Item_question = ""

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': query_question,
                },
                {
                    'from': 'gpt',
                    'value': f'Question: {Item_question} Answer: {Item_answer}',
                }
            ]
        }
        return ret