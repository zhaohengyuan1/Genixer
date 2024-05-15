import re

from .. import BaseComputeMetrics
from ..root import (
    DATASETS,
    METRICS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    POINTS_PLACEHOLDER,
)
from ..utils import MInstrDataset

from typing import List, Dict, Any, Tuple, Union

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]

# noinspection PyPep8Naming
@DATASETS.register_module()
class Point_QA_local(MInstrDataset):
    def __init__(self, *args, version='p', qbp_p_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        assert version in ['b', 'p', 'bp']
        self.version = version
        self.qbp_p_prob = qbp_p_prob

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # answer
        answer = item['answer']
        # question
        question = item['question']
        bbox = item['bbox']
        point = item['point']

        version = self.version
        if version == 'bp':
            version = 'p' if self.rng.random() < self.qbp_p_prob else 'b'
        if version == 'b':
            question = question + BOXES_PLACEHOLDER
            query_boxes_seq = [[0]]
            query_points_seq = None
        elif version == 'p':
            question = question + POINTS_PLACEHOLDER
            query_boxes_seq = None
            query_points_seq = [[0]]
        else:
            assert False
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'points': [point],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                    'points_seq': query_points_seq,
                },
                {
                    'from': 'gpt',
                    'value': f'The answer is {answer} .',
                }
            ]
        }
        return ret



def norm_point_xyxy(point, *, w, h):
    x, y = point
    norm_x = max(0.0, min(x / w, 1.0))
    norm_y = max(0.0, min(y / h, 1.0))
    point = norm_x, norm_y
    return point


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
class GenPoint_QA_local(MInstrDataset):
    def __init__(self, *args, version='p', qbp_p_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        assert version in ['b', 'p', 'bp']
        self.version = version
        self.qbp_p_prob = qbp_p_prob


    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # answer
        Item_answer = item['answer']
        # question
        question = item['question']
        bbox = item['bbox']
        point = item['point']


        height = item['img_h']
        width = item['img_w']

        version = self.version
        if version == 'bp':
            version = 'p' if self.rng.random() < self.qbp_p_prob else 'b'
        if version == 'b':
            normed_bbox = norm_box_xyxy(bbox, w=width, h=height)

            box_strs = []
            box_strs.append(','.join([f"{elem:.{3}f}" for elem in normed_bbox]))
            box_str = '[' + ';'.join(box_strs)+ ']'

            question = question + box_str

        elif version == 'p':

            normed_point = norm_point_xyxy(point, w=width, h=height)

            point_strs = []
            point_strs.append(','.join([f"{elem:.{3}f}" for elem in normed_point]))
            point_str = '[' + ';'.join(point_strs)+ ']'

            question = question + point_str

        else:
            assert False

        Item_question = question

        query_question = self.get_template() + ' This is a PointQA task.'

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
class GenPoint_QA_local_Eval(MInstrDataset):
    def __init__(self, *args, version='p', qbp_p_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        assert version in ['b', 'p', 'bp']
        self.version = version
        self.qbp_p_prob = qbp_p_prob


    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        if 'img_path' in item:
            img_path = item['img_path']
        elif 'image_path' in item:
            img_path = item['image_path']

        image = self.get_image(img_path)
        # answer
        Item_answer = ""
        # question
        Item_question = ""

        query_question = self.get_template() + ' This is a PointQA task.'
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

# noinspection PyPep8Naming
@DATASETS.register_module()
class Point_QA_twice(MInstrDataset):
    def __init__(self, *args, version='gq-p', bp_p_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        self.bp_p_prob = bp_p_prob
        qtype, rtype = version.split('-')
        assert qtype in ['oq', 'sq', 'gq']
        assert rtype in ['b', 'p', 'bp']
        self.qtype = qtype
        self.rtype = rtype

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # answer
        answer = item['answer']
        # question
        bbox = item['bbox']
        point = item['point']
        if self.qtype == 'oq':
            question = item['obj_question']
        elif self.qtype == 'sq':
            question = item['super_question']
        elif self.qtype == 'gq':
            question = item['general_question']
        else:
            assert False
        rtype = self.rtype
        if rtype == 'bp':
            rtype = 'p' if self.rng.random() < self.bp_p_prob else 'b'
        if rtype == 'p':
            question = question + POINTS_PLACEHOLDER
            query_boxes_seq = None
            query_points_seq = [[0]]
        elif rtype == 'b':
            question = question + BOXES_PLACEHOLDER
            query_boxes_seq = [[0]]
            query_points_seq = None
        else:
            assert False
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'points': [point],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                    'points_seq': query_points_seq,
                },
                {
                    'from': 'gpt',
                    'value': f'The answer is {answer} .',
                }
            ]
        }
        return ret


# noinspection PyPep8Naming
@DATASETS.register_module()
class V7W_POINT(MInstrDataset):
    def __init__(self, *args, version, do_shuffle_choice=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        self.do_shuffle_choice = do_shuffle_choice
        assert version in ['p', 'b']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # question
        bboxes = item['candidates']
        points = []
        final_question = item['question'] + ' Candidates: ' + " ".join([BOXES_PLACEHOLDER for _ in range(len(bboxes))])
        query_boxes_seq = []
        for _ in range(len(bboxes)):
            query_boxes_seq.append([_])
        # answer
        if self.version == 'p':
            final_question += f" answer in point format."
            points.append(item['point'])
            final_answer = f"The answer is {POINTS_PLACEHOLDER} ."
            answer_boxes_seq = None
            answer_points_seq = [[0]]
        elif self.version == 'b':
            final_question += f" answer in box format."
            idx = bboxes.index(item['answer'])
            final_answer = f"The answer is {BOXES_PLACEHOLDER} ."
            answer_boxes_seq = [[idx]]
            answer_points_seq = None
        else:
            assert False
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, final_question)
        if self.do_shuffle_choice:
            self.rng.shuffle(query_boxes_seq)

        ret = {
            'image': image,
            'target': {
                'boxes': bboxes,
                'points': points,
            },
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
                    'points_seq': answer_points_seq,

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

@DATASETS.register_module()
class GenV7W_POINT(MInstrDataset):
    def __init__(self, *args, version, do_shuffle_choice=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        self.do_shuffle_choice = do_shuffle_choice
        assert version in ['p', 'b']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # question
        bboxes = item['candidates']

        height = item['img_h']
        width = item['img_w']

        # points = []

        Item_question = item['question'] + ' Candidates: ' + " ".join([BOXES_PLACEHOLDER for _ in range(len(bboxes))])
        query_boxes_seq = []
        for _ in range(len(bboxes)):
            query_boxes_seq.append([_])

        query_question = self.get_template()

        query_question += f" This is a PointQA task."

        # answer
        if self.version == 'p':
           
            point = item['point']
            normed_point = norm_point_xyxy(point, w=width, h=height)
            point_strs = []
            point_strs.append(','.join([f"{elem:.{3}f}" for elem in normed_point]))
            point_str = '[' + ';'.join(point_strs)+ ']'

            Item_answer = point_str

        elif self.version == 'b':

            normed_bbox = norm_box_xyxy(item['answer'], w=width, h=height)

            box_strs = []
            box_strs.append(','.join([f"{elem:.{3}f}" for elem in normed_bbox]))
            box_str = '[' + ';'.join(box_strs)+ ']'

            Item_answer = box_str

        else:
            assert False

        if self.do_shuffle_choice:
            self.rng.shuffle(query_boxes_seq)

        normed_bboxes = [norm_box_xyxy(bbox, w=width, h=height) for bbox in bboxes]

        q_bboxes_seq = map_obj(normed_bboxes, query_boxes_seq)
        q_bboxes_strs = [format_box(bboxes) for bboxes in q_bboxes_seq]
        Item_question = Item_question.replace(BOXES_PLACEHOLDER, '{}').format(*q_bboxes_strs)

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


ANS_EXTRACT_PAT = re.compile(r'(?:The answer is (.+?)\.)')


@METRICS.register_module()
class PointQAComputeMetrics(BaseComputeMetrics):
    def extract_ans(self, string: str):
        try:
            found = ANS_EXTRACT_PAT.findall(string.strip())
            if len(found) != 1:
                return None
            return found[0].strip()
        except (IndexError, AttributeError):
            return None
