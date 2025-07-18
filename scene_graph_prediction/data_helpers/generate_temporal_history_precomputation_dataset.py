import random
import warnings
from pathlib import Path
from random import shuffle

import json_tricks as json  # Allows to load integers etc. correctly
from tqdm import tqdm

from scene_graph_prediction.data_helpers.generate_dataset_format_for_qwen2 import scene_graph_to_string
from scene_graph_prediction.scene_graph_helpers.dataset.egosurgery_dataset import EgoSurgeryDataset
from scene_graph_prediction.scene_graph_helpers.dataset.mvor_dataset import MVORDataset
from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset

warnings.filterwarnings('ignore')

QUESTIONS_TO_BASE_FORMULATION = {
    'current_action': 'What is currently happening?',
    'current_scene_graph': 'What is the current scene graph?',
}

SPLIT = 'train'  # train/val/test as needed
OR4D_DATASET = ORDataset({'USE_VIS_DESC': False}, SPLIT, load_4dor=True, load_mmor=False)
MMOR_DATASET = ORDataset({'USE_VIS_DESC': False}, SPLIT, load_4dor=False, load_mmor=True)
MVOR_DATASET = MVORDataset(SPLIT)
EGOSURGERY_DATASET = EgoSurgeryDataset(SPLIT)

DATASETS = {'4D-OR': OR4D_DATASET, 'MM-OR': MMOR_DATASET, 'MVOR': MVOR_DATASET, 'EgoSurgery': EGOSURGERY_DATASET}


def _get_current_actions(sg):
    current_actions = set()
    for sub, obj, rel in sg:
        rel = rel.lower()
        if rel in ['closeto', 'lyingon', 'holding', 'touching', 'assisting']:  # discard these
            continue
        current_actions.add(rel)
    return current_actions


def _get_qa_formatting(question_type, answer, question_format_args=None):
    question_formulations = [QUESTIONS_TO_BASE_FORMULATION[question_type]]
    question = QUESTIONS_TO_BASE_FORMULATION[question_type]
    question_formulation = random.choice(question_formulations)
    if question_format_args is not None:
        question = question.format(*question_format_args)
        question_formulation = question_formulation.format(*question_format_args)
    question_formulation = f'{question_formulation} Answer concisely.'
    answer_formulation = answer
    answer_formulation = answer_formulation[0].upper() + answer_formulation[1:]
    return question, question_formulation, answer_formulation


def _get_qa_pair_current_action(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    if dataset_name == 'MVOR':
        current_action = sample['action_label']
    elif dataset_name == 'EgoSurgery':
        current_action = sample['phase_label']
        if current_action is None:
            current_action = 'nothing'
    else:
        current_actions = _get_current_actions(sample['relationships'])
        if len(current_actions) == 0:
            current_action = 'nothing'
        else:
            current_action = random.choice(list(current_actions))
    answer = current_action
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_current_scene_graph(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    sg = sample['relationships']
    shuffle(sg)
    answer = scene_graph_to_string(sg)
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def main():
    save_root = Path(f'data/final_qa_pairs_temporal_history_precomputation_{SPLIT}.json')
    qa_pairs = []
    for dataset_name, dataset in tqdm(DATASETS.items(), desc='Processing datasets'):
        for sample in tqdm(dataset, desc=f'Processing {dataset_name}'):
            answer_form = 'concise'
            question_type = 'current_scene_graph' if dataset_name in ['4D-OR', 'MM-OR'] else 'current_action'
            if question_type == 'current_scene_graph':
                qa_pair = _get_qa_pair_current_scene_graph(sample, dataset_name, question_type, answer_form)
            elif question_type == 'current_action':
                qa_pair = _get_qa_pair_current_action(sample, dataset_name, question_type, answer_form)
            else:
                raise f'Not implemented question type: {question_type}'
            if qa_pair is not None:
                qa_pairs.append(qa_pair)

    with save_root.open('w') as f:
        json.dump(qa_pairs, f, indent=4)


if __name__ == '__main__':
    main()
