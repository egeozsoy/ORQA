import argparse
import random
import warnings
from copy import deepcopy
from pathlib import Path
from random import shuffle

import json_tricks as json  # Allows to load integers etc. correctly
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

from helpers.configurations import TRACKER_OBJECT_MAP
from scene_graph_prediction.data_helpers.generate_qa_dataset import SPLIT, DATASETS
from scene_graph_prediction.data_helpers.scene_graph_converters import scene_graphs_to_surgery_sg, surgery_sg_to_memory_str, action_labels_to_surgery_sg, surgery_sg_actions_to_memory_str

warnings.filterwarnings('ignore')


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def scene_graph_to_string(scene_graph):
    '''
    Scene graph is a list of relations in the form of (subject, relation, object)
    Return format: "<SG> subject,object,relation; subject,object,relation; </SG>"
    '''
    out = '<SG> '
    for (subject, object, relation) in scene_graph:
        subject = subject.replace('_', ' ').lower()
        object = object.replace('_', ' ').lower()
        out += f'{subject},{object},{relation}; '
    # remove the last ";" and add the end token.
    out = out.rstrip('; ') + ' </SG>'
    return out


def apply_template(qa_pair, image_paths, scene_graph, timepoint, sample_id, pc, audio, raw_audio, robot_metadata, tracker_metadata, speech_transcript, segmasks, dataset_name, take_name):
    human_prompt = qa_pair['question']
    with_scene_graph_grounding = random.random() < 0.5 if scene_graph is not None else False
    if with_scene_graph_grounding:  # 50% chance ask it to ground its thinking to the scene graphs (essentially ask it to predict the scene graph before the answer).No n
        human_prompt += ' Use the scene graph to guide your answer.'
    if robot_metadata is not None:
        with open(robot_metadata, 'r') as f:
            robot_metadata = json.load(f)
        robot_metadata_str = ''
        for key, value in sorted(robot_metadata.items()):
            robot_metadata_str += f'{value["type"]}: {value["template_name"]}, '
        robot_metadata_str = robot_metadata_str.rstrip(', ')
        # insert this into the beginning of the human prompt
        human_prompt = f'<robot_metadata_start>: {robot_metadata_str} <robot_metadata_end>. {human_prompt}'
    if tracker_metadata is not None:
        unique_id_dicts = tracker_metadata['unique_id_dicts']
        tracker_metadata_str = ''
        for unique_id_dict in unique_id_dicts:
            tool_name = TRACKER_OBJECT_MAP[unique_id_dict['unique_id']]
            tool_state = unique_id_dict['button_state']
            tool_translation = np.asarray(unique_id_dict['Translation']).astype(int)
            tool_translation_str = ' '.join(tool_translation.astype(str))
            tool_rotation = np.asarray(unique_id_dict['euler_rot']).astype(int)
            tool_rotation_str = ' '.join(tool_rotation.astype(str))
            tracker_metadata_str += f'{tool_name}: state {tool_state}, translation {tool_translation_str}, euler angles {tool_rotation_str}; '
        tracker_metadata_str = tracker_metadata_str.rstrip('; ')
        # insert this into the beginning of the human prompt
        human_prompt = f'<tracker_metadata_start>: {tracker_metadata_str} <tracker_metadata_end>. {human_prompt}'
    if speech_transcript is not None:
        with open(speech_transcript, 'r') as f:
            speech_transcript = json.load(f)
        speech_transcript_str = speech_transcript['text']
        # insert this into the beginning of the human prompt
        human_prompt = f'<speech_transcript_start>: {speech_transcript_str} <speech_transcript_end>. {human_prompt}'

    # Qwen format: messages with role. Keep id and timepoint as top-level keys.
    image_token = '<image>'
    image_tokens = image_token * len(image_paths)
    answer = qa_pair['answer']
    if with_scene_graph_grounding:
        # Should first predict the scene graph, then answer the question
        answer = f'{scene_graph} {answer}'
    sample = {
        "id": sample_id,
        "timepoint": timepoint,
        "messages": [
            {
                "role": "user",
                "content": f"{image_tokens}{human_prompt}"
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }

    if len(image_paths) > 0:
        sample['images'] = [str(image_path.absolute()) for image_path in image_paths]
    # these are not fully supported actually
    if len(segmasks) > 0:
        sample['segmasks'] = [str(segmask.absolute()) for segmask in segmasks]
    if pc is not None:
        sample['pc'] = str(pc.absolute())
    if audio is not None:
        sample['audio'] = str(audio.absolute())
    if raw_audio is not None:
        sample['raw_audio'] = str(raw_audio.absolute())

    sample['_question'] = qa_pair['_question']
    sample['_answer'] = qa_pair['_answer']
    sample['_question_type'] = qa_pair['_question_type']
    sample['_sg'] = scene_graph
    sample['_dataset_name'] = dataset_name
    sample['_take_name'] = take_name

    return sample


def generate_finetuning_samples(qa_pairs, MULTIMODAL_DROP_PROP):
    samples = []
    for qa_pair in tqdm(qa_pairs, desc='Generating samples'):
        # first we determine the dataset and fetch that dataset
        dataset_name = qa_pair['dataset']
        dataset = DATASETS[dataset_name]
        if qa_pair['_question_type'] == 'tool_equipment_attribute':
            if '_' in qa_pair['frame_id']:
                take_name, frame_id = qa_pair["frame_id"].rsplit('_', 1)
            else:
                take_name, frame_id = qa_pair['take_name'], qa_pair['frame_id']
            qa_pair["take_name"] = take_name
            qa_pair["frame_id"] = frame_id
        sample_key = f'{qa_pair["take_name"]}_{qa_pair["frame_id"]}'
        # fetch all the info about that sample, including all the modalities
        sample = dataset.get_sample_by_key(sample_key)
        sample, multimodal_data = sample['sample'], deepcopy(sample['multimodal_data'])
        # get length of multimodal data. As multimodal data is a dictionary with multiple keys, we consider the length to be the longest list
        multimodal_data_len = max([len(multimodal_data[key]) for key in multimodal_data])
        if multimodal_data_len == 0:
            continue

        sample_id = sample['sample_id']
        frame_id = sample['frame_id']
        image_paths = []
        azure_image_paths = multimodal_data['azure']
        simstation_image_paths = multimodal_data['simstation'] if 'simstation' in multimodal_data else []
        trackercam_image_paths = multimodal_data['trackercam'] if 'trackercam' in multimodal_data else []
        simstation_views_to_use = None
        if '4D-OR' in dataset_name:
            azure_views_to_use = (2, 1, 3, 5)
        elif 'MM-OR' in dataset_name:
            azure_views_to_use = (1, 4, 5, 2, 3)
            simstation_views_to_use = (2, 0, 1, 3)
        elif 'MVOR' in dataset_name:
            azure_views_to_use = (3, 1, 2)
        else:
            azure_views_to_use = (1,)
        if len(azure_image_paths) > 0:
            azure_image_paths = [azure_image_paths[view_idx - 1] for view_idx in azure_views_to_use]
            image_paths.extend(azure_image_paths)
        if len(simstation_image_paths) > 0:
            simstation_image_paths = [simstation_image_paths[view_idx] for view_idx in simstation_views_to_use]
            image_paths.extend(simstation_image_paths)
        if len(trackercam_image_paths) > 0:
            image_paths.extend(trackercam_image_paths[:1])

        raw_audio = multimodal_data['raw_audio'][0] if 'raw_audio' in multimodal_data else None
        pc = multimodal_data['pc'][0] if 'pc' in multimodal_data else None
        # segmasks = multimodal_data['segmasks'] if 'segmasks' in multimodal_data else []
        audio = multimodal_data['audio'][0] if 'audio' in multimodal_data else None
        robot_metadata = multimodal_data['robot_metadata'][0] if 'robot_metadata' in multimodal_data else None
        tracker_metadata = multimodal_data['tracker'][0] if 'tracker' in multimodal_data else None
        speech_transcript = multimodal_data['speech_transcript'][0] if 'speech_transcript' in multimodal_data else None

        if 'image_path' in qa_pair:
            new_image_paths = Path(qa_pair['image_path'])
            image_paths = [new_image_paths]
        elif 'new_image_paths' in qa_pair:
            new_image_paths = [Path(image_path).absolute() for image_path in qa_pair['new_image_paths']]
            if len(new_image_paths) == 0:
                continue
            image_paths = new_image_paths

        # shuffle the image_paths
        shuffle(image_paths)
        # with a random chance, we drop modalities(pc, audio, robot, tracker, speech). Go over them one by one. Also never allow more then 7 images.
        if len(image_paths) > 7:  # randomly take 7
            image_paths = random.sample(image_paths, 7)
        if random.random() < MULTIMODAL_DROP_PROP:
            pc = None
        if random.random() < MULTIMODAL_DROP_PROP:
            audio = None
        if random.random() < MULTIMODAL_DROP_PROP:
            robot_metadata = None
        if random.random() < MULTIMODAL_DROP_PROP:
            tracker_metadata = None
        if random.random() < MULTIMODAL_DROP_PROP:
            speech_transcript = None

        # fetch the scene graph, and decide if to prompt the model to include it or not. we can in any case include it in the json but not as GT.
        if 'relationships' in sample:
            sg = sample['relationships']
            shuffle(sg)
            scene_graph_string = scene_graph_to_string(sg)
        else:
            scene_graph_string = None

        sample_data = apply_template(
            qa_pair, image_paths, scene_graph_string, timepoint=int(frame_id), sample_id=sample_id,
            pc=pc, audio=audio, raw_audio=raw_audio, robot_metadata=robot_metadata,
            tracker_metadata=tracker_metadata, speech_transcript=speech_transcript, segmasks=[], dataset_name=dataset_name, take_name=qa_pair['take_name'])
        samples.append(sample_data)

    return samples


def main():
    ADD_TEMPORAL = False  # TODO set to True to add temporal information, necessary for training the temporal model
    WITH_TEMPORAL_AUG = True
    DROP_HISTORY = 0.5  # either False or float
    QA_PAIRS_COUNT = 1_000_000 if SPLIT == 'train' else 100_000
    MULTIMODAL_DROP_PROP = 0.50
    QA_PAIRS_PATH = f'data/final_qa_pairs_{SPLIT}_{QA_PAIRS_COUNT}.json'
    NAME = f'{SPLIT}_{QA_PAIRS_COUNT}_{ADD_TEMPORAL}temp_{WITH_TEMPORAL_AUG}tempaug_{MULTIMODAL_DROP_PROP}mmdrop_QA'
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    print(f'Creating samples for QWEN dataset with name {NAME}')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='no_gt_image.json', help='configuration file name.')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)

    with open(QA_PAIRS_PATH) as f:
        qa_pairs = json.load(f)

    samples = generate_finetuning_samples(qa_pairs, MULTIMODAL_DROP_PROP)

    shuffle(samples)

    if ADD_TEMPORAL:
        # if no scene graphs are available for that dataset (MVOR, EgoSurgery), use action labels instead.
        all_ref_samples = []
        for key, dataset in DATASETS.items():
            for i in range(len(dataset)):
                ref_sample = dataset[i]
                ref_sample['_dataset_name'] = key
                ref_sample['_take_name'] = ref_sample['sample']['take_name']
                ref_sample['timepoint'] = int(ref_sample['sample']['frame_id'])
                all_ref_samples.append(ref_sample)
        take_to_history = {}
        take_timepoint_to_memory_str = {}
        take_keys = {f'{elem["_dataset_name"]}_{elem["_take_name"]}' for elem in samples}
        for take_key in tqdm(take_keys, desc='Processing takes for temporal information'):
            dataset_name, take_name = take_key.split('_', 1)
            take_key = f'{dataset_name}_{take_name}'
            take_samples = [elem for elem in all_ref_samples if elem['_dataset_name'] == dataset_name and elem['_take_name'] == take_name]
            take_samples = sorted(take_samples, key=lambda x: x['timepoint'])
            take_timepoints_with_ids = [(elem['timepoint'], elem['sample']['sample_id']) for elem in take_samples]
            if dataset_name in ['4D-OR', 'MM-OR']:  # we have scene graphs
                take_scene_graphs = []
                for tsg in take_samples:
                    scene_graph = tsg['sample']['relationships']
                    take_scene_graphs.append({'timepoint_idx': tsg['timepoint'], 'scene_graph': scene_graph})
                surgery_sg = scene_graphs_to_surgery_sg(take_scene_graphs, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
            else:  # use action_labels as pseudo scene graphs
                take_action_labels = []
                for tsg in take_samples:
                    if dataset_name == 'MVOR':
                        action_label = tsg['sample']['action_label']
                    else:
                        action_label = tsg['sample']['phase_label']
                    take_action_labels.append({'timepoint_idx': tsg['timepoint'], 'action_label': action_label})
                surgery_sg = action_labels_to_surgery_sg(take_action_labels)

            with open(f'data/qwen2vl_samples/surgery_sg_{take_key}.json', 'w') as f:
                json.dump(surgery_sg, f)
            with open(f'data/qwen2vl_samples/take_timepoints_with_ids_{take_key}.json', 'w') as f:
                json.dump(take_timepoints_with_ids, f)
            take_to_history[take_key] = surgery_sg

        samples_with_history = []
        for sample in tqdm(samples, desc='Augmenting samples with temporal information'):
            question_type = sample['_question_type']
            if question_type in ['tool_equipment_attribute']:
                # temporality does not make sense, just skip
                samples_with_history.append(sample)
                continue
            dataset_name = sample['_dataset_name']
            take_name = sample['_take_name']
            take_key = f'{dataset_name}_{take_name}'
            surgery_sg = take_to_history[take_key]
            timepoint = sample['timepoint']
            surgery_sg_triplets_up_to_current = [elem for elem in surgery_sg if elem[0] < timepoint]
            if dataset_name in ['4D-OR', 'MM-OR']:
                surgery_sg_to_memory_func = surgery_sg_to_memory_str
            else:
                surgery_sg_to_memory_func = surgery_sg_actions_to_memory_str
            memory_str = surgery_sg_to_memory_func(surgery_sg_triplets_up_to_current, current_timepoint=timepoint)

            if WITH_TEMPORAL_AUG:
                p = random.random()
                if p < 0.5:
                    memory_str = None
                elif p < 0.666:
                    memory_str = surgery_sg_to_memory_func(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='short', DROP_HISTORY=DROP_HISTORY)
                elif p < 0.833:
                    memory_str = surgery_sg_to_memory_func(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='long', DROP_HISTORY=DROP_HISTORY)
                else:
                    memory_str = surgery_sg_to_memory_func(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='longshort', DROP_HISTORY=DROP_HISTORY)

            take_timepoint_to_memory_str[f'{take_key}_{timepoint}'] = memory_str if memory_str else ''
            if memory_str is not None:
                # Inject memory into the user's prompt
                user_msg = sample['messages'][0]['content']
                # find the last image token
                image_token_idx = user_msg.rfind('<image>')
                # Insert after <image>, before the rest. Elegantly.
                start_idx = image_token_idx + len('<image>')
                user_msg = user_msg[:start_idx] + f'<memory_start>: {memory_str}<memory_end>. ' + user_msg[start_idx:]
                sample['messages'][0]['content'] = user_msg

            samples_with_history.append(sample)

        samples = samples_with_history

        with open(f'data/qwen2vl_samples/{NAME}_take_timepoint_to_memory_str.json', 'w') as f:
            json.dump(take_timepoint_to_memory_str, f)

    with open(f'data/qwen2vl_samples/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)


if __name__ == '__main__':
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
