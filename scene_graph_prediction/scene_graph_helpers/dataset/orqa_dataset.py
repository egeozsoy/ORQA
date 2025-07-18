import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from helpers.configurations import TRACKER_OBJECT_MAP
from scene_graph_prediction.data_helpers.scene_graph_converters import scene_graphs_to_surgery_sg, action_labels_to_surgery_sg, surgery_sg_to_memory_str, surgery_sg_actions_to_memory_str
from scene_graph_prediction.scene_graph_helpers.dataset.egosurgery_dataset import EgoSurgeryDataset
from scene_graph_prediction.scene_graph_helpers.dataset.mvor_dataset import MVORDataset
from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset


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


class ORQADataset(Dataset):
    def __init__(self,
                 config,
                 split='train',
                 concise=True,
                 with_sg_grounding=True,
                 only_history_generation=False,
                 only_summary_precomputation=False,
                 eval_size='small'):
        assert split in ['train', 'val', 'test']
        assert eval_size in ['small', 'medium', 'large']
        # small used for continuous development, medium for important validations, large for final evaluations
        self.split = split
        self.concise = concise
        self.with_sg_grounding = with_sg_grounding
        self.temporality = config.get('temporality', None)
        if only_history_generation:
            print('Only generating history')
            QA_PAIRS_PATH = f'data/final_qa_pairs_temporal_history_precomputation_{split}.json'
        elif only_summary_precomputation:
            print('Only generating summary context precomputation')
            QA_PAIRS_PATH = f'data/final_qa_pairs_summary_context_precomputation_{split}_maxcontext_200.json'
        else:
            if eval_size == 'small':
                QA_PAIRS_COUNT = 1_000
            elif eval_size == 'medium':
                QA_PAIRS_COUNT = 10_000
            else:
                QA_PAIRS_COUNT = 100_000
            print(f'Using evaluation size {eval_size} with {QA_PAIRS_COUNT} QA pairs')
            QA_PAIRS_PATH = f'data/final_qa_pairs_{split}_{QA_PAIRS_COUNT}.json'

        OR4D_DATASET = ORDataset({'USE_VIS_DESC': False}, split, load_4dor=True, load_mmor=False)
        OR4D_DATASET_TRAIN = ORDataset({'USE_VIS_DESC': False}, 'train', load_4dor=True, load_mmor=False)  # needed for synthetic
        MMOR_DATASET = ORDataset({'USE_VIS_DESC': False}, split, load_4dor=False, load_mmor=True)
        MMOR_DATASET_TRAIN = ORDataset({'USE_VIS_DESC': False}, 'train', load_4dor=False, load_mmor=True)  # needed for synthetic
        MVOR_DATASET = MVORDataset(split)
        EGOSURGERY_DATASET = EgoSurgeryDataset(split)
        self.datasets = {'4D-OR': OR4D_DATASET, 'MM-OR': MMOR_DATASET, 'MVOR': MVOR_DATASET, 'EgoSurgery': EGOSURGERY_DATASET,
                         '4D-OR_train': OR4D_DATASET_TRAIN, 'MM-OR_train': MMOR_DATASET_TRAIN}

        # Load the QA pairs
        with open(QA_PAIRS_PATH, 'r') as f:
            raw_qa_pairs = json.load(f)

        # Preprocess any 'tool_equipment_attribute' pairs to ensure take_name/frame_id are set
        for qa_pair in raw_qa_pairs:
            if qa_pair.get('_question_type') == 'tool_equipment_attribute':
                if '_' in qa_pair["frame_id"]:  # was wrong, fix those cases.
                    take_name, frame_id = qa_pair["frame_id"].rsplit('_', 1)
                    qa_pair["take_name"] = take_name
                    qa_pair["frame_id"] = frame_id
                # dataset_type should be appended with '_train' for synthetic data
                qa_pair['dataset'] = f'{qa_pair["dataset"]}_train'

        # Filter samples here
        self.samples = self._filter_invalid_samples(raw_qa_pairs)
        self.sample_to_memory_str = {}
        if self.temporality is not None:
            take_to_history = {}
            if self.temporality == 'GT':
                print('Using GT temporality')
                take_keys = {f'{elem["dataset"]}_{elem["take_name"]}' for elem in self.samples if '_train' not in elem['dataset']}
                for take_key in tqdm(take_keys, desc='Processing takes for temporal information'):
                    with open(f'data/qwen2vl_samples/surgery_sg_{take_key}.json', 'r') as f:
                        surgery_sg = json.load(f)
                    take_to_history[take_key] = surgery_sg
            elif self.temporality == 'PRED':
                print('Using PRED temporality')
                history_save_path = Path(f'data/pred_history/qwen2vl_lora_sft_qlora_1000000_unfreeze8_0.5mmdrop_336res_578imgtoks_{split}.json')
                assert history_save_path.exists(), f'History file not found at {history_save_path}'
                with open(history_save_path, 'r') as f:
                    history = json.load(f)
                all_ref_samples = []
                for key, dataset in self.datasets.items():
                    if '_train' in key:
                        continue
                    for i in range(len(dataset)):
                        ref_sample = dataset[i]
                        ref_sample['_dataset_name'] = key
                        ref_sample['_take_name'] = ref_sample['sample']['take_name']
                        ref_sample['timepoint'] = int(ref_sample['sample']['frame_id'])
                        sample_id = ref_sample['sample']['sample_id']
                        # modify its scene graph or action label here. or phase label
                        try:
                            if key in ['4D-OR', 'MM-OR']:
                                ref_sample['sample']['relationships'] = history[sample_id]
                            elif key == 'MVOR':
                                ref_sample['sample']['action_label'] = history[sample_id]
                            else:
                                ref_sample['sample']['phase_label'] = history[sample_id]
                            all_ref_samples.append(ref_sample)
                        except KeyError:
                            print(f'KeyError for sample {sample_id} in dataset {key}')
                take_keys = {f'{elem["dataset"]}_{elem["take_name"]}' for elem in self.samples if '_train' not in elem['dataset']}
                for take_key in tqdm(take_keys, desc='Processing takes for temporal information'):
                    dataset_name, take_name = take_key.split('_', 1)

                    take_samples = [elem for elem in all_ref_samples if elem['_dataset_name'] == dataset_name and elem['_take_name'] == take_name]
                    take_samples = sorted(take_samples, key=lambda x: x['timepoint'])
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
                    take_to_history[take_key] = surgery_sg

            for sample in tqdm(self.samples, desc='Augmenting samples with temporal information'):
                question_type = sample['_question_type']
                if question_type in ['tool_equipment_attribute']:
                    # temporality does not make sense, just skip
                    continue
                dataset_name = sample['dataset'].replace('_train', '')
                take_name = sample['take_name']
                take_key = f'{dataset_name}_{take_name}'
                surgery_sg = take_to_history[take_key]
                timepoint = int(sample['frame_id'])
                surgery_sg_triplets_up_to_current = [elem for elem in surgery_sg if elem[0] < timepoint]
                if dataset_name in ['4D-OR', 'MM-OR']:
                    surgery_sg_to_memory_func = surgery_sg_to_memory_str
                else:
                    surgery_sg_to_memory_func = surgery_sg_actions_to_memory_str
                memory_str = surgery_sg_to_memory_func(surgery_sg_triplets_up_to_current, current_timepoint=timepoint)
                self.sample_to_memory_str[f'{take_key}_{timepoint}'] = memory_str if memory_str else ''
        else:
            print('Not using temporality')

    def __len__(self):
        return len(self.samples)

    def _filter_invalid_samples(self, qa_pairs):
        def is_valid_sample(qa_pair):
            if 'new_image_paths' in qa_pair and len(qa_pair['new_image_paths']) == 0:
                # print(f'Missing image paths for sample {qa_pair["take_name"]}_{qa_pair["frame_id"]}')
                return False
            # Uncomment and modify the following block if needed for additional validation
            # dataset = self.datasets[qa_pair['dataset']]
            # sample = dataset.get_sample_by_key(f"{qa_pair['take_name"]}_{qa_pair['frame_id']}")
            # multimodal_data = sample['multimodal_data']
            # multimodal_data_len = max([len(multimodal_data[key]) for key in multimodal_data])
            # if multimodal_data_len == 0:
            #     return False
            return True

        # Use a list comprehension to filter valid samples
        filtered_samples = [qa_pair for qa_pair in tqdm(qa_pairs, desc="Filtering invalid samples") if is_valid_sample(qa_pair)]
        return filtered_samples

    def apply_template(self, qa_pair, image_paths, scene_graph, timepoint, sample_id, pc, audio, robot_metadata, tracker_metadata, speech_transcript, segmasks, dataset_name, take_name,
                       memory_str=None):
        human_prompt = qa_pair['question']
        if self.concise:
            if 'Answer concisely' not in human_prompt:
                human_prompt += ' Answer concisely.'
        if self.with_sg_grounding:  # ground its thinking to the scene graphs (essentially ask it to predict the scene graph before the answer).No n
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
                    "content": ''
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

        sample['_question'] = qa_pair['_question']
        sample['_answer'] = qa_pair['_answer']
        sample['question'] = qa_pair['question']
        sample['answer'] = qa_pair['answer']
        sample['_question_type'] = qa_pair['_question_type']
        sample['_sg'] = scene_graph
        sample['_dataset_name'] = dataset_name
        sample['_take_name'] = take_name
        if memory_str is not None and (self.temporality == 'GT' or self.temporality == 'PRED'):
            # Inject memory into the user's prompt
            user_msg = sample['messages'][0]['content']
            # find the last image token
            image_token_idx = user_msg.rfind('<image>')
            # Insert after <image>, before the rest. Elegantly.
            start_idx = image_token_idx + len('<image>')
            user_msg = user_msg[:start_idx] + f'<memory_start>: {memory_str}<memory_end>. ' + user_msg[start_idx:]
            sample['messages'][0]['content'] = user_msg
        return sample

    def __getitem__(self, index):
        qa_pair = self.samples[index]
        # first we determine the dataset and fetch that dataset
        dataset_name = qa_pair['dataset']
        dataset = self.datasets[dataset_name]

        sample_key = f'{qa_pair["take_name"]}_{qa_pair["frame_id"]}'
        # fetch all the info about that sample, including all the modalities
        sample = dataset.get_sample_by_key(sample_key)
        sample, multimodal_data = sample['sample'], deepcopy(sample['multimodal_data'])
        sample_id = sample['sample_id']
        frame_id = sample['frame_id']
        timepoint = int(frame_id)
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
        else:  # use simstation views only as backup, if azure is not available
            if len(simstation_image_paths) > 0:
                simstation_image_paths = [simstation_image_paths[view_idx] for view_idx in simstation_views_to_use]
                image_paths.extend(simstation_image_paths)

        # add robot screen regardless (simstation view idx=1)
        if len(simstation_image_paths) > 1:
            image_paths.append(simstation_image_paths[1])
        # add trackercam regardless
        if len(trackercam_image_paths) > 0:
            image_paths.append(trackercam_image_paths[0])

        pc = multimodal_data['pc'][0] if 'pc' in multimodal_data else None
        # segmasks = multimodal_data['segmasks'] if 'segmasks' in multimodal_data else []
        audio = multimodal_data['audio'][0] if 'audio' in multimodal_data else None
        robot_metadata = multimodal_data['robot_metadata'][0] if 'robot_metadata' in multimodal_data else None
        tracker_metadata = multimodal_data['tracker'][0] if 'tracker' in multimodal_data else None
        speech_transcript = multimodal_data['speech_transcript'][0] if 'speech_transcript' in multimodal_data else None
        memory_str = self.sample_to_memory_str.get(f'{dataset_name}_{qa_pair["take_name"]}_{timepoint}', None)

        if 'image_path' in qa_pair:
            new_image_paths = Path(qa_pair['image_path'])
            image_paths = [new_image_paths]
        elif 'new_image_paths' in qa_pair:
            new_image_paths = [Path(image_path) for image_path in qa_pair['new_image_paths']]
            image_paths = new_image_paths

        # fetch the scene graph, and decide if to prompt the model to include it or not. we can in any case include it in the json but not as GT.
        if 'relationships' in sample:
            sg = sample['relationships']
            scene_graph_string = scene_graph_to_string(sg)
        else:
            scene_graph_string = None

        sample_data = self.apply_template(
            qa_pair, image_paths, scene_graph_string, timepoint=int(frame_id), sample_id=sample_id,
            pc=pc, audio=audio, robot_metadata=robot_metadata,
            tracker_metadata=tracker_metadata, speech_transcript=speech_transcript, segmasks=[], dataset_name=dataset_name, take_name=qa_pair['take_name'], memory_str=memory_str)

        return sample_data


if __name__ == '__main__':
    dataset = ORQADataset({}, split='train', concise=True, with_sg_grounding=True)
    sample = dataset[0]
    print(sample)
    a = 1
