import random
import warnings
from pathlib import Path
from random import shuffle

import cv2
import json_tricks as json  # Allows to load integers etc. correctly
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from helpers.configurations import MMOR_DATA_ROOT_PATH, TRACKER_OBJECT_MAP, OR_4D_DATA_ROOT_PATH, OR4D_TAKE_NAME_TO_FOLDER, MMOR_TAKE_NAME_TO_FOLDER
from scene_graph_prediction.scene_graph_helpers.dataset.egosurgery_dataset import EgoSurgeryDataset
from scene_graph_prediction.scene_graph_helpers.dataset.mvor_dataset import MVORDataset
from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.dataset.orqa_dataset import scene_graph_to_string

TRACK_TO_METAINFO = {
    'instrument_table': {'color': (255, 51, 153), 'label': 1},
    'ae': {'color': (0, 0, 255), 'label': 2},
    'ot': {'color': (255, 255, 0), 'label': 3},
    'mps_station': {'color': (133, 0, 133), 'label': 4},
    'patient': {'color': (255, 0, 0), 'label': 5},
    'drape': {'color': (183, 91, 255), 'label': 6},
    'anest': {'color': (177, 255, 110), 'label': 7},
    'circulator': {'color': (255, 128, 0), 'label': 8},
    'assistant_surgeon': {'color': (116, 166, 116), 'label': 9},
    'head_surgeon': {'color': (76, 161, 245), 'label': 10},
    'mps': {'color': (125, 100, 25), 'label': 11},
    'nurse': {'color': (128, 255, 0), 'label': 12},
    'drill': {'color': (0, 255, 128), 'label': 13},  # Changed
    'hammer': {'color': (204, 0, 0), 'label': 15},
    'saw': {'color': (0, 255, 234), 'label': 16},
    'tracker': {'color': (255, 128, 128), 'label': 17},  # Changed
    'mako_robot': {'color': (60, 75, 255), 'label': 18},  # Changed
    'monitor': {'color': (255, 255, 128), 'label': 24},  # Changed
    'c_arm': {'color': (0, 204, 128), 'label': 25},  # Changed
    'unrelated_person': {'color': (255, 255, 255), 'label': 26},
    'student': {'color': (162, 232, 108), 'label': 27},
    'secondary_table': {'color': (153, 0, 153), 'label': 28},
    'cementer': {'color': (153, 76, 0), 'label': 29},
    '__background__': {'color': (0, 0, 0), 'label': 0}
}
# sorted classes by their label
sorted_classes = sorted(TRACK_TO_METAINFO.keys(), key=lambda x: TRACK_TO_METAINFO[x]['label'])
label_to_category_id = {TRACK_TO_METAINFO[track]['label']: i for i, track in enumerate(sorted_classes)}  # 0 is background
color_to_category_id = {TRACK_TO_METAINFO[track]['color']: i for i, track in enumerate(sorted_classes)}  # 0 is background
for key, value in TRACK_TO_METAINFO.items():
    c = value['color']
    segment_id = c[0] + c[1] * 256 + c[2] * 256 * 256
    value['segment_id'] = segment_id

warnings.filterwarnings('ignore')
'''
Questions per Dataset:
Automatically create synomyns for the same question. Maybe rule based, maybe with the QWEN model.
If scene graph available, the network should first generate the scene graph and then answer the question. Sometimes. Sometimes directly. Sometimes the generated scene graph should be wrong in ground truth to confuse the model.

MM-OR:
    - How many people are in the OR? (count)
    - Who is in the OR? (role)
    - What is the interaction between X and Y? (scene graph based)
    - What is the color/shape (5 attributes) of the tool/equipment(using the synthetic dataset) Both Mmor and 4dor synthetic_dataset_generation/synthetic_mv_hybridor
    - How long till the next step/certain step? (time)
    - What is currently happening? (main action of now)
    - Is base array visible? (yes/no) From tracker data
    - Is robot calibrated? (yes/no) From tracker data, or monitor
    - Is sterility breach happening. IF yes what?
    - What is the next step for the robot? (action)
    - Why is X being done? (reasoning, kind of zero-shot)
    - Where is X given image 2D, or directly in 3D? (use segmentation masks, either projected or not projected)
    - How far is X from Y? (distance, 3D)
    - Which tools are currently being used? – Derive from robot logs, tracker data, or segmented views and sg

4D-OR:
    - How many people are in the OR? (count)
    - Who is in the OR? (role)
    - What is the interaction between X and Y? (scene graph based)
    - What is the color/shape (5 attributes) of the tool/equipment(using the synthetic dataset) both mmor and 4dor synthetic_dataset_generation/synthetic_mv_hybridor
    - How long till the next step/certain step? (time)
    - What is currently happening? (main action of now)
    - Why is X being done? (reasoning, kind of zero-shot)
    - Where is X given image 2D, or directly in 3D? (use segmentation masks, either projected or not projected)
    - How far is X from Y? (distance, 3D)
    - Which tools are currently being used? – Derive from segmented views and sg

MVOR: https://github.com/CAMMA-public/MVOR 
    - How many people are in the OR? (count)
    - Who is in the OR? (role) IF WE CAN GET IT. Otherwise only differentiate between clinican and patient.
    - How long till the next step/certain step? (time)
    - What is currently happening? (main action of now). Custom annotations from back in IDP days.
    - Where is PERSON given image 2D, or directly in 3D? (use human poses masks, either projected or not projected). Because we can not differentiate, we might need the model to output all poses at once.
    - How far is PERSON from PERSON? (distance, 3D)

EgoSurgery:
    - Which tools are in the OR?
    - How long till the next step/certain step? (time)
    - What is currently happening? (main action of now) EITHER PHASE or Tool.
    - Why is X being done? (reasoning, kind of zero-shot)
    - Where is X(tool) given image 2D? (use bounding boxes)
    - How far is X from Y? (distance, 2D)
    - Where is the surgeon looking at?

'''

QUESTIONS_TO_BASE_FORMULATION = {
    'count_people': 'How many people are in the OR?',
    'role_people': 'Who is in the OR?',
    'interaction': 'What is the interaction of {} with {}?',
    'tool_equipment_attribute': 'What is the {} of the {}?',
    'time_until_step': 'How long until {}?',
    'status_action': 'What is the status of {}?',
    'current_action': 'What is currently happening?',
    'did_happen': 'Did {} happen?',
    'is_base_array_visible': 'Is the base array visible?',
    'is_robot_calibrated': 'Is the robot calibrated?',
    'sterility_breach': 'Is there a sterility breach happening?',
    'current_robot_step': 'What is the current step of the robot?',
    'next_robot_step': 'What is the next step for the robot?',
    # 'why_action': 'Why is {} being done?', # this is a rather stupid thing because it is just hallucinated. I hope to not need this.
    'where_2d': 'Where is {} in the image?',
    'where_3d': 'Where is {} in the scene?',
    'distance_3d': 'How far is {} from {}?',
    'tools_used': 'Which tools are currently being used?',
    'current_scene_graph': 'What is the current scene graph?',
    'list_all_entities': 'List all entities in the OR.',
    'list_all_entities_ordered_2D': 'List all entities in the OR from left to right.',
    'gaze_location': 'Where is the surgeon looking at?',
    'gaze_object': 'What is the surgeon looking at?',
    'monitor_reading': 'Summarize the information on the monitor.',
}

SYNONYM_QUESTIONS = {
    'count_people': [
        "How many individuals are present in the OR?",
        "What is the number of persons currently in the OR?",
        "Can you count how many people are inside the OR?",
        "Give me the total number of people inside the OR.",
        "Please provide the headcount of individuals in the OR.",
        "Could you enumerate the persons in the OR?",
        "How many people can be found in the OR?",
    ],
    'role_people': [
        "Who are the individuals present in the OR?",
        "Identify the persons currently in the OR.",
        "List the people who are inside the OR.",
        "Who is there in the OR right now?",
        "Can you name the persons present in the OR?",
        "Tell me the roles of people currently in the OR.",
        "Provide the identities of those in the OR.",
        "Give me the list of people who are in the OR.",
    ],
    'interaction': [
        "What kind of interaction is taking place between {} and {}?",
        "How are {} and {} interacting?",
        "Describe the interaction involving {} and {}.",
        "What is happening between {} and {}?",
        "How do {} and {} relate to each other right now?",
        "Can you explain the relationship between {} and {} at this moment?",
        "How are {} and {} connected in this scenario?",
        "In what way are {} and {} interacting?",
        "How would you describe the link between {} and {}?",
    ],
    'tool_equipment_attribute': [
        "What is the {} property of the {}?",
        "Can you specify the {} of the {}?",
        "Provide the {} of the {}.",
        "What {} does the {} possess?",
        "Tell me the {} of the {}.",
        "Describe the {} that characterizes the {}.",
        "What {} attribute belongs to the {}?",
        "What is the {} associated with the {}?",
        "State the {} for the {}.",
        "What can you say about the {} of the {}?",
        "Give me information on the {} of the {}.",
    ],
    'time_until_step': [
        "How long until {} takes place?",
        "What is the remaining time before {} occurs?",
        "How much time is left before {}?",
        "When can we expect {} to occur?",
        "How long do we have before {}?",
        "Please estimate the time left for {}.",
        "How much time remains until {} starts?",
    ],
    'status_action': [
        "What is the progress status of {}?",
        "How is {} progressing?",
        "What is the current state of {}?",
        "Can you update me on the status of {}?",
        "What stage is {} currently at?",
        "What is the present condition of {}?",
        "In what state is {} right now?",
        "What is the current progress of {}?",
        "How far along are we with {}?"
    ],
    'current_action': [
        "What is happening right now?",
        "Describe the activity currently taking place.",
        "What's going on at this moment?",
        "What is the ongoing action at present?",
        "What action is being performed at this time?",
        "Can you describe the present activity?",
        "What's ongoing in the OR at this instant?",
    ],
    'did_happen': [
        "Did {} occur?",
        "Was {} done?",
        "Has {} happened?",
        "Did {} take place?",
        "Was {} completed?",
        "Has {} been executed?",
    ],
    'is_base_array_visible': [
        "Is the base array currently visible?",
        "Can we see the base array?",
        "Is the base array in sight?",
        "Is the base array detectable right now?",
        "Is the base array clearly visible?",
    ],
    'is_robot_calibrated': [
        "Has the robot been calibrated?",
        "Is the robot in a calibrated state?",
        "Is the robot calibration complete?",
        "Is the robot properly calibrated?",
        "Has the robot calibration process finished?",
        "Is the robot's calibration done?",
    ],
    'sterility_breach': [
        "Is there any breach of sterility?",
        "Has sterility been compromised?",
        "Is sterility broken at the moment?",
        "Is a sterility breach occurring?",
        "Can we detect a sterility violation?",
        "Is sterility currently intact?",
        "Is the sterile field compromised?",
        "Is there evidence of sterility break?",
    ],
    'next_robot_step': [
        "What step will be performed next with the robot?",
        "Which action will be taken next using the robot?",
        "What is the next planned operation involving the robot?",
        "What will be the next activity conducted with the robot?",
        "What is the subsequent action to be done with the robot?",
        "Can you tell me the next step planned with the robot?",
    ],
    'current_robot_step': [
        "What is the current step being performed by the robot?",
        "What is the robot currently doing?",
        "What action is the robot currently engaged in?",
        "What is the present operation of the robot?",
    ],
    'why_action': [
        "For what reason is {} being done?",
        "Why is {} taking place?",
        "Why are we performing {}?",
        "Can you explain the purpose of {}?",
        "Why has {} been initiated?",
        "Why do we need {}?",
        "Why is {} necessary?",
        "Why is {} required?",
        "Why is {} important?"
    ],  # this is a rather stupid thing because it is just hallucinated. I hope to not need this.
    'where_2d': [
        "Where is {} located in the image?",
        "Where can I find {} in this image?",
        "What is the position of {} in the image?",
        "Locate {} within the image.",
        "Where does {} appear in this image?",
        "Indicate {}'s location on the image.",
        "Identify where {} is in the image.",
    ],
    'where_3d': [
        "Where is {} located in the scene?",
        "In the OR, where can {} be found?",
        "What is the position of {} in the scene?",
        "Where exactly is {} situated in the OR?",
        "Locate {} within the OR.",
        "Where does {} appear in the 3D environment?",
        "Can you specify the 3D coordinates of {}?",
    ],
    'distance_3d': [
        "How far apart are {} and {}?",
        "What is the distance between {} and {}?",
        "Can you measure how far {} is from {}?",
        "What is the spatial separation of {} and {}?",
        "What's the distance between {} and {} in the scene?",
        "What is the 3D distance from {} to {}?",
        "How far is {} located from {}?",
    ],
    'tools_used': [
        "Which tools are in use right now?",
        "What tools are currently being utilized?",
        "Can you list the tools that are being used?",
        "Which instruments are presently employed?",
        "What tools are at work at the moment?",
        "Identify the tools that are in operation now.",
        "What tools are involved in the ongoing activity?",
        "Name the tools currently in action.",
        "Which instruments can we see being used?",
    ],
    'current_scene_graph': [
        "What does the current scene graph look like?",
        "Provide the current scene graph.",
        "Can you describe the current scene graph?",
        "Show me the present scene graph.",
        "Give the scene graph representation right now.",
        "Describe the entities and relations in the current scene.",
        "What's the scene graph at this moment?",
    ],
    'list_all_entities': [
        "Which entities are in the OR?",
        "List every entity present.",
        "Can you name all entities?",
        "Enumerate all the entities in the OR.",
        "What are all the entities found in the OR?",
        "Identify all entities visible in the OR.",
    ],
    'list_all_entities_ordered_2D': [
        "List all entities in the OR from left to right.",
        "Can you order all entities by their left-to-right position?",
        "Provide entities arranged from the leftmost to the rightmost in the image.",
        "What are all the entities in the image?",
        "Enumerate the entities in the OR in horizontal order.",
        "Which entities appear from left to right in the image?",
        "Can you list the OR entities by their horizontal position?"
    ],
    'gaze_location': [
        "Where is the surgeon looking?",
        "In which direction is the surgeon's gaze?",
        "Where are the surgeon's eyes focused?",
        "Where is the surgeon directing their gaze?",
    ],
    'gaze_object': [
        "What is the surgeon looking at?",
        "At what is the surgeon currently looking?",
        "What is the surgeon visually attending to?",
        "What is the surgeon observing right now?",
        "Which object is the surgeon looking at?",

    ],
    'monitor_reading': [
        "Summarize the information displayed on the monitor.",
        "What does the monitor currently show?",
        "Can you describe the data on the monitor?",
        "What readings appear on the monitor?",
        "What information is presented on the monitor screen?",
        "Give me a summary of what's on the monitor.",
        "What details does the monitor provide?"
    ]
}

ANSWER_TEMPLATES = {
    'count_people': [
        "There are {} people in the OR.",
        "I see {} individuals in the OR.",
        "The total number of people in the OR is {}.",
        "{} people are currently present in the OR.",
        "At the moment, {} persons occupy the OR.",
        "There appear to be {} people.",
    ],
    'role_people': [
        "Currently, the OR is occupied by {}.",
        "In the OR, you have {}.",
        "The people inside the OR are {}.",
        "I can see {} in the OR.",
        "Those present in the OR are {}.",
        "The OR's occupants are {}.",
        "The OR currently has {}.",
    ],
    'interaction': [
        "The interaction between {} and {} is {}.",
        "{} and {} have an interaction of {}.",
        "Between {} and {}, there is {}.",
    ],
    'tool_equipment_attribute': [
        "The {} of the {} is {}.",
        "The {} associated with the {} is {}.",
    ],
    'time_until_step': [
        "There's about {} until the next step.",
        "We have {} remaining before it occurs.",
        "The time left is approximately {}.",
        "It will happen in about {}.",
        "It will start in {}.",
        "{} remains until that step is taken.",
        "It starts in about {}.",
    ],
    'status_action': [
        "Currently, {} is {}.",
        "The status of {} is {}.",
    ],
    'current_action': [
        "They are currently {}.",
        "Right now, the activity is {}.",
        "At this moment, they are {}.",
        "We can see {} happening right now.",
        "The current action: {}.",
        "They're performing {} at this time.",
    ],
    'did_happen': [
        lambda x: "{} has happened." if x == "yes" else "{} has not happened.",
        lambda x: "{} is done." if x == "yes" else "{} is not done.",
        lambda x: "{} occurred." if x == "yes" else "{} did not occur.",
        lambda x: "{} is completed." if x == "yes" else "{} is not completed.",
        lambda x: "{} has taken place." if x == "yes" else "{} has not taken place.",
    ],
    'is_base_array_visible': [
        lambda x: "Yes, the base array is visible." if x == "yes" else "No, the base array is not visible.",
        lambda x: "The base array can be observed." if x == "yes" else "The base array can't be observed.",
        lambda x: "The base array can be seen." if x == "yes" else "The base array can't be seen.",
        lambda x: "It is visible." if x == "yes" else "It is not visible.",
    ],
    'is_robot_calibrated': [
        lambda x: "Yes, the robot is calibrated." if x == "yes" else "No, the robot is not calibrated.",
        lambda x: "Calibration is achieved." if x == "yes" else "Calibration is not achieved.",
        lambda x: "Yes, calibration complete." if x == "yes" else "No, calibration incomplete.",
        lambda x: "Successfully calibrated." if x == "yes" else "Unsuccessful calibration.",
    ],
    'sterility_breach': [
        lambda x: "There is a sterility breach." if x == "yes" else "No sterility breach is detected.",
        lambda x: "Sterility has been compromised." if x == "yes" else "Sterility remains intact.",
        lambda x: "I see signs of a breach in sterility." if x == "yes" else "I see no sign of sterility breach.",
        lambda x: "Yes, sterility is broken." if x == "yes" else "No, sterility is maintained.",
    ],
    'next_robot_step': [
        "The next operation with the robot is {}.",
        "The robot is going to {} next.",
        "The subsequent procedure with the robot: {}.",
        "In the following step, the robot will be {}."
    ],
    'current_robot_step': [
        "The robot is currently {}.",
        "The robot is {} at the moment.",
        "The robot is performing {} right now.",
        "The robot is engaged in {}.",
        "The robot is currently in the {} phase.",
        "The robot is in the process of {}.",
    ],
    'why_action': [  # this is a rather stupid thing because it is just hallucinated. I hope to not need this.
        "{} is done because {}.",
        "The reason for {} is {}.",
        "We perform {} so that {}.",
        "The purpose of {} is {}."
    ],
    'where_2d': [
        "{} is located {} in the image.",
        "You can find {} in {}.",
        "{} appears {} on the image.",
        "In the image, {} is positioned {}.",
        "The position of {} is {} in the image."
    ],
    'where_3d': [
        "{} is located {} in the scene.",
        "{} is positioned in {}.",
        "{} can be found in {} in the 3D environment.",
        "In three dimensions, {} is in {}.",
    ],
    'distance_3d': [
        "The distance between is {}.",
        "They are {} apart.",
        "The 3D gap between them measures {}.",
        "Their distance is {}."
    ],
    'tools_used': [
        "The tools currently in use are {}.",
        "I see {} being utilized.",
        "At the moment, {} are employed.",
        "They are using {} right now.",
        "The tools in action: {}."
    ],
    'current_scene_graph': [
        "The current scene graph is: {}",
        "Right now, the scene graph is {}.",
        "We have the following scene graph: {}.",
        "The present scene graph representation: {}.",
    ],
    'list_all_entities': [
        "All entities present are {}.",
        "You have these entities: {}.",
        "The OR includes: {}.",
        "We can list the entities as {}.",
        "All entities I see: {}.",
        "Current entities are {}.",
        "All recognized entities: {}.",
        "Here are the entities: {}.",
        "The OR has {}.",
    ],
    'list_all_entities_ordered_2D': [
        "From left to right, the entities are {}.",
        "In left-to-right order: {}.",
        "Viewed from the left side first, we see {}.",
        "Across the image from left to right: {}.",
    ],
    'gaze_location': [
        "The surgeon is looking at {}.",
        "Currently, the surgeon looks at {}.",
        "The surgeon glances at {}."
    ],
    'gaze_object': [
        "It seems the surgeon's gaze is on {}.",
        "I believe the surgeon focuses on {}.",
        "Their eyes are directed towards {}.",
        "The surgeon's attention is on {}.",
    ],
    'monitor_reading': [
        "{}",
    ]
}

ALLOWED_QUESTIONS_PER_DATASET = {

    '4D-OR': [
        'count_people', 'role_people', 'interaction', 'tool_equipment_attribute', 'time_until_step', 'status_action', 'current_action', 'did_happen',
        'where_2d', 'tools_used', 'current_scene_graph', 'list_all_entities', 'list_all_entities_ordered_2D'],  # we could do where_3D and distance_3D also here.
    'MM-OR': [
        'count_people', 'role_people', 'interaction', 'tool_equipment_attribute', 'time_until_step', 'status_action', 'current_action', 'did_happen',
        'is_base_array_visible', 'is_robot_calibrated', 'sterility_breach', 'next_robot_step', 'current_robot_step', 'where_2d', 'where_3d', 'distance_3d',
        'tools_used', 'current_scene_graph', 'list_all_entities', 'list_all_entities_ordered_2D', 'monitor_reading'],
    'MVOR': [
        'count_people', 'role_people', 'current_action', 'where_2d', 'where_3d', 'distance_3d', 'list_all_entities', 'list_all_entities_ordered_2D'],
    'EgoSurgery': [
        'time_until_step', 'status_action', 'current_action', 'did_happen', 'where_2d', 'tools_used', 'list_all_entities', 'list_all_entities_ordered_2D', 'gaze_location', 'gaze_object'],
}

SPLIT = 'train'
OR4D_DATASET = ORDataset({'USE_VIS_DESC': False}, SPLIT, load_4dor=True, load_mmor=False)
MMOR_DATASET = ORDataset({'USE_VIS_DESC': False}, SPLIT, load_4dor=False, load_mmor=True)
MVOR_DATASET = MVORDataset(SPLIT)
EGOSURGERY_DATASET = EgoSurgeryDataset(SPLIT)
SYNTHETIC_DATASET_PATH = Path(f'synthetic_dataset_generation/synthetic_mv_hybridor')
ALL_SYNTHETIC_JSONS = list(SYNTHETIC_DATASET_PATH.glob('*.json'))

DATASETS = {'4D-OR': OR4D_DATASET, 'MM-OR': MMOR_DATASET, 'MVOR': MVOR_DATASET, 'EgoSurgery': EGOSURGERY_DATASET}
# Custom weights: MM-OR is the biggest, EgoSurgery close second, 4D-OR is 1/3 of MM-OR and MVOR is tiny. But we still want 10% MVOR
DATASET_WEIGHTS = {'4D-OR': 0.20, 'MM-OR': 0.40, 'MVOR': 0.10, 'EgoSurgery': 0.30}  # TOOD maybe adjust?
DATASET_NAMES_LIST = []
DATASET_WEIGHTS_LIST = []
for dataset_name, weight in DATASET_WEIGHTS.items():
    DATASET_NAMES_LIST.append(dataset_name)
    DATASET_WEIGHTS_LIST.append(weight)


def _get_entities_from_samples(dataset_name, sample, multimodal_data, entities_of_interest, threshold=10):
    entities = set()
    if dataset_name in ['4D-OR', 'MM-OR']:
        if 'segmasks' not in multimodal_data:
            return entities
        for segmask_path in multimodal_data['segmasks']:
            segmask = Image.open(segmask_path)
            unique_labels, count = np.unique(np.asarray(segmask), return_counts=True)
            count = count.tolist()
            for label, c in zip(unique_labels, count):
                if label == 0:
                    continue
                if c < threshold:
                    continue
                category_id = label_to_category_id[label]
                category_name = sorted_classes[category_id]
                if entities_of_interest is not None and category_name not in entities_of_interest:
                    continue
                category_name = 'anaesthetist' if category_name == 'anest' else category_name
                category_name = 'operating_table' if category_name == 'ot' else category_name
                category_name = 'anesthesia_equipment' if category_name == 'ae' else category_name
                category_name = category_name.replace('_', ' ')
                entities.add(category_name)
    elif dataset_name == 'MVOR':
        for human_idx, human_pose_3d in enumerate(sample['human_poses_3d']):
            category_name = f'human {human_idx + 1}'
            if entities_of_interest is None or category_name in entities_of_interest:
                entities.add(category_name)
        patient_pose_3d = sample['patient_pose_3d']
        if patient_pose_3d is not None and (entities_of_interest is None or 'patient' in entities_of_interest):
            entities.add('patient')
        for object_name, object_pose_3d in sample['object_poses_3d'].items():
            object_name = object_name.replace('_', ' ')
            if entities_of_interest is None or object_name in entities_of_interest:
                entities.add(object_name)
    else:
        raise NotImplementedError
    return entities


def _get_roles_from_sample(sample, multimodal_data, dataset_name):
    if dataset_name in ['4D-OR', 'MM-OR']:
        # open all the segmentation masks, extract all the human labels > 1000, count this.
        roles = _get_entities_from_samples(dataset_name, sample, multimodal_data,
                                           {'patient', 'nurse', 'assistant_surgeon', 'head_surgeon', 'circulator', 'anest', 'unrelated_person', 'student', 'mps'})
        return roles
    elif dataset_name == 'MVOR':
        roles = _get_entities_from_samples(dataset_name, sample, multimodal_data, {'human 1', 'human 2', 'human 3', 'human 4', 'human 5', 'patient'})
        return roles
    else:
        raise NotImplementedError


def _get_current_actions(sg):
    current_actions = set()
    for sub, obj, rel in sg:
        rel = rel.lower()
        if rel in ['closeto', 'lyingon', 'holding', 'touching', 'assisting']:  # discard these
            continue
        current_actions.add(rel)
    return current_actions


def _get_qa_formatting(question_type, answer, answer_form, question_format_args=None, answer_format_args=None, answer_call_args=None):
    question_formulations = [QUESTIONS_TO_BASE_FORMULATION[question_type]] + SYNONYM_QUESTIONS[question_type]
    question = QUESTIONS_TO_BASE_FORMULATION[question_type]
    question_formulation = random.choice(question_formulations)
    if question_format_args is not None:
        question = question.format(*question_format_args)
        question_formulation = question_formulation.format(*question_format_args)
    if answer_form == 'concise':
        question_formulation = f'{question_formulation} Answer concisely.'
        answer_formulation = answer
    else:
        if answer_call_args is not None:
            answer_formulation = random.choice(ANSWER_TEMPLATES[question_type])(*answer_call_args)
            if answer_format_args is not None:
                answer_formulation = answer_formulation.format(*answer_format_args)
        elif answer_format_args is not None:
            answer_formulation = random.choice(ANSWER_TEMPLATES[question_type]).format(*answer_format_args)
        else:
            answer_formulation = random.choice(ANSWER_TEMPLATES[question_type]).format(answer)
    if len(answer_formulation) > 0:
        answer_formulation = answer_formulation[0].upper() + answer_formulation[1:]
    return question, question_formulation, answer_formulation


def _get_qa_pair_count_people(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    try:
        roles = _get_roles_from_sample(sample, multimodal_data, dataset_name)
    except Exception as e:
        if for_inference:
            roles = []
        else:
            raise e
    people_count = len(roles)
    answer = str(people_count)
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_role_people(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    try:
        roles = _get_roles_from_sample(sample, multimodal_data, dataset_name)
    except Exception as e:
        if for_inference:
            roles = []
        else:
            raise e
    # random order
    roles = list(roles)
    shuffle(roles)
    if len(roles) == 0:
        # rarely allow "no one" as an answer
        if random.random() < 0.1 or for_inference:
            answer = 'no one'
        else:
            return None
    else:
        answer = ', '.join(roles)
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_interaction(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    relationships = sample['relationships']
    if len(relationships) == 0:
        return None
    # pick random relationship to talk about
    sub, obj, rel = random.choice(relationships)
    sub = 'operating_table' if sub == 'ot' else sub
    sub = 'anesthesia_equipment' if sub == 'ae' else sub
    sub = 'anaesthetist' if sub == 'anest' else sub
    obj = 'operating_table' if obj == 'ot' else obj
    obj = 'anesthesia_equipment' if obj == 'ae' else obj
    obj = 'anaesthetist' if obj == 'anest' else obj
    sub = sub.replace('_', ' ')
    obj = obj.replace('_', ' ')
    rel = rel.lower()

    answer = rel
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[sub, obj], answer_format_args=[sub, obj, answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_tool_equipment_attribute(all_synthetic_jsons, question_type, answer_form):
    synthetic_json = random.choice(all_synthetic_jsons)
    with open(synthetic_json, 'r') as f:
        synthetic_data = json.load(f)
        take_name = synthetic_data["take_name"]
        frame_id = synthetic_data["frame_id"]
        sample_id = f'{synthetic_data["take_name"]}_{synthetic_data["frame_id"]}'
        new_image_paths = []
        azure_views_to_use = (2, 1, 3, 5) if '4DOR' in sample_id else (1, 4, 5, 2, 3)
        for view_idx in azure_views_to_use:  # these come from synthetic samples
            img_path = synthetic_json.parent / f'{synthetic_json.stem}_cidx{view_idx}.jpg'
            if img_path.exists():
                new_image_paths.append(str(img_path.absolute()))
        descriptor = synthetic_data['descriptors'][random.choice(list(synthetic_data['descriptors'].keys()))]
        attribute_to_ask = random.choice(['color', 'shape', 'size', 'texture'])
        object_name = descriptor['object_type'].replace('_', ' ')
        answer = descriptor[attribute_to_ask]
        question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[attribute_to_ask, object_name],
                                                                                answer_format_args=[attribute_to_ask, object_name, answer])
        return {'dataset': synthetic_data['dataset_type'], 'take_name': take_name, 'frame_id': frame_id, 'question': question_formulation,
                'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer,
                'new_image_paths': new_image_paths}


def _get_qa_pair_time_until_step(sample, dataset_name, dataset, question_type, answer_form):
    '''
    This requires us to smartly fetch all the scene graphs from the future from this one take, gather all the actions that will happen, pick one as question, determine approximately how long in the future this is etc.
    '''
    # get all the samples belonging to the same take. then sort them cronomologically
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    if dataset_name == 'EgoSurgery':
        fps = 0.5  # will be used to convert timestampish to seconds
        take_future_samples = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] > sample['frame_id']], key=lambda x: x['frame_id'])
        all_future_actions_with_timestamps = {}
        current_timestampish = sample['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
        for s in take_future_samples:
            s_timestampish = s['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
            action = s['phase_label']
            if action not in all_future_actions_with_timestamps:
                all_future_actions_with_timestamps[action] = s_timestampish
        if len(all_future_actions_with_timestamps) == 0:
            return None
        action = random.choice(list(all_future_actions_with_timestamps.keys()))
        action_timestampish = all_future_actions_with_timestamps[action]
        second_difference = int((int(action_timestampish) - int(current_timestampish)) / fps)
    else:
        take_samples = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] > sample['frame_id']], key=lambda x: x['frame_id'])
        all_future_actions_with_timestamps = {}
        for s in take_samples:
            sg = s['relationships']
            actions = _get_current_actions(sg)
            for action in actions:
                if action not in all_future_actions_with_timestamps:
                    all_future_actions_with_timestamps[action] = s['frame_id']
        if len(all_future_actions_with_timestamps) == 0:
            return None

        # select random action
        action = random.choice(list(all_future_actions_with_timestamps.keys()))
        action_frame_id = all_future_actions_with_timestamps[action]
        timestamps = dataset.take_to_timestamps[sample['take_name']]
        current_timestamp = int(timestamps[int(sample['frame_id'])][0])
        future_timestamp = int(timestamps[int(action_frame_id)][0])
        second_difference = future_timestamp - current_timestamp
        # 4D-OR uses nanoseconds, MM-OR uses seconds
        if dataset_name == '4D-OR':
            second_difference = int(second_difference // 1e9)

    answer = f'{second_difference} seconds'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[action])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_status_action(sample, dataset_name, dataset, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    if dataset_name == 'EgoSurgery':
        take_samples_before = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] <= sample['frame_id']], key=lambda x: x['frame_id'])
        take_samples_after = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] > sample['frame_id']], key=lambda x: x['frame_id'])
        current_action = sample['phase_label']
        if current_action is None:
            return None
        # determine progress by finding the start and end. First go back from this index to find the start of the action
        start_timestampish = sample['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
        end_timestampish = sample['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
        current_timestampish = sample['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
        # find the frame where this action started
        for s in take_samples_before[::-1]:
            if s['phase_label'] == current_action:
                start_timestampish = s['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
            else:
                break
        # find the frame where this action ended
        for s in take_samples_after:
            if s['phase_label'] == current_action:
                end_timestampish = s['file_name'].rsplit('_', 1)[-1].replace('.jpg', '')
            else:
                break
        try:
            percantage = int((int(current_timestampish) - int(start_timestampish)) / (int(end_timestampish) - int(start_timestampish)) * 100)
        except ZeroDivisionError:
            return None

    else:
        take_samples_before = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] < sample['frame_id']], key=lambda x: x['frame_id'])
        take_samples_after = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] > sample['frame_id']], key=lambda x: x['frame_id'])
        current_actions = _get_current_actions(sample['relationships'])
        if len(current_actions) == 0:
            return None
        current_action = random.choice(list(current_actions))
        # determine progress by finding the start and end. First go back from this index to find the start of the action
        start_frame_id = sample['frame_id']
        end_frame_id = sample['frame_id']
        # find the frame where this action started
        for s in take_samples_before[::-1]:
            actions = _get_current_actions(s['relationships'])
            if current_action in actions:
                start_frame_id = s['frame_id']
            else:
                break

        # find the frame where this action ended
        for s in take_samples_after:
            actions = _get_current_actions(s['relationships'])
            if current_action in actions:
                end_frame_id = s['frame_id']
            else:
                break

        timestamps = dataset.take_to_timestamps[sample['take_name']]
        start_timestamp = int(timestamps[int(start_frame_id)][0])
        current_timestamp = int(timestamps[int(sample['frame_id'])][0])
        end_timestamp = int(timestamps[int(end_frame_id)][0])

        # calculate the percentage of the action that has been completed
        try:
            percantage = int((current_timestamp - start_timestamp) / (end_timestamp - start_timestamp) * 100)
        except ZeroDivisionError:
            return None

    answer = f'{percantage}%'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[current_action], answer_format_args=[current_action, answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_current_action(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    if dataset_name == 'MVOR':
        current_action = sample['action_label']
    elif dataset_name == 'EgoSurgery':
        current_action = sample['phase_label']
        if current_action is None:
            # only rarely allow this
            if random.random() < 0.1 or for_inference:
                current_action = 'nothing'
            else:
                return None
    else:
        current_actions = _get_current_actions(sample['relationships'])
        if len(current_actions) == 0:
            # only rarely allow this
            if random.random() < 0.1 or for_inference:
                current_action = 'nothing'
            else:
                return None
        else:
            current_action = random.choice(list(current_actions))
    answer = current_action
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_did_happen(sample, dataset_name, dataset, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    past_action = set()
    future_actions = set()
    take_samples_before = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] <= sample['frame_id']], key=lambda x: x['frame_id'])
    take_samples_after = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] > sample['frame_id']], key=lambda x: x['frame_id'])
    if dataset_name == 'EgoSurgery':
        for s in take_samples_before:
            past_action.add(s['phase_label'])
        for s in take_samples_after:
            future_actions.add(s['phase_label'])
    else:
        for s in take_samples_before:
            past_action.update(_get_current_actions(s['relationships']))
        for s in take_samples_after:
            future_actions.update(_get_current_actions(s['relationships']))
    all_actions = past_action.union(future_actions)
    current_action = random.choice(list(all_actions))
    answer = 'yes' if current_action in past_action else 'no'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[current_action], answer_format_args=[current_action],
                                                                            answer_call_args=[answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_is_base_array_visible(sample, dataset_name, dataset, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    take = sample['take_name'].replace('_MMOR', '')
    tracker_tracks_path = MMOR_DATA_ROOT_PATH / 'take_tracks' / f'{take}.json'
    if tracker_tracks_path.exists():
        with tracker_tracks_path.open() as f:
            tracker_track = json.load(f)
    else:
        return None

    timestamps = dataset.take_to_timestamps[sample['take_name']]
    timestamp = int(timestamps[int(sample['frame_id'])][0])
    tracker_state = tracker_track[timestamp]['unique_id_dicts']
    tracker_visible_tools = {TRACKER_OBJECT_MAP[unique_id_dict['unique_id']] for unique_id_dict in tracker_state}
    answer = 'yes' if 'base_array' in tracker_visible_tools else 'no'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, answer_call_args=[answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_is_robot_calibrated(sample, dataset_name, dataset, question_type, answer_form):
    '''
    If relationship robot calibrating was seen already once and will not be seen at all from now on we assume it is calibrated.
    '''
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    take_samples_before = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] <= sample['frame_id']], key=lambda x: x['frame_id'])
    take_samples_after = sorted([s for s in dataset.samples if s['take_name'] == sample['take_name'] and s['frame_id'] > sample['frame_id']], key=lambda x: x['frame_id'])
    robot_started_calibration = False
    robot_finished_calibration = True
    for s in take_samples_before:
        robot_calibration_found = False
        for sub, obj, rel in s['relationships']:
            if rel == 'calibrating' and obj == 'mako_robot':
                robot_calibration_found = True
                break
        if robot_calibration_found:
            robot_started_calibration = True
            break
    for s in take_samples_after:
        if not robot_finished_calibration:
            break
        for sub, obj, rel in s['relationships']:
            if rel == 'calibrating' and obj == 'mako_robot':
                robot_finished_calibration = False
                break
    answer = 'yes' if robot_started_calibration and robot_finished_calibration else 'no'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, answer_call_args=[answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_is_sterility_breach(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    take = sample['take_name'].replace('_MMOR', '')
    try:
        with open(MMOR_DATA_ROOT_PATH / 'take_timestamp_to_sterility_breach' / f'{take}.json', 'r') as f:
            timestamp_to_sterility_breach = json.load(f)
        sterility_breach = timestamp_to_sterility_breach[sample['frame_id']]
    except FileNotFoundError as e:
        if for_inference:  # also a placeholder answer is okey
            sterility_breach = []
        else:
            raise e
    answer = 'yes' if len(sterility_breach) > 0 else 'no'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, answer_call_args=[answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_next_robot_step(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    take = sample['take_name'].replace('_MMOR', '')
    with open(MMOR_DATA_ROOT_PATH / 'take_timestamp_to_robot_phase' / f'{take}.json', 'r') as f:
        timestamp_to_robot_phase = json.load(f)

    current_robot_phase = timestamp_to_robot_phase[sample['frame_id']]
    # get the first robot_phase that is not this phase
    next_timestamps = sorted([key for key in timestamp_to_robot_phase.keys() if key > sample['frame_id']])
    next_robot_phase = None
    for timestamp in next_timestamps:
        if timestamp_to_robot_phase[timestamp] != current_robot_phase:
            next_robot_phase = timestamp_to_robot_phase[timestamp]
            break

    if next_robot_phase is None or len(next_robot_phase) == 0:
        return None
    answer = next_robot_phase
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_current_robot_step(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    take = sample['take_name'].replace('_MMOR', '')
    with open(MMOR_DATA_ROOT_PATH / 'take_timestamp_to_robot_phase' / f'{take}.json', 'r') as f:
        timestamp_to_robot_phase = json.load(f)

    robot_phase = timestamp_to_robot_phase[sample['frame_id']]
    if robot_phase is None or len(robot_phase) == 0:
        return None
    answer = robot_phase
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_random_segmask_with_path(sample, dataset_name):
    take_name = sample['take_name'].replace('_MMOR', '')
    if dataset_name == '4D-OR':
        take_folder = OR4D_TAKE_NAME_TO_FOLDER.get(take_name, take_name)
        take_path = OR_4D_DATA_ROOT_PATH / take_folder
    else:
        take_folder = MMOR_TAKE_NAME_TO_FOLDER.get(take_name, take_name)
        take_path = MMOR_DATA_ROOT_PATH / take_folder
    if dataset_name == "MM-OR":
        json_path = Path('../MM-OR') / 'take_jsons' / f'{take_name}.json'
        # Read MMOR/Simstation JSON file for timestamps and image paths
        with json_path.open() as f:
            take_json = json.load(f)
            timestamps = take_json['timestamps']
            timestamps = {int(k): v for k, v in timestamps.items()}
            timestamps = sorted(timestamps.items())
    elif dataset_name == "4D-OR":
        internal_take_name = f'export_holistic_take{int(take_name.replace("_4DOR", ""))}_processed'
        json_path = Path('../4D-OR') / internal_take_name / 'timestamp_to_pcd_and_frames_list.json'
        # Read 4D-OR JSON file for timestamps and image paths
        with json_path.open() as f:
            take_json = json.load(f)
            timestamps = sorted(take_json)

    # Iterate through timestamps and generate multiview frames
    timestamp, image_files = timestamps[int(sample['frame_id'])]
    segmask = None
    image_path = None
    if dataset_name == "MM-OR":
        c_idx = random.choice([1, 4, 5])
        rgb_path = take_path / 'colorimage' / f'camera0{c_idx}_colorimage-{image_files["azure"]}.jpg'
        mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
        if mask_path.exists():
            segmask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            image_path = rgb_path
        else:
            c_idx = random.choice([0, 2, 3])
            simstation_rgb_path = take_path / 'simstation' / f'camera0{c_idx}_{image_files["simstation"]}.jpg'
            simstation_mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{simstation_rgb_path.stem}.png'
            if simstation_mask_path.exists():
                segmask = cv2.imread(str(simstation_mask_path), cv2.IMREAD_GRAYSCALE)
                image_path = simstation_rgb_path
    elif dataset_name == "4D-OR":
        # 4D-OR uses different camera indices (1, 2, 5)
        c_idx = random.choice([1, 2, 5])
        color_idx_str = image_files[f'color_{c_idx}']
        rgb_path = take_path / f'colorimage/camera0{c_idx}_colorimage-{color_idx_str}.jpg'
        mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
        if mask_path.exists():
            segmask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            image_path = rgb_path

    if segmask is None:
        return None

    return segmask, image_path


def _get_qa_pair_where_2d(sample, dataset_name, question_type, answer_form, num_bins=1000):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    scene_entities_to_2D_positions = {}
    if dataset_name == 'MVOR':  # different but easier. Do not forget the binning part though.
        v_idx = random.choice([0, 1, 2])
        W = 640
        H = 480
        image_path = multimodal_data['azure'][v_idx]
        human_poses_2d = sample['human_poses_2d_per_camera'][v_idx]
        patient_pose_2d = sample['patient_pose_2d_per_camera'][v_idx]
        object_poses_2d = sample['object_poses_2d_per_camera'][v_idx]
        for human_idx, human_pose_2d in enumerate(human_poses_2d):
            scene_entities_to_2D_positions[f'human {human_idx + 1}'] = human_pose_2d
        if patient_pose_2d is not None:
            scene_entities_to_2D_positions['patient'] = patient_pose_2d
        for object_name, object_pose_2d in object_poses_2d.items():
            object_name = object_name.replace('_', ' ').lower()
            scene_entities_to_2D_positions[object_name] = object_pose_2d
        # post processing and binning
        for entity, pose in scene_entities_to_2D_positions.items():
            x_min, y_min = pose.min(axis=0)
            x_max, y_max = pose.max(axis=0)
            bin_x_min = int((x_min / W) * (num_bins - 1))
            bin_x_max = int((x_max / W) * (num_bins - 1))
            bin_y_min = int((y_min / H) * (num_bins - 1))
            bin_y_max = int((y_max / H) * (num_bins - 1))
            scene_entities_to_2D_positions[entity] = (bin_x_min, bin_x_max, bin_y_min, bin_y_max)
    elif dataset_name == 'EgoSurgery':
        W = 1920
        H = 1080
        image_path = multimodal_data['azure'][0]
        for hand_bbox in sample['hands_bboxes']:
            category_name = hand_bbox['category'].replace('_', ' ').lower()
            scene_entities_to_2D_positions[category_name] = hand_bbox['bbox']
        for tool_bbox in sample['tools_bboxes']:
            category_name = tool_bbox['category'].replace('_', ' ').lower()
            scene_entities_to_2D_positions[category_name] = tool_bbox['bbox']
        # post processing and binning
        for entity, bbox in scene_entities_to_2D_positions.items():
            x_min, y_min, x_max, y_max = bbox
            bin_x_min = int((x_min / W) * (num_bins - 1))
            bin_x_max = int((x_max / W) * (num_bins - 1))
            bin_y_min = int((y_min / H) * (num_bins - 1))
            bin_y_max = int((y_max / H) * (num_bins - 1))
            scene_entities_to_2D_positions[entity] = (bin_x_min, bin_x_max, bin_y_min, bin_y_max)
    else:
        ret = _get_random_segmask_with_path(sample, dataset_name)
        if ret is None:
            return None
        segmask, image_path = ret
        unique_labels, count = np.unique(np.asarray(segmask), return_counts=True)
        W, H = segmask.shape[1], segmask.shape[0]
        count = count.tolist()
        for label, c in zip(unique_labels, count):
            if label == 0:
                continue
            if c < 10:
                continue
            try:
                category_id = label_to_category_id[label]
            except KeyError:
                continue
            category_name = sorted_classes[category_id]
            category_name = 'anaesthetist' if category_name == 'anest' else category_name
            category_name = 'operating table' if category_name == 'ot' else category_name
            category_name = 'anesthesia equipment' if category_name == 'ae' else category_name
            category_name = category_name.replace('_', ' ')
            # find the bounding box of this entity
            mask = (segmask == label).astype(np.uint8)
            coords = np.argwhere(mask == 1)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
            else:
                # No object pixels found
                return None
            bin_x_min = int((x_min / W) * (num_bins - 1))
            bin_x_max = int((x_max / W) * (num_bins - 1))
            bin_y_min = int((y_min / H) * (num_bins - 1))
            bin_y_max = int((y_max / H) * (num_bins - 1))
            scene_entities_to_2D_positions[category_name] = (bin_x_min, bin_x_max, bin_y_min, bin_y_max)

    if len(scene_entities_to_2D_positions) == 0:
        return None
    # randomly choose an entity to ask about
    entity = random.choice(list(scene_entities_to_2D_positions.keys()))
    x_min, x_max, y_min, y_max = scene_entities_to_2D_positions[entity]
    answer = f'{x_min}, {x_max}, {y_min}, {y_max}'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[entity], answer_format_args=[entity, answer])
    # # plot the bounding box on top of the image, plot the name as well, save as matplotlib
    # from matplotlib import pyplot as plt
    # import matplotlib.patches as patches
    # fig, ax = plt.subplots()
    # # reaad image_path
    # img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    # ax.imshow(img)
    # # reinterpret, xmin, etc as pixel values
    # original_xmin = int(x_min / num_bins * W)
    # original_xmax = int(x_max / num_bins * W)
    # original_ymin = int(y_min / num_bins * H)
    # original_ymax = int(y_max / num_bins * H)
    # rect = patches.Rectangle((original_xmin, original_ymin), original_xmax - original_xmin, original_ymax - original_ymin, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    # ax.text(original_xmin, original_ymin, entity, color='red')
    # plt.savefig('debug.jpg')

    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer, 'image_path': str(image_path)}


def _get_scene_entities_to_3D_positions(sample, dataset_name, multimodal_data, normalize=True):
    scene_entities_to_3D_positions = {}
    if dataset_name == 'MVOR':  # it is different for this dataset, but actually even simpler because we precompute most stuff
        for human_idx, human_pose_3d in enumerate(sample['human_poses_3d']):
            scene_entities_to_3D_positions[f'human {human_idx + 1}'] = human_pose_3d
        patient_pose_3d = sample['patient_pose_3d']
        if patient_pose_3d is not None:
            scene_entities_to_3D_positions['patient'] = patient_pose_3d
        for object_name, object_pose_3d in sample['object_poses_3d'].items():
            object_name = object_name.replace('_', ' ')
            scene_entities_to_3D_positions[object_name] = object_pose_3d

        # post processing and optional normalization
        for entity, pose in scene_entities_to_3D_positions.items():
            x_min, y_min, z_min = pose.min(axis=0)
            x_max, y_max, z_max = pose.max(axis=0)
            if normalize:
                original_range = (-3500, 3500)
                x_min = int((x_min - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                x_max = int((x_max - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                y_min = int((y_min - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                y_max = int((y_max - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                z_min = int((z_min - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                z_max = int((z_max - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            z_center = (z_min + z_max) // 2
            scene_entities_to_3D_positions[entity] = (x_center, y_center, z_center)
        return scene_entities_to_3D_positions
    if 'pc' not in multimodal_data:
        return None
    take = sample['take_name'].replace('_MMOR', '')
    seg_pc_path = MMOR_DATA_ROOT_PATH / 'take_point_clouds_seg_lowres' / take / f'{multimodal_data["pc"][0].stem}_seg.pcd'
    seg_pcd = o3d.io.read_point_cloud(str(seg_pc_path))
    seg_pcd_points = np.asarray(seg_pcd.points)
    seg_pcd_colors = np.asarray(seg_pcd.colors)
    if seg_pcd_points.size == 0:
        return None
    unique_colors = np.unique(seg_pcd_colors, axis=0)
    # now this is not mapped to labels but instead to colors. But essentially we need to find the 3D center.
    for color in unique_colors:
        color_new = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        if color_new == (0, 0, 0):
            continue
        category_id = color_to_category_id[color_new]
        category_name = sorted_classes[category_id]
        category_name = 'anaesthetist' if category_name == 'anest' else category_name
        category_name = 'operating table' if category_name == 'ot' else category_name
        category_name = 'anesthesia equipment' if category_name == 'ae' else category_name
        category_name = category_name.replace('_', ' ')
        mask = (seg_pcd_colors == color).all(axis=1)
        coords = seg_pcd_points[mask]
        if coords.size > 0:
            x_min, y_min, z_min = coords.min(axis=0)
            x_max, y_max, z_max = coords.max(axis=0)
            if normalize:
                # normalize between -1000 and 1000. Originally it seems to be between -1500 and 1500. Assume it can be maximum 1500 in any direction. and start by dividing by this
                if dataset_name == 'MM-OR':
                    original_range = (-1500, 1500)
                elif dataset_name == '4D-OR':
                    raise NotImplementedError
                    original_range = (-1000, 1000)
                x_min = int((x_min - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                x_max = int((x_max - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                y_min = int((y_min - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                y_max = int((y_max - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                z_min = int((z_min - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
                z_max = int((z_max - original_range[0]) / (original_range[1] - original_range[0]) * 2000 - 1000)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            z_center = (z_min + z_max) // 2
        else:
            return None
        scene_entities_to_3D_positions[category_name] = (x_center, y_center, z_center)
    return scene_entities_to_3D_positions


def _get_qa_pair_where_3d(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    scene_entities_to_3D_positions = _get_scene_entities_to_3D_positions(sample, dataset_name, multimodal_data)
    if scene_entities_to_3D_positions is None:
        return None

    # randomly choose an entity to ask about
    entity = random.choice(list(scene_entities_to_3D_positions.keys()))
    x_center, y_center, z_center = scene_entities_to_3D_positions[entity]
    answer = f'{x_center}, {y_center}, {z_center}'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[entity], answer_format_args=[entity, answer])

    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_distance_3d(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    scene_entities_to_3D_positions = _get_scene_entities_to_3D_positions(sample, dataset_name, multimodal_data, normalize=False)
    if scene_entities_to_3D_positions is None:
        return None
    # randomly choose two entities to ask about
    entity1, entity2 = random.sample(list(scene_entities_to_3D_positions.keys()), 2)
    x1, y1, z1 = scene_entities_to_3D_positions[entity1]
    x2, y2, z2 = scene_entities_to_3D_positions[entity2]
    # we would like to have it in accurate mm, which should be possible, even tough this valued was manipulated a lot.
    # the original value was in meters, then it got divided by 1000, and multiplied by 500. Essentially, this means it got divided by 2. We can multiply by 2 to get the original value.
    x1, y1, z1 = x1 * 2, y1 * 2, z1 * 2
    x2, y2, z2 = x2 * 2, y2 * 2, z2 * 2
    distance = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))
    answer = f'{distance} mm'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, question_format_args=[entity1, entity2])
    # # plot one image from samples together with the entity names and mm distance to verify
    # from matplotlib import pyplot as plt
    # import matplotlib.patches as patches
    # fig, ax = plt.subplots()
    # # reaad image_path
    # img = cv2.cvtColor(cv2.imread(str(multimodal_data['azure'][0])), cv2.COLOR_BGR2RGB)
    # ax.imshow(img)
    # # reinterpret, xmin, etc as pixel values
    # ax.text(0, 0, entity1, color='red', fontsize=10)
    # ax.text(750, 0, entity2, color='red', fontsize=10)
    # ax.text(1500, 0, answer, color='red', fontsize=10)
    # plt.savefig('debug.jpg')
    # plt.close()
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_tools_used(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    tools_used = set()
    if dataset_name == 'EgoSurgery':
        for tool_bbox in sample['tools_bboxes']:
            category_name = tool_bbox['category'].replace('_', ' ').lower()
            tools_used.add(category_name)
    else:
        # Determine tool used from scene graph instead
        try:
            sg = sample['relationships']
            for sub, obj, rel in sg:
                if rel == 'holding':
                    if obj in ['drill', 'saw', 'hammer']:
                        tools_used.add(obj)
                elif rel == 'cementing':
                    tools_used.add('cementer')
                elif rel == 'cutting':
                    tools_used.add('scalpel')
                elif rel == 'drilling':
                    tools_used.add('drill')
                elif rel == 'hammering':
                    tools_used.add('hammer')
                elif rel == 'sawing':
                    tools_used.add('saw')
                elif rel == 'suturing':
                    tools_used.add('forceps')
        except KeyError:
            if for_inference:
                tools_used = set()
            else:
                return None

    if len(tools_used) == 0:
        # sometimes allow it as "no" but usually return None
        if random.random() < 0.1 or for_inference:
            answer = 'nothing'
        else:
            return None
    else:
        # random order
        tools_used = list(tools_used)
        shuffle(tools_used)
        answer = ', '.join(tools_used)
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form, answer_format_args=[answer])
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_current_scene_graph(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    try:
        sg = sample['relationships']
        shuffle(sg)
        answer = scene_graph_to_string(sg)
    except KeyError as e:
        if for_inference:
            answer = ''
        else:
            raise e
    if len(answer) == 0:
        if for_inference:
            answer = ''
        else:
            return None
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_list_all_entities(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    if dataset_name == 'EgoSurgery':
        scene_entities = set()
        for hand_bbox in sample['hands_bboxes']:
            category_name = hand_bbox['category'].replace('_', ' ').lower()
            scene_entities.add(category_name)
        for tool_bbox in sample['tools_bboxes']:
            category_name = tool_bbox['category'].replace('_', ' ').lower()
            scene_entities.add(category_name)
        if len(scene_entities) == 0 and not for_inference:
            return None
    else:
        scene_entities = _get_entities_from_samples(dataset_name, sample, multimodal_data, entities_of_interest=None)
    if scene_entities is None:
        return None
    scene_entities = list(scene_entities)
    shuffle(scene_entities)
    if len(scene_entities) == 0:
        # rarely allow it as "nothing" but usually return None
        if random.random() < 0.1 or for_inference:
            answer = 'nothing'
        else:
            return None
    else:
        answer = ', '.join(scene_entities)
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_list_all_entities_ordered_2D(sample, dataset_name, question_type, answer_form, num_bins=1000):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    scene_entities_to_2D_positions = {}
    if dataset_name == 'MVOR':  # different but easier. Do not forget the binning part though.
        v_idx = random.choice([0, 1, 2])
        W = 640
        image_path = multimodal_data['azure'][v_idx]
        human_poses_2d = sample['human_poses_2d_per_camera'][v_idx]
        patient_pose_2d = sample['patient_pose_2d_per_camera'][v_idx]
        object_poses_2d = sample['object_poses_2d_per_camera'][v_idx]
        for human_idx, human_pose_2d in enumerate(human_poses_2d):
            scene_entities_to_2D_positions[f'human {human_idx + 1}'] = human_pose_2d
        if patient_pose_2d is not None:
            scene_entities_to_2D_positions['patient'] = patient_pose_2d
        for object_name, object_pose_2d in object_poses_2d.items():
            object_name = object_name.replace('_', ' ')
            scene_entities_to_2D_positions[object_name] = object_pose_2d
        # post processing and binning
        for entity, pose in scene_entities_to_2D_positions.items():
            x_min, y_min = pose.min(axis=0)
            x_max, y_max = pose.max(axis=0)
            bin_x_min = int((x_min / W) * (num_bins - 1))
            bin_x_max = int((x_max / W) * (num_bins - 1))
            x_center = (bin_x_min + bin_x_max) // 2
            scene_entities_to_2D_positions[entity] = x_center
    elif dataset_name == 'EgoSurgery':
        image_path = multimodal_data['azure'][0]
        for hand_bbox in sample['hands_bboxes']:
            category_name = hand_bbox['category'].replace('_', ' ').lower()
            scene_entities_to_2D_positions[category_name] = hand_bbox['bbox']
        for tool_bbox in sample['tools_bboxes']:
            category_name = tool_bbox['category'].replace('_', ' ').lower()
            scene_entities_to_2D_positions[category_name] = tool_bbox['bbox']
        # post processing and binning
        for entity, bbox in scene_entities_to_2D_positions.items():
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) // 2
            scene_entities_to_2D_positions[entity] = x_center

        if len(scene_entities_to_2D_positions) == 0:
            return None
    else:
        ret = _get_random_segmask_with_path(sample, dataset_name)
        if ret is None:
            return None
        segmask, image_path = ret
        unique_labels, count = np.unique(np.asarray(segmask), return_counts=True)
        W, H = segmask.shape[1], segmask.shape[0]
        count = count.tolist()
        for label, c in zip(unique_labels, count):
            if label == 0:
                continue
            if c < 10:
                continue
            try:
                category_id = label_to_category_id[label]
            except KeyError:
                continue
            category_name = sorted_classes[category_id]
            category_name = 'anaesthetist' if category_name == 'anest' else category_name
            category_name = 'operating table' if category_name == 'ot' else category_name
            category_name = 'anesthesia equipment' if category_name == 'ae' else category_name
            category_name = category_name.replace('_', ' ')
            # find the bounding box of this entity
            mask = (segmask == label).astype(np.uint8)
            coords = np.argwhere(mask == 1)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
            else:
                # No object pixels found
                return None
            bin_x_min = int((x_min / W) * (num_bins - 1))
            bin_x_max = int((x_max / W) * (num_bins - 1))
            x_center = (bin_x_min + bin_x_max) // 2
            scene_entities_to_2D_positions[category_name] = x_center

    # now sort the entities based on their x_center
    scene_entities_to_2D_positions = {k: v for k, v in sorted(scene_entities_to_2D_positions.items(), key=lambda item: item[1])}
    if len(scene_entities_to_2D_positions) == 0:
        # rarely allow it as "nothing" but usually return None
        if random.random() < 0.1:
            answer = 'nothing'
        else:
            return None
    else:
        answer = ', '.join(scene_entities_to_2D_positions.keys())
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer, 'image_path': str(image_path)}


def _get_qa_pair_monitor_reading(sample, dataset_name, question_type, answer_form):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    take = sample['take_name'].replace('_MMOR', '')
    TAKE_SIMSTATION_PATH = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take, take) / 'simstation'
    EXPORT_PATH = MMOR_DATA_ROOT_PATH / 'screen_summaries_text' / take
    if 'simstation' not in multimodal_data or len(multimodal_data['simstation']) == 0:
        return None
    simstation_idx = multimodal_data["simstation"][1].stem.split('_')[1]
    screen_path = TAKE_SIMSTATION_PATH / f'camera01_{simstation_idx}.jpg'
    save_path = EXPORT_PATH / f'{simstation_idx}.txt'
    if not save_path.exists():
        return None
    with save_path.open() as f:
        text_summary = f.read().strip().replace('"', '').replace("'", '')
    answer = text_summary
    if len(answer) == 0:
        return None
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer, 'image_path': str(screen_path)}


def _get_qa_pair_gaze_location(sample, dataset_name, question_type, answer_form, num_bins=1000):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    gaze = sample['gaze']
    # here we just return it, however binning is meaninful
    W = 1920
    H = 1080
    x, y = gaze
    bin_x = int((x / W) * (num_bins - 1))
    bin_y = int((y / H) * (num_bins - 1))
    answer = f'{bin_x}, {bin_y}'
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    # # plot gaze
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # # reaad image_path
    # img = cv2.cvtColor(cv2.imread(str(multimodal_data['azure'][0])), cv2.COLOR_BGR2RGB)
    # ax.imshow(img)
    # ax.scatter(x, y, color='green')
    # plt.savefig('debug.jpg')
    # plt.close()
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _get_qa_pair_gaze_object(sample, dataset_name, question_type, answer_form, for_inference=False):
    sample, multimodal_data = sample['sample'], sample['multimodal_data']  # might need adjustments for other datasets
    # here we will see if it intersects with any of the tools or hands. Prefer tool
    try:
        gaze = sample['gaze']
        gaze_x, gaze_y = gaze
        answer = None
        for tool_bbox in sample['tools_bboxes']:
            x_min, y_min, x_max, y_max = tool_bbox['bbox']
            if gaze_x >= x_min and gaze_x <= x_max and gaze_y >= y_min and gaze_y <= y_max:
                answer = tool_bbox['category'].replace('_', ' ').lower()
                break
        if answer is None:
            for hand_bbox in sample['hands_bboxes']:
                x_min, y_min, x_max, y_max = hand_bbox['bbox']
                if gaze_x >= x_min and gaze_x <= x_max and gaze_y >= y_min and gaze_y <= y_max:
                    answer = hand_bbox['category'].replace('_', ' ').lower()
                    break
        if answer is None:
            if for_inference:
                answer = 'nothing'
            else:
                return None
    except KeyError:
        if for_inference:
            answer = 'nothing'
        else:
            return None
    question, question_formulation, answer_formulation = _get_qa_formatting(question_type, answer, answer_form)
    # # plot gaze
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # # reaad image_path
    # img = cv2.cvtColor(cv2.imread(str(multimodal_data['azure'][0])), cv2.COLOR_BGR2RGB)
    # ax.imshow(img)
    # ax.scatter(gaze_x, gaze_y, color='green')
    # # plot the intersected object
    # ax.text(0, 0, answer, color='blue')
    # plt.savefig('debug.jpg')
    # plt.close()
    return {'dataset': dataset_name, 'take_name': sample['take_name'], 'frame_id': sample['frame_id'], 'question': question_formulation,
            'answer': answer_formulation, '_question_type': question_type, '_question': question, '_answer': answer}


def _process_fn(_):
    # sample dataset based on weight
    dataset_name = np.random.choice(DATASET_NAMES_LIST, p=DATASET_WEIGHTS_LIST)
    dataset = DATASETS[dataset_name]
    answer_form = 'concise' if random.random() < 0.5 else 'verbose'  # maybe less then 50 percent
    question_type = random.choice(ALLOWED_QUESTIONS_PER_DATASET[dataset_name])
    # randomly sample one thing from the dataset
    sample = dataset[random.randint(0, len(dataset) - 1)]
    if question_type == 'count_people':
        qa_pair = _get_qa_pair_count_people(sample, dataset_name, question_type, answer_form)
    elif question_type == 'role_people':
        qa_pair = _get_qa_pair_role_people(sample, dataset_name, question_type, answer_form)
    elif question_type == 'interaction':
        qa_pair = _get_qa_pair_interaction(sample, dataset_name, question_type, answer_form)
    elif question_type == 'tool_equipment_attribute':
        qa_pair = _get_qa_pair_tool_equipment_attribute(ALL_SYNTHETIC_JSONS, question_type, answer_form)
    elif question_type == 'time_until_step':
        qa_pair = _get_qa_pair_time_until_step(sample, dataset_name, dataset, question_type, answer_form)
    elif question_type == 'status_action':
        qa_pair = _get_qa_pair_status_action(sample, dataset_name, dataset, question_type, answer_form)
    elif question_type == 'current_action':
        qa_pair = _get_qa_pair_current_action(sample, dataset_name, question_type, answer_form)
    elif question_type == 'did_happen':
        qa_pair = _get_qa_pair_did_happen(sample, dataset_name, dataset, question_type, answer_form)
    elif question_type == 'is_base_array_visible':
        qa_pair = _get_qa_pair_is_base_array_visible(sample, dataset_name, dataset, question_type, answer_form)
    elif question_type == 'is_robot_calibrated':
        qa_pair = _get_qa_pair_is_robot_calibrated(sample, dataset_name, dataset, question_type, answer_form)
    elif question_type == 'sterility_breach':
        qa_pair = _get_qa_pair_is_sterility_breach(sample, dataset_name, question_type, answer_form)
    elif question_type == 'next_robot_step':
        qa_pair = _get_qa_pair_next_robot_step(sample, dataset_name, question_type, answer_form)
    elif question_type == 'current_robot_step':
        qa_pair = _get_qa_pair_current_robot_step(sample, dataset_name, question_type, answer_form)
    elif question_type == 'where_2d':
        qa_pair = _get_qa_pair_where_2d(sample, dataset_name, question_type, answer_form)
    elif question_type == 'where_3d':
        qa_pair = _get_qa_pair_where_3d(sample, dataset_name, question_type, answer_form)
    elif question_type == 'distance_3d':
        qa_pair = _get_qa_pair_distance_3d(sample, dataset_name, question_type, answer_form)
    elif question_type == 'tools_used':
        qa_pair = _get_qa_pair_tools_used(sample, dataset_name, question_type, answer_form)
    elif question_type == 'current_scene_graph':
        qa_pair = _get_qa_pair_current_scene_graph(sample, dataset_name, question_type, answer_form)
    elif question_type == 'list_all_entities':
        qa_pair = _get_qa_pair_list_all_entities(sample, dataset_name, question_type, answer_form)
    elif question_type == 'list_all_entities_ordered_2D':
        qa_pair = _get_qa_pair_list_all_entities_ordered_2D(sample, dataset_name, question_type, answer_form)
    elif question_type == 'monitor_reading':
        qa_pair = _get_qa_pair_monitor_reading(sample, dataset_name, question_type, answer_form)
    elif question_type == 'gaze_location':
        qa_pair = _get_qa_pair_gaze_location(sample, dataset_name, question_type, answer_form)
    elif question_type == 'gaze_object':
        qa_pair = _get_qa_pair_gaze_object(sample, dataset_name, question_type, answer_form)
    else:
        raise f'Not implemented question type: {question_type}'

    return qa_pair


def main():
    '''
    Sample representation: Dataset, Take/Timepoint (unique identifier to the existing dataset), Question and Synonmy (original question always not synonmy), Correct Answer, Extra Info: If 2D, specify which image)
    The details regarding that timepoint can be fetched regularly from the dataset, these questions are orthogonal to the dataset.
    '''
    N_QUESTIONS = 100_000_000 if SPLIT == 'train' else 10_000_000  # Number of questions to generate, can be adjusted
    MAX_CHUNK_SIZE = 1_000_000  # Maximum size of each JSON file
    N_WORKERS = 64  # Number of parallel workers
    chunk_index = 0  # To track the chunk file number
    qa_pairs = []  # Accumulator for current chunk
    total_generated_pairs = 0
    save_root = Path(f'data/qa_pairs_{SPLIT}_{N_QUESTIONS}')
    save_root.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(range(N_QUESTIONS), desc='Generating QA pairs')
    while total_generated_pairs < N_QUESTIONS:
        remaining = N_QUESTIONS - len(qa_pairs)
        chunk_remaining = MAX_CHUNK_SIZE - len(qa_pairs)
        work_remaining = min(remaining, chunk_remaining)
        # if work remaining less then N_WORKERS, just do single processing
        if work_remaining < N_WORKERS:
            results = process_map(_process_fn, range(work_remaining), total=work_remaining, max_workers=1, chunksize=1)
        else:
            optimal_chunk_size = max(1, (work_remaining // N_WORKERS // 10))
            results = process_map(_process_fn, range(work_remaining), total=work_remaining, max_workers=N_WORKERS, chunksize=optimal_chunk_size)

        # Filter valid QA pairs
        valid_pairs = [r for r in results if r is not None]
        qa_pairs.extend(valid_pairs)
        progress_bar.update(len(valid_pairs))
        total_generated_pairs += len(valid_pairs)

        # Save to file if the chunk size is reached
        if len(qa_pairs) >= MAX_CHUNK_SIZE:
            save_path = save_root / f'qa_pairs_chunk_{chunk_index}.json'
            with save_path.open('w') as f:
                json.dump(qa_pairs[:MAX_CHUNK_SIZE], f, indent=4)
            chunk_index += 1
            qa_pairs = qa_pairs[MAX_CHUNK_SIZE:]

    # Save any remaining QA pairs in the last chunk
    if len(qa_pairs) > 0:
        save_path = save_root / f'qa_pairs_chunk_{chunk_index}.json'
        with save_path.open('w') as f:
            json.dump(qa_pairs, f, indent=4)

    progress_bar.close()


if __name__ == '__main__':
    main()
