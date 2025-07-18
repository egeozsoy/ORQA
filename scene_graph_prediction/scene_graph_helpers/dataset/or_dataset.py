import json
from pathlib import Path

from torch.utils.data import Dataset

from helpers.configurations import OR4D_TAKE_NAME_TO_FOLDER, OR4D_TAKE_NAMES, MMOR_TAKE_NAMES, OR_4D_DATA_ROOT_PATH, MMOR_DATA_ROOT_PATH, \
    MMOR_TAKE_NAME_TO_FOLDER
from scene_graph_prediction.utils import util


class ORDataset(Dataset):
    def __init__(self,
                 config,
                 split='train',
                 load_4dor=True,
                 load_mmor=True):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.config = config
        self.data_path = Path('data')

        self.take_to_timestamps = {}
        self.take_to_trackertracks = {}
        if load_4dor:
            for take in OR4D_TAKE_NAMES:
                with (OR_4D_DATA_ROOT_PATH / OR4D_TAKE_NAME_TO_FOLDER[take] / f'timestamp_to_pcd_and_frames_list.json').open() as f:
                    self.take_to_timestamps[take] = json.load(f)
        if load_mmor:
            for take in MMOR_TAKE_NAMES:
                timestamp_json_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take, take) / f'timestamp_to_pcd_and_frames_list_{take}.json'  # first try the specific one
                if not timestamp_json_path.exists():
                    # try to find without suffix
                    timestamp_json_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take, take) / 'timestamp_to_pcd_and_frames_list.json'
                    assert timestamp_json_path.exists(), f'Could not find timestamp json for take {take}'
                with timestamp_json_path.open() as f:
                    timestamps = json.load(f)
                    self.take_to_timestamps[f'{take}_MMOR'] = timestamps
                # try to see if a corresponding tracker_tracks exists
                tracker_tracks_path = MMOR_DATA_ROOT_PATH / 'take_tracks' / f'{take}.json'
                if tracker_tracks_path.exists():
                    with tracker_tracks_path.open() as f:
                        self.take_to_trackertracks[take] = json.load(f)

        self.classes = util.read_txt_to_list(self.data_path / 'classes.txt')
        self.relations = util.read_relationships(self.data_path / 'relationships.txt')
        if 'none' not in self.relations:
            self.relations.append('none')
        self.samples_path = self.data_path / f'relationships_validation.json' if split == 'val' else self.data_path / f'relationships_{split}.json'

        with self.samples_path.open() as f:
            self.samples = json.load(f)
            if not load_4dor:
                print('Removing 4DOR takes')
                self.samples = [x for x in self.samples if '4DOR' not in x['take_name']]
            if not load_mmor:
                print('Removing MMOR takes')
                self.samples = [x for x in self.samples if 'MMOR' not in x['take_name']]

        self.key_to_sample_idx = {f'{sample["take_name"]}_{sample["frame_id"]}': idx for idx, sample in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def get_sample_by_key(self, key):
        sample_idx = self.key_to_sample_idx[key]
        # use the get_item function which already takes care of loading multimodal data
        return self[sample_idx]

    def _load_multimodal_data(self, sample, load_azure=False, load_simstation=False, load_trackercam=False, load_pc=False, load_robot_metadata=False, load_tracking=False, load_audio=False,
                              load_speech_transcript=False, load_segmasks=False):
        multimodal_data = {}
        if load_azure:
            image_paths = []
            if '4DOR' in sample['take_name']:
                for c_idx in range(1, 7):
                    color_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][1][f'color_{c_idx}']
                    color_path = OR_4D_DATA_ROOT_PATH / OR4D_TAKE_NAME_TO_FOLDER.get(sample["take_name"], sample["take_name"]) / 'colorimage' / f'camera0{c_idx}_colorimage-{color_idx_str}.jpg'
                    if color_path.exists():
                        image_paths.append(color_path)
                    else:
                        # print(f'{color_path} does not exist')
                        a = 1
            elif 'MMOR' in sample['take_name']:
                for c_idx in range(1, 6):
                    take_name = sample["take_name"].replace('_MMOR', '')
                    color_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][1]['azure']
                    color_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take_name, take_name) / 'colorimage' / f'camera0{c_idx}_colorimage-{color_idx_str}.jpg'
                    if color_path.exists():
                        image_paths.append(color_path)
                    else:
                        # print(f'{color_path} does not exist')
                        a = 1
            else:
                raise ValueError('Unknown take type')
            multimodal_data['azure'] = image_paths
        if load_simstation:
            image_paths = []
            if 'MMOR' in sample['take_name']:  # only MMOR has simstation
                take_name = sample["take_name"].replace('_MMOR', '')
                simstation_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][1]['simstation']
                for i in range(0, 4):
                    img_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take_name, take_name) / 'simstation' / f'camera0{i}_{simstation_idx_str}.jpg'
                    if img_path.exists():
                        image_paths.append(img_path)
                    else:
                        # print(f'{img_path} does not exist')
                        a = 1
                multimodal_data['simstation'] = image_paths
        if load_trackercam:
            # this is very dark, is designed to capture the highlights. However we can try brightening it to see if it works better.
            image_paths = []
            if 'MMOR' in sample['take_name']:  # only MMOR has trackercam
                take_name = sample["take_name"].replace('_MMOR', '')
                trackercam_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][1]['trackercam']
                img_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take_name, take_name) / 'trackercam' / f'{trackercam_idx_str}.jpg'
                if img_path.exists():
                    image_paths.append(img_path)
                else:
                    # print(f'{img_path} does not exist')
                    a = 1
                multimodal_data['trackercam'] = image_paths
        if load_pc:
            if '4DOR' in sample['take_name']:
                pcd_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][1]['pcd']
                pcd_path = OR_4D_DATA_ROOT_PATH / OR4D_TAKE_NAME_TO_FOLDER.get(sample["take_name"], sample["take_name"]) / 'pcds_sparse' / f'{pcd_idx_str}.pcd'  # using sparse for now
                if pcd_path.exists():
                    multimodal_data['pc'] = [pcd_path]
                else:
                    # print(f'{pcd_path} does not exist')
                    a = 1
            elif 'MMOR' in sample['take_name']:
                take_name = sample["take_name"].replace('_MMOR', '')
                timestamp_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][0]  # this is the way to get the timestamp idx, will be needed for audio
                pcd_path = MMOR_DATA_ROOT_PATH / 'take_point_clouds_sparse' / take_name / f'{timestamp_idx_str}.pcd'
                if pcd_path.exists():
                    multimodal_data['pc'] = [pcd_path]
                else:
                    # print(f'{pcd_path} does not exist')
                    a = 1
        if load_robot_metadata:  # this should contain both the robot screen and also the internal logs. If internal logs are not ready we will just go with the screen.
            if 'MMOR' in sample['take_name']:  # robot metadata only exists in MMOR
                take_name = sample["take_name"].replace('_MMOR', '')
                simstation_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][1]['simstation']
                robot_screen_summary_path = MMOR_DATA_ROOT_PATH / 'screen_summaries' / take_name / f'{simstation_idx_str}.json'
                if robot_screen_summary_path.exists():
                    multimodal_data['robot_metadata'] = [robot_screen_summary_path]
        if load_tracking:
            if 'MMOR' in sample['take_name']:
                take_name = sample["take_name"].replace('_MMOR', '')
                timestamp_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][0]
                # get the corresponding track and then the corresponding timepoint
                if take_name in self.take_to_trackertracks:
                    tracker_info = self.take_to_trackertracks[take_name][int(timestamp_idx_str)]
                    multimodal_data['tracker'] = [tracker_info]
                else:
                    # print(f'No tracker track found for take {take_name}')
                    a = 1

        if load_audio:
            if 'MMOR' in sample['take_name']:  # only MMOR has audio
                take_name = sample["take_name"].replace('_MMOR', '')
                timestamp_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][0]  # this is the way to get the timestamp idx, will be needed for audio
                audio_embedding_path = MMOR_DATA_ROOT_PATH / 'take_audio_embeddings_per_timepoint' / take_name / f'{timestamp_idx_str}.pt'
                if audio_embedding_path.exists():
                    multimodal_data['audio'] = [audio_embedding_path]
                raw_audio_path = MMOR_DATA_ROOT_PATH / 'take_audio_per_timepoint' / take_name / f'{timestamp_idx_str}.mp3'  # not given to the model, but can be used for debugging
                if raw_audio_path.exists():
                    multimodal_data['raw_audio'] = [raw_audio_path]

        if load_speech_transcript:
            if 'MMOR' in sample['take_name']:
                take_name = sample["take_name"].replace('_MMOR', '')
                timestamp_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][0]
                speech_transcript_path = MMOR_DATA_ROOT_PATH / 'take_transcripts_per_timepoint' / take_name / f'{timestamp_idx_str}.json'
                if speech_transcript_path.exists():
                    multimodal_data['speech_transcript'] = [speech_transcript_path]

        if load_segmasks:
            # USE_GT = True  # decide if segmasks are from GT or from predictions
            USE_GT = False
            # this is available for both 4DOR and MMOR
            segmasks = []
            if '4DOR' in sample['take_name']:
                take_name = sample["take_name"]
                timestamp_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][0]
                segmask_path = OR_4D_DATA_ROOT_PATH / 'take_segmasks_per_timepoint' / take_name
                for i in range(3):  # there can be up to 3 segmasks
                    segmask_path_i = segmask_path / f'{timestamp_idx_str}_{i}_GT{USE_GT}.png'
                    if segmask_path_i.exists():
                        segmasks.append(segmask_path_i)
            elif 'MMOR' in sample['take_name']:
                take_name = sample["take_name"].replace('_MMOR', '')
                timestamp_idx_str = self.take_to_timestamps[sample['take_name']][int(sample['frame_id'])][0]
                segmask_path = MMOR_DATA_ROOT_PATH / 'take_segmasks_per_timepoint' / take_name
                for i in range(3):
                    segmask_path_i = segmask_path / f'{timestamp_idx_str}_{i}_GT{USE_GT}.png'
                    if segmask_path_i.exists():
                        segmasks.append(segmask_path_i)
            if len(segmasks) > 0:
                multimodal_data['segmasks'] = segmasks

        return multimodal_data

    def __getitem__(self, index):
        # this get function should really only return paths to different multimodal stuff. The actual loading of these will be in different locations.
        sample = self.samples[index]
        sample['sample_id'] = f'{sample["take_name"]}_{sample["frame_id"]}'
        # Azure > Simstation > TrackerCam > PC > Audio > Speech > RobotMeta > TrackingMeta > Segmasks
        multimodal_data = self._load_multimodal_data(sample, load_azure=True, load_simstation=True, load_trackercam=True, load_pc=True, load_audio=True, load_speech_transcript=True,
                                                     load_robot_metadata=True, load_tracking=True, load_segmasks=True)

        if self.config['USE_VIS_DESC']:
            sample['vis_descriptor_embs'] = self.vis_descriptor_embs
        return {'sample': sample, 'multimodal_data': multimodal_data}
