import csv
import json
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Dataset


class EgoSurgeryDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='../EgoSurgery'):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_root = Path(data_root)

        # Load bounding box annotations for this split
        hands_anno_path = self.data_root / 'annotations' / 'bbox' / 'by_split' / 'hands' / f'{split}.json'
        tools_anno_path = self.data_root / 'annotations' / 'bbox' / 'by_split' / 'tools' / f'{split}.json'
        with hands_anno_path.open() as f:
            data = json.load(f)
            self.images = {elem['id']: elem for elem in data['images']}
            hand_annotations = data['annotations']
            hand_categories = {cat['id']: cat['name'] for cat in data['categories']}
            self.hand_annotations = defaultdict(list)
            for annot in hand_annotations:
                image_id = annot['image_id']
                annot_id = annot['id']
                category_name = hand_categories[annot['category_id']]
                x, y, w, h = annot['bbox']
                x_min, x_max = x, x + w
                y_min, y_max = y, y + h
                bbox = (x_min, y_min, x_max, y_max)
                annot = {'id': annot_id, 'category': category_name, 'bbox': bbox}
                assert image_id in self.images
                self.hand_annotations[image_id].append(annot)
        with tools_anno_path.open() as f:
            data = json.load(f)
            new_images = {elem['id']: elem for elem in data['images']}
            self.images.update(new_images)
            tool_annotations = data['annotations']
            tool_categories = {cat['id']: cat['name'] for cat in data['categories']}
            self.tools_annotations = defaultdict(list)
            for annot in tool_annotations:
                image_id = annot['image_id']
                annot_id = annot['id']
                category_name = tool_categories[annot['category_id']]
                x, y, w, h = annot['bbox']
                x_min, x_max = x, x + w
                y_min, y_max = y, y + h
                bbox = (x_min, y_min, x_max, y_max)
                annot = {'id': annot_id, 'category': category_name, 'bbox': bbox}
                assert image_id in self.images
                self.tools_annotations[image_id].append(annot)

        # frame_keys are like "video_id/frame_id". We'll call video_id as take_name for consistency.
        self.take_to_frames = defaultdict(list)
        for image_id, image in self.images.items():
            take_name = image['file_name'].rsplit('_', 1)[0]
            self.take_to_frames[take_name].append({'file_name': image['file_name'], 'image_id': image_id})
        # sort take_to_frames individually
        self.take_to_frames = {take_name: sorted(frames, key=lambda x: x['file_name']) for take_name, frames in self.take_to_frames.items()}

        # Load phase annotations into a dict {(take_name, frame_id): phase_label}
        self.phase_dict = {}
        phase_dir = self.data_root / 'annotations' / 'phase'
        for take_name in self.take_to_frames:
            csv_path = phase_dir / f'{take_name}.csv'
            if csv_path.exists():
                with csv_path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'Frame' in row and 'Phase' in row:
                            self.phase_dict[(take_name, row['Frame'])] = row['Phase']
        # Load gaze annotations into a dict {(take_name, frame_id): (gaze_x, gaze_y)}
        self.gaze_dict = {}
        gaze_dir = self.data_root / 'gaze'
        W = 1920
        H = 1080
        for take_name in self.take_to_frames:
            csv_path = gaze_dir / f'{take_name}.csv'
            if csv_path.exists():
                with csv_path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'Frame' in row and 'norm_pos_x' in row and 'norm_pos_y' in row:
                            # we don't want normalized, we want in the original image dimension
                            gaze_x = int(float(row['norm_pos_x']) * W)
                            gaze_y = int(float(row['norm_pos_y']) * H)
                            self.gaze_dict[(take_name, row['Frame'])] = (gaze_x, gaze_y)

        # Build samples list
        self.samples = []
        for take_name, frame_metadatas in self.take_to_frames.items():
            for frame_metadata in frame_metadatas:
                image_id = frame_metadata['image_id']
                file_name = frame_metadata['file_name']
                img_path = self.data_root / 'images' / take_name.split('_')[0] / f'{file_name}'
                if not img_path.exists():
                    continue
                # Get bounding boxes
                hands_bboxes = self.hand_annotations.get(image_id, [])
                tools_bboxes = self.tools_annotations.get(image_id, [])

                # Phase and gaze
                phase_label = self.phase_dict.get((take_name, file_name.replace('.jpg', '')), None)
                gaze_coords = self.gaze_dict.get((take_name, file_name.replace('.jpg', '')), None)
                sample = {
                    'image_id': image_id,
                    'take_name': take_name,
                    'file_name': file_name,
                    'img_path': img_path,
                    'sample_id': file_name.replace('.jpg', ''),
                    'frame_id': file_name.replace('.jpg', ''),
                    'hands_bboxes': hands_bboxes,
                    'tools_bboxes': tools_bboxes,
                    'phase_label': phase_label,
                    'gaze': gaze_coords

                }
                self.samples.append(sample)

        self.key_to_sample_idx = {f'{sample["take_name"]}_{sample["frame_id"]}': idx for idx, sample in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def get_sample_by_key(self, key):
        sample_idx = self.key_to_sample_idx[key]
        # use the get_item function which already takes care of loading multimodal data
        return self[sample_idx]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load multimodal data paths
        multimodal_data = {'azure': [sample['img_path']]}
        return {'sample': sample, 'multimodal_data': multimodal_data}


if __name__ == '__main__':
    dataset = EgoSurgeryDataset(split='train')
    sample = dataset[0]
