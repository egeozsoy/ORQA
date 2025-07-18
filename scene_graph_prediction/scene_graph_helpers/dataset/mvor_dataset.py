import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from math import cos, sin
from torch.utils.data import Dataset


def get_transform_matrix(rot_deg=0., rot_axis='x'):
    rot_deg = np.deg2rad(rot_deg)
    transform_matrix = np.eye(4, dtype=np.float64)
    if rot_axis == 'x':
        transform_matrix[1:3, 1:3] = np.array([[cos(rot_deg), -sin(rot_deg)],
                                               [sin(rot_deg), cos(rot_deg)]], dtype=np.float64)
    elif rot_axis == 'y':
        transform_matrix = np.array([[cos(rot_deg), 0, sin(rot_deg), 0],
                                     [0, 1, 0, 0],
                                     [-sin(rot_deg), 0, cos(rot_deg), 0],
                                     [0, 0, 0, 1]], dtype=np.float64)
    elif rot_axis == 'z':
        transform_matrix[0:2, 0:2] = np.array([[cos(rot_deg), -sin(rot_deg)],
                                               [sin(rot_deg), cos(rot_deg)]], dtype=np.float64)

    return transform_matrix


def transform_to_axis_aligned_points(points):
    '''
    Transform an (N, 3) numpy array of points to axis-aligned coordinates.
    x is left-right, bigger x is more right
    y is up-down, bigger y is more up
    z is behind-front, bigger z is closer
    '''
    # Ensure points is Nx3
    assert points.ndim == 2 and points.shape[1] == 3, "points should be of shape (N, 3)"
    # Convert to homogeneous coordinates (N,4)
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])  # (N,4)

    # First transform:
    # [[1, 0, 0, 0],
    #  [0,-1, 0, 0],
    #  [0, 0,-1, 0],
    #  [0, 0, 0, 1]]
    mat_1 = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)

    # Apply mat_1
    points_h = (mat_1 @ points_h.T).T  # (N,4)

    # Rotate 90 deg around y
    mat_y = get_transform_matrix(rot_deg=90, rot_axis='y')
    points_h = (mat_y @ points_h.T).T

    # Rotate 35 deg around z
    mat_z = get_transform_matrix(rot_deg=35, rot_axis='z')
    points_h = (mat_z @ points_h.T).T

    # Convert back to (N,3)
    transformed_points = points_h[:, :3]

    # Optional: recentering

    center = np.array([-2050, -2493, -217])
    transformed_points -= center
    return transformed_points


def load_camera_params(camma_json_path):
    with open(camma_json_path, 'r') as f:
        camma_data = json.load(f)
    intrinsics_list = camma_data['cameras_info']['camParams']['intrinsics']
    extrinsics_list = camma_data['cameras_info']['camParams']['extrinsics']
    # Each element in intrinsics_list/extrinsics_list corresponds to a camera (1,2,3)
    camera_params = {}
    for i in range(3):
        camera_params[i + 1] = {
            'intrinsics': intrinsics_list[i],
            'extrinsics': np.array(extrinsics_list[i]).reshape(4, 4)
        }
    return camera_params


def load_image_paths(data_root, day, frame_num):
    image_paths = []
    for cam_id in [1, 2, 3]:
        rgb_path = data_root / f'day{day}' / f'cam{cam_id}' / 'color' / f'{frame_num}.png'
        if rgb_path.exists():
            image_paths.append(rgb_path)
    return image_paths


def load_point_cloud(data_root, camera_params, day, frame_num):
    # rest of the project uses numpy==1.26.4 but here we need numpy==1.23.0.
    export_path = data_root / 'point_clouds' / f'day{day}_frame_{frame_num}.ply'
    if not export_path.exists():
        assert np.__version__ == '1.23.0', 'pip install numpy==1.23.0 for this function to work'
        # Build combined PC from 3 cams
        pcds = []
        for cam_id in [1, 2, 3]:
            camera_intrinsics = camera_params[cam_id]['intrinsics']
            extrinsics_list = camera_params[cam_id]['extrinsics']
            color_img = np.asarray(Image.open(data_root / f'day{day}' / f'cam{cam_id}' / 'color' / f'{frame_num}.png'))
            depth_img = np.asarray(Image.open(data_root / f'day{day}' / f'cam{cam_id}' / 'depth' / f'{frame_num}.png'))
            color_img = o3d.geometry.Image(np.ascontiguousarray(np.array(color_img)[:, :, 0:3]))  # Ignore alpha channel
            depth_img = o3d.geometry.Image(np.ascontiguousarray(np.array(depth_img, dtype=np.float32) / 10000))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity=False)
            intrinsics = o3d.camera.PinholeCameraIntrinsic(camera_intrinsics['imagesize'][0],
                                                           camera_intrinsics['imagesize'][1],
                                                           camera_intrinsics['focallength'][0],
                                                           camera_intrinsics['focallength'][1],
                                                           camera_intrinsics['principalpoint'][0],
                                                           camera_intrinsics['principalpoint'][1])
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 10000000)
            extrinsics = np.array(extrinsics_list).reshape(4, 4)
            pcd.transform(extrinsics)
            pcds.append(pcd)
        merged_pcd = pcds[0] + pcds[1] + pcds[2]
        points = np.asarray(merged_pcd.points)
        points = transform_to_axis_aligned_points(points)
        merged_pcd.points = o3d.utility.Vector3dVector(points)
        # downsample
        choices = np.random.choice(len(merged_pcd.points), 5_000, replace=len(merged_pcd.points) < 5_000)
        merged_pcd.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd.points)[choices])
        merged_pcd.colors = o3d.utility.Vector3dVector(np.asarray(merged_pcd.colors)[choices])
        o3d.io.write_point_cloud(str(export_path), merged_pcd)
    # return o3d.io.read_point_cloud(str(export_path))
    # just return the path
    return export_path


def load_3d_human_poses(data_root, day, frame_num):
    pose_3d_file = data_root / 'human_poses_3D' / f'pred_day_{day}_frame_{frame_num}.npy'
    if pose_3d_file.exists():
        data = np.load(str(pose_3d_file))
        return data
    else:
        return []


def load_3d_patient_pose(dummy_patient_pose, patient_centers, day, frame_num):
    key = f'day{day}_{frame_num}'
    if key in patient_centers:
        patient_center = patient_centers[key]
    else:
        return None
    current_center = (dummy_patient_pose[4] + dummy_patient_pose[5]) / 2
    pose = dummy_patient_pose.copy()
    pose = pose + (patient_center - current_center)
    return pose


def project_3d_to_2d(points3D, cam_intrinsics, cam_extrinsics):
    if points3D.shape[0] == 0:
        return np.array([])  # Return empty if no points exist

    # Step 1: Transform points using the inverse of extrinsics
    ones = np.ones((points3D.shape[0], 1))  # Add homogeneous coordinate
    points_homogeneous = np.hstack((points3D, ones))  # Nx4 matrix
    transformed_points = (np.linalg.inv(cam_extrinsics) @ points_homogeneous.T).T  # Transform points

    # Step 2: Extract X, Y, Z from transformed points
    X, Y, Z = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]

    # Avoid division by zero for Z
    Z[Z == 0.0] = 1.0

    # Step 3: Apply the 3D-to-2D projection formula
    fx, fy = cam_intrinsics['focallength']
    cx, cy = cam_intrinsics['principalpoint']

    x_values = ((X / Z) * fx) + cx
    y_values = ((Y / Z) * fy) + cy

    # Combine into Nx2 array
    points2D = np.stack([x_values, y_values], axis=1)

    return points2D


def load_3d_object_poses(data_root, day, frame_num, DEPTH_SCALING=4000):
    # Loads object 3D poses from separate npz
    # Format: np.load -> dict {object_name: np.array( ... 3D info ... ), ...}
    object_3d_file = data_root / 'object_poses_3D' / f'day_{day}_frame_{frame_num}.npz'
    object_names_to_points = {}
    if object_3d_file.exists():
        objects = np.load(str(object_3d_file), allow_pickle=True)['arr_0'].item()
        for object_scan_path, transformation in objects.items():
            transformation[:3, 3] = transformation[:3, 3] * DEPTH_SCALING
            object_name = object_scan_path.split("/")[1]
            object_scan = o3d.io.read_point_cloud(str(data_root / object_scan_path))
            object_scan.transform(transformation)
            # downsample num points to 1000, randomly.
            object_points = np.asarray(object_scan.points).astype(np.float32)
            num_points = object_points.shape[0]
            if num_points > 1000:
                indices = np.random.choice(num_points, size=1000, replace=False)
                object_points = object_points[indices]
            object_names_to_points[object_name] = object_points
    return object_names_to_points


def calculate_2D_bounding_box(points_2D):
    """
    Calculate bounding boxes for a list of 2D human keypoints.

    :param human_poses_2d: Nx2 numpy array of 2D points for one human pose.
    :return: Bounding box (xmin, ymin, xmax, ymax).
    """
    x_min = int(np.min(points_2D[:, 0]))
    y_min = int(np.min(points_2D[:, 1]))
    x_max = int(np.max(points_2D[:, 0]))
    y_max = int(np.max(points_2D[:, 1]))
    return x_min, y_min, x_max, y_max


class MVORDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='../MVOR'):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_root = Path(data_root)
        self.mvor_cache = self.data_root / 'mvor_cache'
        self.mvor_cache.mkdir(exist_ok=True)

        self.clips_path = self.data_root / 'clips.csv'
        self.frame_labels_path = self.data_root / 'frame_labels.csv'
        self.splits_path = self.data_root / 'splits.json'
        self.camma_json = self.data_root / 'camma_mvor_2018.json'

        with open(self.splits_path, 'r') as f:
            splits = json.load(f)
        self.split_frames = splits[self.split]

        # determine takes/clips
        self.take_to_start_end_frames = {}
        with open(self.clips_path, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                clip_start_frame = row['Clip start'].strip()
                clip_end_frame = row['Clip end'].strip()
                take_name = str(idx).zfill(3)
                self.take_to_start_end_frames[take_name] = (clip_start_frame, clip_end_frame)

        self.start_end_frames_to_take = {v: k for k, v in self.take_to_start_end_frames.items()}

        # Load frame labels
        self.frame_labels = {}
        with open(self.frame_labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_name = row['Frame number'].strip()
                label = row['Label'].strip()
                self.frame_labels[frame_name] = label

        # Prepare samples
        self.samples = []
        for frame_name in self.split_frames:
            day, frame_id = frame_name.split('_')
            action_label = self.frame_labels[frame_name].lower()
            take_name = None
            for start_end_frames, t_name in self.start_end_frames_to_take.items():
                start_frame, end_frame = start_end_frames
                if frame_name >= start_frame and frame_name <= end_frame:
                    # found the take
                    take_name = t_name
                    break
            sample = {
                'frame_id': frame_id,
                'take_name': take_name,
                'frame_name': frame_name,
                'action_label': action_label,
                'day': day
            }
            self.samples.append(sample)

        # Load camera parameters
        self.camera_params = load_camera_params(self.camma_json)
        self.patient_centers = np.load(self.data_root / 'patient_centers_GT.npz', allow_pickle=True)['arr_0'].item()
        with open(self.data_root / 'dummy_patient_pose.json') as f:
            self.dummy_patient_pose = json.load(f)
            self.dummy_patient_pose = np.array(self.dummy_patient_pose)

        self.key_to_sample_idx = {f'{sample["take_name"]}_{sample["frame_id"]}': idx for idx, sample in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def get_sample_by_key(self, key):
        sample_idx = self.key_to_sample_idx[key]
        # use the get_item function which already takes care of loading multimodal data
        return self[sample_idx]

    def __getitem__(self, index):
        sample = self.samples[index]
        sample['sample_id'] = f'{sample["take_name"]}_{sample["frame_id"]}'
        cache_path = self.mvor_cache / f'{sample["sample_id"]}.npz'
        if not cache_path.exists():
            # Load images
            images = load_image_paths(self.data_root, sample['day'], sample['frame_id'])
            # Load pointcloud if requested
            point_cloud = load_point_cloud(self.data_root, self.camera_params, sample['day'], sample['frame_id'])
            # Load all poses
            human_poses_3d = load_3d_human_poses(self.data_root, sample['day'], sample['frame_id'])
            patient_pose_3d = load_3d_patient_pose(self.dummy_patient_pose, self.patient_centers, sample['day'], sample['frame_id'])
            object_poses_3d = load_3d_object_poses(self.data_root, sample['day'], sample['frame_id'])

            human_poses_2d_per_camera = []
            patient_pose_2d_per_camera = []
            object_poses_2d_per_camera = []
            for cam_id in [1, 2, 3]:
                cam_intrinsics = self.camera_params[cam_id]['intrinsics']
                cam_extrinsics = self.camera_params[cam_id]['extrinsics']
                human_poses_2d = []
                for human_pose_3d in human_poses_3d:
                    human_pose_2d = project_3d_to_2d(human_pose_3d, cam_intrinsics, cam_extrinsics)
                    human_poses_2d.append(human_pose_2d)
                human_poses_2d_per_camera.append(human_poses_2d)
                if patient_pose_3d is not None:
                    patient_pose_2d = project_3d_to_2d(patient_pose_3d, cam_intrinsics, cam_extrinsics)
                else:
                    patient_pose_2d = None
                patient_pose_2d_per_camera.append(patient_pose_2d)

                object_poses_2d = {}
                for object_name, object_pose_3d in object_poses_3d.items():
                    object_pose_2d = project_3d_to_2d(object_pose_3d, cam_intrinsics, cam_extrinsics)
                    object_poses_2d[object_name] = object_pose_2d
                object_poses_2d_per_camera.append(object_poses_2d)

                # from matplotlib import pyplot as plt
                # import matplotlib.patches as patches
                # fig, ax = plt.subplots()
                # img = np.asarray(images[cam_id - 1])
                # ax.imshow(img)
                # for human_pose_2d in human_poses_2d:
                #     x_min, y_min, x_max, y_max = calculate_2D_bounding_box(human_pose_2d)
                #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                #     ax.add_patch(rect)
                # if patient_pose_2d is not None:
                #     x_min, y_min, x_max, y_max = calculate_2D_bounding_box(patient_pose_2d)
                #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='b', facecolor='none')
                #     ax.add_patch(rect)
                # for object_name, object_pose_2d in object_poses_2d.items():
                #     x_min, y_min, x_max, y_max = calculate_2D_bounding_box(object_pose_2d)
                #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='g', facecolor='none')
                #     ax.add_patch(rect)
                #     ax.text(x_min, y_min, object_name, fontsize=12, color='g')
                # plt.savefig('debug.jpg')
                # plt.close()

            # Convert all labels to axis-aligned & scaled & centered coordinates
            if len(human_poses_3d) > 0:
                human_poses_3d = np.stack([transform_to_axis_aligned_points(human_pose_3d) for human_pose_3d in human_poses_3d])
            if patient_pose_3d is not None:
                patient_pose_3d = transform_to_axis_aligned_points(patient_pose_3d)
            object_poses_3d = {k: transform_to_axis_aligned_points(v) for k, v in object_poses_3d.items()}
            sample['human_poses_3d'] = human_poses_3d
            sample['patient_pose_3d'] = patient_pose_3d
            sample['object_poses_3d'] = object_poses_3d
            sample['human_poses_2d_per_camera'] = human_poses_2d_per_camera
            sample['patient_pose_2d_per_camera'] = patient_pose_2d_per_camera
            sample['object_poses_2d_per_camera'] = object_poses_2d_per_camera

            multimodal_data = {
                'azure': images,
                'pc': [point_cloud],
            }
            sample = {
                'sample': sample,
                'multimodal_data': multimodal_data
            }
            np.savez(str(cache_path), sample)

        sample = np.load(str(cache_path), allow_pickle=True)['arr_0'].tolist()
        # if sample['multimodal_data']['pc'] is not a list, make it to a list
        if not isinstance(sample['multimodal_data']['pc'], list):
            sample['multimodal_data']['pc'] = [sample['multimodal_data']['pc']]
        return sample


if __name__ == '__main__':
    mvor_dataset = MVORDataset(split='train')
    sample = mvor_dataset[0]
