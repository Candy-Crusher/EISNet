# ---------------------------------------------------------------
# Code adapted from https://github.com/uzh-rpg/ess/blob/main/DSEC/dataset/sequence.py.
# ---------------------------------------------------------------

from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from joblib import Parallel, delayed
from datasets.extract_data_tools.DSEC.eventslicer import EventSlicer
import albumentations as A
import datasets.data_util as data_util
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
import re


class Sequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   └──data
    # │   │   │       ├── 000000.png
    # │   │   │       └── ...
    # │   │   └── 19classes
    # │   │       └──data
    # │   │           ├── 000000.png
    # │   │           └── ...
    # │   └── timestamps.txt
    # ├── events
    # │   └── left
    # │       ├── events.h5
    # │       └── rectify_map.h5
    # └── images
    #     └── left
    #         ├── rectified
    #         │    ├── 000000.png
    #         │    └── ...
    #         ├── ev_inf
    #         │    ├── 000000.png
    #         │    └── ...
    #         └── timestamps.txt
    #
    # Note: Folder "ev_inf" contains paired image samples that are spatially aligned with the event data.

    def __init__(self, seq_name: str, mode: str='train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20, nr_events_per_data: int = 100000,
                 nr_bins_per_data: int = 5, require_paired_data=False, normalize_event=False, separate_pol=False,
                 semseg_num_classes: int = 11, augmentation=False, fixed_duration=False, remove_time_window: int = 250,
                 random_crop=False):
        assert nr_bins_per_data >= 1
        # assert seq_path.is_dir()
        # self.sequence_name = seq_name
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.random_crop = random_crop
        self.shape_crop = [448, 448]

        # Set event representation
        self.nr_events_data = nr_events_data
        self.num_bins = nr_bins_per_data
        self.nr_events_per_data = nr_events_per_data
        self.event_representation = event_representation
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.locations = ['left']
        self.semseg_num_classes = semseg_num_classes
        self.augmentation = augmentation

        # Save delta timestamp
        self.fixed_duration = fixed_duration
        if self.fixed_duration:
            delta_t_ms = nr_events_data * delta_t_per_data
            self.delta_t_us = delta_t_ms * 1000
        self.remove_time_window = remove_time_window

        self.require_paired_data = require_paired_data

        # load timestamps
        # seq_name = os.path.basename(seq_path)
        seq_name = seq_name
        # self.timestamps = np.loadtxt(str(seq_path / f'{seq_name}_semantic_timestamps.txt'), dtype='int64')
        # 从第11个开始 每隔9个取一个
        # self.timestamps = self.timestamps[11::9]
        # self.timestamps = self.timestamps[12::9]
        # my_root = Path('/mnt/sdc/lxy/datasets/DSEC/DSEC/dsec_anytime_100_N/day/')
        my_root = Path('/mnt/sdc/lxy/datasets/DSEC/DSEC/dsec_anytime_100/synthesis_N11_45_100/')
        # load label paths
        if self.semseg_num_classes == 11:
            label_dir = my_root / 'gtFine_t2' / mode / seq_name
        elif self.semseg_num_classes == 19:
            label_dir = seq_path / '19classes'
        else:
            raise ValueError
        print(label_dir)
        assert label_dir.is_dir()
        label_pathstrings = list()
        for entry in label_dir.iterdir():
            if str(entry.name).endswith('_gtFine_labelTrainIds11.png'):
                label_pathstrings.append(str(entry))
        label_pathstrings.sort()
        self.label_pathstrings = label_pathstrings
        # assert len(self.label_pathstrings) == self.timestamps.size

        # load images paths
        if self.require_paired_data:
            img_left_dir = my_root / 'leftImg8bit_t0' / mode/seq_name
            assert img_left_dir.is_dir()
            img_left_pathstrings = list()
            for entry in img_left_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_left_pathstrings.append(str(entry))
            img_left_pathstrings.sort()
            self.img_left_pathstrings = img_left_pathstrings
            # assert len(self.img_left_pathstrings) == self.timestamps.size
            # print(f'{seq_name}: Found {len(self.img_left_pathstrings)} paired image samples.')

        # # Remove several label paths and corresponding timestamps in the remove_time_window (i.e., the first six samples).
        # # This is necessary because we do not have enough events before the first label.
        # self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]
        # del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        # assert len(self.label_pathstrings) == self.timestamps.size
        # if self.require_paired_data:
        #     del self.img_left_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        #     assert len(self.img_left_pathstrings) == self.timestamps.size

        # self.h5f = dict()
        # self.rectify_ev_maps = dict()
        # self.event_slicers = dict()

        # ev_dir = seq_path / 'events'
        # for location in self.locations:
        #     ev_dir_location = ev_dir / location
        #     ev_data_file = ev_dir_location / 'events.h5'
        #     ev_rect_file = ev_dir_location / 'rectify_map.h5'

        #     h5f_location = h5py.File(str(ev_data_file), 'r')
        #     self.h5f[location] = h5f_location
        #     self.event_slicers[location] = EventSlicer(h5f_location)
        #     with h5py.File(str(ev_rect_file), 'r') as h5_rect:
        #         self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        # Data Augmentation Configuration
        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.RandomScale(scale_limit=(0.02, 0.2), p=1),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
                A.HorizontalFlip(p=0.5)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32') / 256

    @staticmethod
    def get_img(filepath: Path):
        assert filepath.is_file()
        img = Image.open(str(filepath))
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = img_transform(img)
        return img_tensor

    @staticmethod
    def get_label(filepath: Path):
        assert filepath.is_file()
        label = Image.open(str(filepath))
        label = np.array(label)
        return label

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return len(self.label_pathstrings)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert np.max(x) < self.width
        assert np.max(y) < self.height
        return rectify_map[y, x]

    def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = data_util.generate_input_representation(events_temp, self.event_representation,
                                                                       (self.height, self.width),
                                                                        nr_temporal_bins=self.num_bins,
                                                                        separate_pol=self.separate_pol)
        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)
        num_bins_temp = event_representation.shape[0]
        event_tensor[(job_id * num_bins_temp):((job_id+1) * num_bins_temp), :, :] = event_representation

    def apply_augmentation(self, transform_a, events, images, label):
        if self.require_paired_data:
            A_data = transform_a(image=images.permute(1, 2, 0).numpy(), mask=label)
            img_tensor = torch.from_numpy(A_data['image']).permute(2, 0, 1)
            label = A_data['mask']
            if self.random_crop and self.mode == 'train':
                events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
            else:
                events_tensor = events
            for k in range(events.shape[0]):
                events_tensor[k, :, :] = torch.from_numpy(
                    A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
            return events_tensor, img_tensor, label
        else:
            A_data = transform_a(image=events[0, :, :].numpy(), mask=label)
            label = A_data['mask']
            if self.random_crop and self.mode == 'train':
                events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
            else:
                events_tensor = events
            for k in range(events.shape[0]):
                events_tensor[k, :, :] = torch.from_numpy(
                    A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
            return events_tensor, label

    def __getitem__(self, index):
        output = {}
        # Load the ground truth
        label_path = Path(self.label_pathstrings[index])
        # print(label_path)
        label = self.get_label(label_path)
        # # Load the events ang generate a frame-based representation
        # ts_end = self.timestamps[index]
        # event_tensor = None
        # for location in self.locations:
        #     if self.fixed_duration:
        #         ts_start = ts_end - self.delta_t_us
        #         self.delta_t_per_data_us = self.delta_t_us / self.nr_events_data
        #         for i in range(self.nr_events_data):
        #             t_s = ts_start + i * self.delta_t_per_data_us
        #             t_end = ts_start + (i+1) * self.delta_t_per_data_us
        #             event_data = self.event_slicers[location].get_events(t_s, t_end)
        #             p = event_data['p']
        #             t = event_data['t']
        #             x = event_data['x']
        #             y = event_data['y']

        #             xy_rect = self.rectify_events(x, y, location)
        #             x_rect = xy_rect[:, 0]
        #             y_rect = xy_rect[:, 1]

        #             if self.event_representation == 'voxel_grid':
        #                 events = np.stack([x_rect, y_rect, t, p], axis=1)
        #                 event_representation = data_util.generate_input_representation(events, self.event_representation,
        #                                                                                (self.height, self.width),
        #                                                                                 nr_temporal_bins=self.num_bins,
        #                                                                                 separate_pol=self.separate_pol)
        #                 event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)
        #             else:
        #                 events = np.stack([x_rect, y_rect, t, p], axis=1)
        #                 event_representation = data_util.generate_input_representation(events, self.event_representation,
        #                                                           (self.height, self.width),nr_temporal_bins=self.num_bins,)
        #                 event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)

        #             if event_tensor is None:
        #                 event_tensor = event_representation
        #             else:
        #                 event_tensor = torch.cat([event_tensor, event_representation], dim=0)

        #     else:
        #         if self.separate_pol:
        #             num_bins_total = self.nr_events_data * self.num_bins * 2
        #         else:
        #             num_bins_total = self.nr_events_data * self.num_bins
        #         if self.event_representation == 'AET':
        #             event_tensor = torch.zeros((num_bins_total * 2, self.height, self.width))
        #         else:
        #             event_tensor = torch.zeros((num_bins_total, self.height, self.width))
        #         self.nr_events = self.nr_events_data * self.nr_events_per_data
        #         event_data = self.event_slicers[location].get_events_fixed_num(ts_end, self.nr_events)

        #         if self.nr_events >= event_data['t'].size:
        #             start_index = 0
        #         else:
        #             start_index = -self.nr_events

        #         p = event_data['p'][start_index:]
        #         t = event_data['t'][start_index:]
        #         x = event_data['x'][start_index:]
        #         y = event_data['y'][start_index:]
        #         nr_events_loaded = t.size

        #         xy_rect = self.rectify_events(x, y, location)
        #         x_rect = xy_rect[:, 0]
        #         y_rect = xy_rect[:, 1]
        #         events = np.stack([x_rect, y_rect, t, p], axis=-1)


        #         nr_events_temp = nr_events_loaded // self.nr_events_data
        #         Parallel(n_jobs=8, backend="threading")(
        #             delayed(self.generate_event_tensor)(i, events, event_tensor, nr_events_temp)
        #             for i in range(self.nr_events_data))

        #     # Remove 40 bottom rows
        #     event_tensor = event_tensor[:, :-40, :]

        # Generate the event-image pair
        img_tensor = None
        if self.require_paired_data:
            img_left_path = Path(self.img_left_pathstrings[index])
            # print(img_left_path)
            img_tensor = self.get_img(img_left_path)[:3, ...]
        event_path_t0_t1 = str(img_left_path).replace('leftImg8bit_t0', 'event_t0_t1/event_20').replace('.png', '.npy')
        event_t0_t1 = np.load(event_path_t0_t1)
        event_t0_t1 = torch.from_numpy(event_t0_t1).float()

        event_path_t1_t2 = str(img_left_path).replace('leftImg8bit_t0', 'event_t1_t2/event_20').replace('.png', '.npy')
        
        filename = os.path.basename(event_path_t1_t2)
        pattern = re.compile(r'(\d+).npy')

        match = pattern.match(filename)
        if match:
            # 提取编号部分
            number = match.group(1)
            # 构建新的文件名
            new_filename = f'{int(number)+5:06d}.npy'
            # 构建完整的路径名
            event_path_t1_t2 = event_path_t1_t2.replace(f'{number}.npy', new_filename)
        event_t1_t2 = np.load(event_path_t1_t2)[:19]
        event_t1_t2 = torch.from_numpy(event_t1_t2).float()

        event_tensor = torch.cat((event_t0_t1, event_t1_t2.float()), dim=0)
        # bin 40 cat 成 bin 3
        bin=13
        event_tensor = torch.cat([event_tensor[bin*i:bin*(i+1)].mean(0).unsqueeze(0) for i in range(3)], dim=0)
        event_tensor = event_tensor[:, :-40, :]
        # Data augmentation
        if self.random_crop and self.mode == 'train':
            if self.augmentation:
                if self.require_paired_data:
                    event_tensor, img_tensor, label = self.apply_augmentation(self.transform_a_random_crop,
                                                                              event_tensor, img_tensor, label)
                else:
                    event_tensor, label = self.apply_augmentation(self.transform_a_random_crop, event_tensor,
                                                                  img_tensor, label)
        else:
            if self.augmentation:
                if self.require_paired_data:
                    event_tensor, img_tensor, label = self.apply_augmentation(self.transform_a, event_tensor,
                                                                              img_tensor, label)
                else:
                    event_tensor, label = self.apply_augmentation(self.transform_a, event_tensor, img_tensor, label)

        label_tensor = torch.from_numpy(label).long()
        if 'representation' not in output:
            output['representation'] = dict()
        output['representation']['left'] = event_tensor
        output['img_left'] = img_tensor

        if self.require_paired_data:
            return output['representation']['left'], output['img_left'], label_tensor
        else:
            return output['representation']['left'], label_tensor
