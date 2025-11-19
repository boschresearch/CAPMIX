from os import path as osp

import numpy as np
import os
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import Box3DMode, Coord3DMode,CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.registry import DATASETS
import pickle
from mmengine.logging import print_log
import torch
import copy
from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Dict
import math

@DATASETS.register_module()
class KradarDataset(Det3DDataset):
    METAINFO = {
        "classes": ['Sedan','Bus or Truck'],
        "box_type_3d": "lidar",
    }
    def __init__(self,
                 data_root,
                 ann_file,
                 folder_name,
                 classes,
                 ratio=1,
                 with_velocity=False,
                 test_mode=False,
                 box_type_3d='LiDAR',
                filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 eval_mode='kitti',
                 select_mode='frames',
                 point_cloud_range=None,
                 **kwargs):
        self.ratio=ratio
        self.evaluate_mode=eval_mode
        self.point_cloud_range = point_cloud_range
        self.select_mode=select_mode
        self.Kradar2KITTIMapping={'Sedan':'Car',
                                  'Bus or Truck':'Van',
                                    'Background': 0,}
        self.CLASSES = classes
        self.Kradar2KITTIMapping=defaultdict(lambda:-1,self.Kradar2KITTIMapping)
        self.folder_name=folder_name
        super(KradarDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,)
        
    def load_data_list(self):
        anno_file=self.ann_file
        if self.ratio<1:
            pkl_path = Path(anno_file)
            if self.select_mode=='seqs':
                new_file_name = f"{pkl_path.stem}_{self.ratio}_seqs{pkl_path.suffix}"
            elif self.select_mode=='frames':
                new_file_name = f"{pkl_path.stem}_{self.ratio}_frames{pkl_path.suffix}"
            new_file_path = pkl_path.parent / new_file_name
            if os.path.exists(new_file_path):
                with open(new_file_path,'rb') as f:
                    data =pickle.load(f)
            else:
                with open(anno_file, 'rb') as f:
                    data = pickle.load(f)
                # select by seqs, rather than frames
                if self.select_mode=='seqs':
                    seqs_num=len(set([d['meta']['seq'] for d in data]))
                    select_num=math.ceil(seqs_num*self.kradar_ratio)
                    # doesnt make sense for new-split, since it doesnt contains all the seqs, but doesn't affect full set result
                    # random_seqs = random.sample(range(1, seqs_num+1), select_num)
                    random_seqs =random.sample(list(set([d['meta']['seq'] for d in data])), select_num)
                    data=[d for d in data if int(d['meta']['seq']) in random_seqs]
                elif self.select_mode=='frames':
                    random.shuffle(data)
                    data=data[:int(len(data)*self.ratio)]
                with open(new_file_path,'wb') as f:
                    pickle.dump(data,f)
        else:
            with open(anno_file, 'rb') as f:
                data = pickle.load(f)
            # seqs=set([d['meta']['seq'] for d in data])
            # print(seqs)
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        filter_empty_data=[]
        for d in data:
            seq = d['meta']['seq']
            rdr_idx = d['meta']['idx']['rdr']   
            # dx, dy, dz = d['meta']['calib']   
            d['token']=seq+'_'+str(rdr_idx)
            should_add = False
            for obj in d['meta']['label']:
                x,y,z=obj[1][0:3]
                # already in the radar coordinate
                # filter out the objects that are not in the point cloud range and only contains other classes
                if obj[0] in self.CLASSES and \
                (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                    should_add=True
                    break
            if should_add:
                filter_empty_data.append(d)

        print_log(f"Annotation file {anno_file} loaded.")
        print_log(f"Selected {len(filter_empty_data)} frames from {len(data)} frames.")

        raw_data_list = filter_empty_data

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')
            
        return data_list
    
    def prepare_data(self, idx: int) -> dict:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            dict: Results passed through ``self.pipeline``.
        """
        if not self.test_mode:
            data_info = self.get_data_info(idx)
            # Pass the dataset to the pipeline during training to support mixed
            # data augmentation, such as polarmix and lasermix.
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

    def parse_data_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Process the 'raw data info' elements.
        This method is called by BaseDataset.__init__ before starting the training/testing and is used to parse
        the elemets of the train|val|test_info.pkl file.
        We override this method since our dataset has a different structure than expected by the base class method.
        :param info: Raw info dict representing a data frame.
        :returns: Dict with ann_info/eval_ann_info for training/testing,validation mode.
        """

        # Since mmdetect3d works with specific keys like lidar_points in later stages we treat the radar sensor as
        # a pseudo lidar.
        dir_rdr_sparse = self.data_root
        folder_name=self.folder_name
        seq = info['meta']['seq']
        rdr_idx = info['meta']['idx']['rdr']
        if 'sparse_radar' in dir_rdr_sparse:
            # load old-split 1 point cloud
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'sprdr_{rdr_idx}.npy')
        else: 
            # load new-split cfar pointcloud
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq,folder_name, f'pointcloud_{rdr_idx}.pcd')
        info["lidar_points"] = {
            "lidar_path": path_rdr_sparse,  # Expected in various classes in mmdetect like visualization_hook.py
            # We do not support this feature. The entry is added to avoid triggering an assertion.
            "num_pts_feats": 5,  # Used in visualization_hook.py
        }
        info['radar'] = path_rdr_sparse

        if not self.test_mode:
            info["ann_info"] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Process the instances in 'raw data info' to 'ann_info' and drops don't care objects.
        Method is implemented according to the reference guide:
        https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/customize_dataset.md
        :param info: Data information of single data sample.
        :returns: Annotation dictionary with following keys:
                    - gt_bboxes_3d: 3D ground truth bboxes.
                    - gt_labels_3d: Labels of ground truths.
        """


        annos = info['meta']['label']
        # we need other objects to avoid collision when sample
        loc=[]
        dims=[]
        rots=[]
        gt_names=[]
        for cls,anno,_,_ in annos:
            if cls in self.CLASSES:
                loc.append(np.array(anno[0:3]).reshape(1,3))
                dims.append(np.array(anno[4:7]).reshape(1,3))
                rots.append(np.array(anno[3]).reshape(1,1))   
                gt_names.append(cls)

        loc=np.concatenate(loc,axis=0)
        dims=np.concatenate(dims,axis=0)
        rots=np.concatenate(rots,axis=0)
        gt_bboxes_3d = np.concatenate([loc, dims, rots],
                                    axis=1).astype(np.float32)

        # 3d box center
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5))

        # selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        # gt_names = gt_names[selected]

        gt_labels = []
        gt_names_3d = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
                gt_names_3d.append(cat)
            else:
                gt_labels.append(-1)
                gt_names_3d.append('unknown')
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        ann_info = {
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,}
        
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1
        return ann_info