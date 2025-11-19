import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes,LiDARPoints
from mmcv.transforms import BaseTransform, Compose
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

import torch
import random
from collections import defaultdict
from mmengine.logging import print_log
from pypcd4 import PointCloud
from typing import List, Optional, Sequence, Tuple, Union

@TRANSFORMS.register_module()
class LoadKradarFrame(BaseTransform):
    def __init__(self,roi=None,input_dim=5,normalize=False):
        self.roi=roi
        self.input_dim=input_dim
        self.normalize=normalize
        pass

    def transform(self, results):
        radar_path = results['radar']
        if '.pcd' in radar_path:
            if self.input_dim == 5:
                points = PointCloud.from_path(radar_path).numpy(('y', 'x', 'z', 'vr', 'power'))
            elif self.input_dim == 6:
                points = PointCloud.from_path(radar_path).numpy(('y', 'x', 'z', 'vr', 'power', 'snr'))
            elif self.input_dim == 4:
                points = PointCloud.from_path(radar_path).numpy(('y', 'x', 'z', 'power'))
            elif self.input_dim == 3:
                points = PointCloud.from_path(radar_path).numpy(('y', 'x', 'z'))
        elif '.npy' in radar_path:
            points = np.load(radar_path)
            if self.input_dim == 3:
                points=points[:,:3]

        # x and y switch
        # ('x', 'y', 'z', 'vr', 'power', 'snr')
        if self.roi is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi
            temp_data = points
            temp_data = temp_data[np.where(
                    (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                    (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                    (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
            points= temp_data

        points = LiDARPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        results['points'] = points
        return results

@TRANSFORMS.register_module()
class CAPMix(BaseTransform):
    def __init__(self, points_dim: int = 5, prob: float = 0.5, roi=None,
                 pre_transform: Optional[Sequence[dict]] = None,):
        """Initialize plugin

        :param config: config as dict
        :param points_dim: Point cloud dimension. Defaults to 5.
        """
        self._points_dim = points_dim
        self.prob=prob
        self.roi = roi
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

        self.voxel_generator = VoxelizationByGridShape(
            point_cloud_range=self.roi,
                max_num_points=1000,
                voxel_size=[2,2,9.6]
        )
        print_log(f'voxel_generator_size for pillarmix: {self.voxel_generator.voxel_size}')
            

        self.moderate_list=[2,2]
        self.sparse_list=[0.1,5]
        self.dense_list=[10,1]

        print_log(f'dynamic mixup beta parameter moderate_list: {self.moderate_list}')
        print_log(f'dynamic mixup beta parameter sparse_list: {self.sparse_list}')
        print_log(f'dynamic mixup beta parameter dense_list: {self.dense_list}')

    
    def transform(self, input_dict):
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through CAPMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before lasermix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = DynamicPillarMix(input_dict, mix_results,self.roi,self.voxel_generator,self.moderate_list,self.sparse_list,self.dense_list)

        return input_dict
            
def DynamicPillarMix(data_source,data_target,pc_range=[-50, -50, -5, 50, 50, 3],
                             voxel_generator=None,moderate_list=[2,2],sparse_list=[2,2],dense_list=[0.5,5]):

    all_class=np.concatenate((data_source['ann_info']['gt_labels_3d'],data_target['ann_info']['gt_labels_3d']),axis=0)
    all_class=np.unique(all_class)
    

    voxels1, coords1, num_points1=voxel_generator(data_source['points'].tensor)
    voxels2,coords2,num_points2=voxel_generator(data_target['points'].tensor)

    exchange_ratio=random.uniform(0.3,0.6)
    num_swap = min([int(len(voxels1) * exchange_ratio),int(len(voxels2) * exchange_ratio)])
    bboxes1 = data_source['ann_info']['gt_bboxes_3d']  # LiDARInstance3DBoxes 
    bboxes2 = data_target['ann_info']['gt_bboxes_3d']

    bbox_pillar_indices1 = map_bboxes_to_pillars_by_corners(bboxes1, voxel_generator.voxel_size, pc_range)
    bbox_pillar_indices2 = map_bboxes_to_pillars_by_corners(bboxes2, voxel_generator.voxel_size, pc_range)

    pillar2cls_dict1=defaultdict(list)
    for i,cls in enumerate(data_source['ann_info']['gt_labels_3d']):
        this_bbox_pillar=bbox_pillar_indices1[i]
        for pillar_idx in this_bbox_pillar:
            pillar2cls_dict1[tuple(pillar_idx)].append(cls)

    pillar2cls_dict2=defaultdict(list)
    for i,cls in enumerate(data_target['ann_info']['gt_labels_3d']):
        this_bbox_pillar=bbox_pillar_indices2[i]
        for pillar_idx in this_bbox_pillar:
            pillar2cls_dict2[tuple(pillar_idx)].append(cls)


    swap_indices = np.random.choice(len(voxels1), num_swap, replace=False)
    
    # switch the voxels here the coords are already sparse format, so not aligned between frames
    swapped_coords = coords1[swap_indices]


    points=[]

    for coord in coords1:
        lam_moderate = np.random.beta(moderate_list[0],moderate_list[1])
        lam_sparse=np.random.beta(sparse_list[0],sparse_list[1])
        lam_dense=np.random.beta(dense_list[0],dense_list[1])
        lam_envs=lam_moderate
        # per pillar operation
        swapped_match = torch.any(torch.all(swapped_coords == coord, dim=1))
        is_in_coords2 = torch.any(torch.all(coords2 == coord, dim=1))
        
        if is_in_coords2:
            matched_idx=torch.where((coords2 == coord).all(dim=1))[0].item()
            voxel = voxels2[matched_idx]
            n_points = num_points2[matched_idx]
            points_in_voxel = voxel[:n_points]
            to_mix_target=shuffle_points(points_in_voxel)
        
        matched_idx=torch.where((coords1 == coord).all(dim=1))[0].item()
        voxel = voxels1[matched_idx]
        n_points = num_points1[matched_idx]
        points_in_voxel = voxel[:n_points]
        to_mix_source=shuffle_points(points_in_voxel)

        # class-0: car, class-1: large vehicle, class-2: pedestrian, class-3: cyclist
        if swapped_match and (tuple(coord.tolist()) in pillar2cls_dict1 or tuple(coord.tolist()) in pillar2cls_dict2):
            cls_list1=[]
            cls_list2=[]
            class_flag='b'
            if tuple(coord.tolist()) in pillar2cls_dict1:
                cls_list1=pillar2cls_dict1[tuple(coord.tolist())]
            if tuple(coord.tolist()) in pillar2cls_dict2:
                cls_list2=pillar2cls_dict2[tuple(coord.tolist())]
            if 2 in cls_list1 or 3 in cls_list1:
            # high mix_rate for sparse points
                class_flag='c'
                if is_in_coords2:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_sparse)]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]* (1 - lam_sparse))]
                    points_in_voxel = np.concatenate((to_mix_source,to_mix_target), axis=0)
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_sparse)]
                    points_in_voxel=to_mix_source

            elif 2 in cls_list2 or 3 in cls_list2:
                class_flag='c'
                if is_in_coords2:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* (1 - lam_sparse))]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]* lam_sparse)]
                    points_in_voxel = np.concatenate((to_mix_source,to_mix_target), axis=0)
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]*(1 - lam_sparse))]
                    points_in_voxel=to_mix_source

            # dense object pillars
            elif 1 in cls_list2 and 0 not in cls_list1 and 0 not in cls_list2:
                class_flag='r'
                if is_in_coords2:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]*(1 - lam_dense))]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]*lam_dense)]
                    points_in_voxel = np.concatenate((to_mix_source,to_mix_target), axis=0)
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]*(1 - lam_dense))]
                    points_in_voxel=to_mix_source
                
            
            elif 1 in cls_list1 and 0 not in cls_list1 and 0 not in cls_list2:
                class_flag='r'

                if is_in_coords2:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_dense)]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]* (1 - lam_dense))]
                    points_in_voxel = np.concatenate((to_mix_source,to_mix_target), axis=0)
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_dense)]
                    points_in_voxel=to_mix_source
            
            # only moderate pillars
            elif 0 in cls_list1:
                class_flag='b'
                # mixup all the points 
                if is_in_coords2:    
                    to_mix_source=to_mix_source[:int( to_mix_source.shape[0]* lam_moderate)]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]* (1 - lam_moderate))]
                    points_in_voxel = np.concatenate((to_mix_source, to_mix_target), axis=0)                       
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_moderate)]
                    points_in_voxel=to_mix_source
            elif 0 in cls_list2:
                class_flag='b'  
                if is_in_coords2:
                    to_mix_source=to_mix_source[:int( to_mix_source.shape[0]* (1 - lam_moderate))]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]*lam_moderate )]
                    points_in_voxel = np.concatenate((to_mix_source, to_mix_target), axis=0)                       
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* (1 - lam_moderate))]
                    points_in_voxel=to_mix_source
            
            else:
                class_flag='b'
                # no object in the pillar
                if is_in_coords2:
                    to_mix_source=to_mix_source[:int( to_mix_source.shape[0]* lam_envs)]
                    to_mix_target=to_mix_target[:int(to_mix_target.shape[0]* (1 - lam_envs))]
                    points_in_voxel = np.concatenate((to_mix_source, to_mix_target), axis=0)                       
                else:
                    to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_envs)]
                    points_in_voxel=to_mix_source

        else:
            class_flag='b'
            # mixup all the unselected points
            if is_in_coords2:
                
                to_mix_source=to_mix_source[:int( to_mix_source.shape[0]* lam_envs)]
                to_mix_target=to_mix_target[:int(to_mix_target.shape[0]* (1 - lam_envs))]
                points_in_voxel = np.concatenate((to_mix_source, to_mix_target), axis=0)                       
            else:
                to_mix_source=to_mix_source[:int(to_mix_source.shape[0]* lam_envs)]
                points_in_voxel=to_mix_source

        points.append(points_in_voxel)
 
            
    # mixup the points which is only in target frame but not source frame
    # lam = np.random.beta(2, 2)
    for i,coord in enumerate(coords2):
        lam = np.random.beta(2, 2)
        is_in_coords1 = torch.any(torch.all(coords1 == coord, dim=1))
        if not is_in_coords1:
            class_flag='b'
            voxel = voxels2[i]
            n_points = num_points2[i]
            points_in_voxel = voxel[:n_points]
            shuffle_points(points_in_voxel)
            points_to_mix=points_in_voxel[:int(points_in_voxel.shape[0]* lam)]
            points.append(points_to_mix)


    # delete the replicated indices
    gt_bboxes_3d = np.concatenate((data_source['ann_info']['gt_bboxes_3d'].tensor.numpy(),
                                    data_target['ann_info']['gt_bboxes_3d'].tensor.numpy()), axis=0)
    gt_labels_3d=np.concatenate((data_source['ann_info']['gt_labels_3d'],
                               data_target['ann_info']['gt_labels_3d']), axis=0)
    

    data_source['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0))
    data_source['ann_info']['gt_labels_3d'] = torch.tensor(gt_labels_3d, dtype=torch.long)

    reconstructed_points = np.vstack(points)

    if len(reconstructed_points)==0:
        return None
    
    data_source['points']=LiDARPoints(
                reconstructed_points, points_dim=reconstructed_points.shape[-1], attribute_dims=None)
    
    return data_source

def map_bboxes_to_pillars_by_corners(bboxes, voxel_size, point_cloud_range):
    bbox_pillar_indices = []
    point_cloud_range = np.array(point_cloud_range)
    voxel_size = np.array(voxel_size)
    grid_size = ((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(int)
    corners_bboxes = bboxes.corners.numpy()
    for corners in corners_bboxes:
        # Get x, y, z coordinates separately
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        z_coords = corners[:, 2]

        # Calculate the min and max of each coordinate
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()

        # Convert the min and max coordinates to pillar indices
        min_coords = np.array([x_min, y_min, z_min])
        max_coords = np.array([x_max, y_max, z_max])

        min_indices = np.floor((min_coords - point_cloud_range[:3]) / voxel_size).astype(int)
        max_indices = np.floor((max_coords - point_cloud_range[:3]) / voxel_size).astype(int)

        # Clip the indices to valid range
        min_indices = np.clip(min_indices, [0, 0, 0], grid_size - 1)
        max_indices = np.clip(max_indices, [0, 0, 0], grid_size - 1)

        # Generate index ranges for each axis
        x_indices = np.arange(min_indices[0], max_indices[0] + 1)
        y_indices = np.arange(min_indices[1], max_indices[1] + 1)
        z_indices = np.arange(min_indices[2], max_indices[2] + 1)

        # Generate all combinations of indices
        xv, yv, zv = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        indices = np.vstack((zv.ravel(), yv.ravel(), xv.ravel())).T  # Convert to (z, y, x) order

        bbox_pillar_indices.append(indices)

    return bbox_pillar_indices

def shuffle_points(points):
    # Input: points in numpy array
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    return points