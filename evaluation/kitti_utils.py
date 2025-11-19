from mmdet3d.evaluation.functional.kitti_utils import *

def kitti_eval_kradar(gt_annos,
               dt_annos,
               current_classes,
               eval_types=['bev', '3d']):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'
    # kradar have different Iou setting
    overlap_hard = np.array([[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                             [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                             [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
    overlap_mod = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    overlap_easy = np.array([[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])
    min_overlaps = np.stack([overlap_hard, overlap_mod,overlap_easy], axis=0)  # [2, 3, 5]


    class_to_name = {
    0: 'Car',
    1: 'Pedestrian',
    2: 'Cyclist',
    3: 'Van',
    4: 'Car_moving',
    5: 'Pedestrian_moving',
    6: 'Cyclist_moving',
    7: 'Van_moving',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''

    mAP11_bbox, mAP11_bev, mAP11_3d, mAP11_aos, mAP40_bbox, mAP40_bev, \
        mAP40_3d, mAP40_aos = do_eval(gt_annos, dt_annos,
                                      current_classes, min_overlaps,
                                      eval_types)

    ret_dict = {}
    difficulty = ['easy', 'moderate', 'hard']


    # Calculate AP40
    result += '\n----------- AP40 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP40@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP40_bev is not None:
                result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_bev[j, :, i])
            if mAP40_3d is not None:
                result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_3d[j, :, i])

            # replace strict or loose with 0.3/0.5/0.7 and no easy/moderate/hard
            ret_dict['{}_3D_AP40_{}'.format(curcls_name,min_overlaps[i, 0, j])] = mAP40_3d[j, 0, i]
            ret_dict['{}_BEV_AP40_{}'.format(curcls_name,min_overlaps[i, 0, j])] = mAP40_bev[j, 0, i]

    
    ret_dict['3D_mAP40_0.3']=np.mean(mAP40_3d[:,0,-1])
    ret_dict['BEV_mAP40_0.3']=np.mean(mAP40_bev[:,0,-1])

    ret_dict['3D_mAP40_0.5']=np.mean(mAP40_3d[:,0,-2])
    ret_dict['BEV_mAP40_0.5']=np.mean(mAP40_bev[:,0,-2])
    
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP40@{}, {}, {}:\n'.format(*difficulty))
        if mAP40_bev is not None:
            mAP40_bev = mAP40_bev.mean(axis=0)
            result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP40_bev[:, 0])
        if mAP40_3d is not None:
            mAP40_3d = mAP40_3d.mean(axis=0)
            result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP40_3d[:,
                                                                            0])


    return result, ret_dict
