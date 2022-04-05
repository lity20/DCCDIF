import os
import sys
import importlib
from datetime import datetime
import numpy as np
import pandas
import trimesh
from scipy.spatial import cKDTree as KDTree


def evaluate(config):
    with open(config['evaluate']['split_file'], 'r') as f:
        shape_ids = f.readlines()
    shape_ids = [l.rstrip() for l in shape_ids]

    cd_list = np.zeros(len(shape_ids)) # chamfer distance
    nc_list = np.zeros(len(shape_ids)) # normal consistency
    fs_list = np.zeros(len(shape_ids)) # f score
    for idx, shape_id in enumerate(shape_ids):
        print('Evaluation progress: %d/%d...' % (idx, len(shape_ids)), end='\r')

        pred_mesh_path = os.path.join(config['evaluate']['recon_mesh_dir'], shape_id+'.ply')
        gt_mesh_path = os.path.join(config['evaluate']['gt_mesh_dir'], shape_id+'.obj')
        pred_mesh = trimesh.load(pred_mesh_path)
        gt_mesh = trimesh.load(gt_mesh_path)

        pred_points, pred_indices = pred_mesh.sample(config['evaluate']['num_surface_samples'], return_index=True)
        pred_points = pred_points.astype(np.float32)
        pred_normals = pred_mesh.face_normals[pred_indices]

        gt_points, gt_indices = gt_mesh.sample(config['evaluate']['num_surface_samples'], return_index=True)
        gt_points = gt_points.astype(np.float32)
        gt_normals = gt_mesh.face_normals[gt_indices]

        kdtree = KDTree(gt_points)
        dist_p2g, indices_p2g = kdtree.query(pred_points)

        kdtree = KDTree(pred_points)
        dist_g2p, indices_g2p = kdtree.query(gt_points)

        normals_p2g = gt_normals[indices_p2g]
        nc_p2g = np.abs(np.sum(normals_p2g * pred_normals, axis=1))
        normals_g2p = pred_normals[indices_g2p]
        nc_g2p = np.abs(np.sum(normals_g2p * gt_normals, axis=1))
        nc = 0.5 * (np.mean(nc_p2g) + np.mean(nc_g2p))

        precision = np.mean((dist_p2g**2 <= config['evaluate']['f_score_tau']**2).astype(np.float32)) * 100.0
        recall = np.mean((dist_g2p**2 <= config['evaluate']['f_score_tau']**2).astype(np.float32)) * 100.0
        fs = (2 * precision * recall) / (precision + recall + 1e-9)

        cd = 1000.0 * (np.mean(dist_p2g**2) + np.mean(dist_g2p**2))

        cd_list[idx] = cd
        nc_list[idx] = nc
        fs_list[idx] = fs

    print('Evaluation progress: %d/%d. Done.' % (len(shape_ids), len(shape_ids)))
    print('Chamfer-L2: %f, Normal Consistency: %f, F-Score: %f.' %(np.mean(cd_list), np.mean(nc_list), np.mean(fs_list)))

    classes = [l.split('/')[0] for l in shape_ids]
    shape_ids = [l.split('/')[1] for l in shape_ids]
    data_frame = pandas.DataFrame({'class': classes, 'shape_id': shape_ids,
                                   'cd': cd_list, 'nc': nc_list, 'fs': fs_list})
    data_frame.to_csv(config['experiment']['eval_results_save_path'], index=False, sep=',')
    print('Results are saved to ' + config['experiment']['eval_results_save_path'])


if __name__ == '__main__':
    config = importlib.import_module(sys.argv[1]).config
    evaluate(config)
