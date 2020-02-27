# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import csv
import importlib
import matplotlib
matplotlib.use('Agg')
import models
import numpy as np
import os
import tensorflow as tf
import time
from open3d import *
from io_util import read_pcd, save_pcd
from tf_util import chamfer, earth_mover
from visu_util import plot_pcd_three_views
from data_util import resample_pcd


def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 6))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0), args.num_channel)

    output = tf.placeholder(tf.float32, (1, args.num_gt_points, 14))
    cd_op = chamfer(output[:,:,0:3], gt[:,:,0:3])
    emd_op = earth_mover(output[:,:,0:3], gt[:,:,0:3])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    os.makedirs(args.results_dir, exist_ok=True)
    csv_file = open(os.path.join(args.results_dir, 'results.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'cd', 'emd'])

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    total_time = 0
    total_cd = 0
    total_emd = 0
    cd_per_cat = {}
    emd_per_cat = {}
    np.random.seed(1)
    for i, model_id in enumerate(model_list):
        if args.experiment == 'shapenet':
            synset_id, model_id = model_id.split('/')
            partial = read_pcd(os.path.join(args.data_dir, 'partial', synset_id, '%s.pcd' % model_id))
            complete = read_pcd(os.path.join(args.data_dir, 'complete', synset_id, '%s.pcd' % model_id))
        elif args.experiment == 'suncg':
            synset_id = 'all_rooms'
            partial = read_pcd(os.path.join(args.data_dir, 'pcd_partial', '%s.pcd' % model_id))
            complete = read_pcd(os.path.join(args.data_dir, 'pcd_complete', '%s.pcd' % model_id))
        rotate = False
        if rotate:    
            angle = np.random.rand(1)*360
            partial = np.stack([np.cos(angle)*partial[:,0] - np.sin(angle)*partial[:,2], partial[:,1], np.sin(angle)*partial[:,0] + np.cos(angle)*partial[:,2]], axis=-1)
            complete = np.stack([np.cos(angle)*complete[:,0] - np.sin(angle)*complete[:,2], complete[:,1], np.sin(angle)*complete[:,0] + np.cos(angle)*complete[:,2], complete[:,3], complete[:,4], complete[:,5]], axis=-1)
        partial = partial[:,:3]
        complete = resample_pcd(complete, 16384)
        start = time.time()
        completion1, completion2, mesh_out = sess.run([model.outputs1, model.outputs2, model.mesh], feed_dict={inputs: [partial], npts: [partial.shape[0]]})
        completion1[0][:, (3+args.num_channel):] *= 0
        completion2[0][:, (3+args.num_channel):] *= 0
        mesh_out[0][:, (3+args.num_channel):] *= 0
        total_time += time.time() - start
        cd, emd = sess.run([cd_op, emd_op], feed_dict={output: completion2, gt: [complete]})
        total_cd += cd
        total_emd += emd
        if not cd_per_cat.get(synset_id):
            cd_per_cat[synset_id] = []
        if not emd_per_cat.get(synset_id):
            emd_per_cat[synset_id] = []
        cd_per_cat[synset_id].append(cd)
        emd_per_cat[synset_id].append(emd)
        writer.writerow([model_id, cd, emd])


        if i % args.plot_freq == 0:
            os.makedirs(os.path.join(args.results_dir, 'plots', synset_id), exist_ok=True)
            plot_path = os.path.join(args.results_dir, 'plots', synset_id, '%s.png' % model_id)
            plot_pcd_three_views(plot_path, [partial, completion1[0], completion2[0], mesh_out[0], complete],
                                 ['input', 'coarse', 'fine', 'mesh', 'ground truth'],
                                 'CD %.4f  EMD %.4f' % (cd, emd),
                                 [5, 0.5, 0.5, 0.5, 0.5])
        if args.save_pcd:
            os.makedirs(os.path.join(args.results_dir, 'input', synset_id), exist_ok=True)
            pts_coord = partial[:,0:3]
            pts_color = matplotlib.cm.cool((partial[:,1]))[:,0:3]
            # save_pcd(os.path.join(args.results_dir, 'input', synset_id, '%s.ply' % model_id), np.concatenate((pts_coord, pts_color), -1))
            pcd = PointCloud()
            pcd.points = Vector3dVector(pts_coord)
            pcd.colors = Vector3dVector(pts_color)
            write_point_cloud(os.path.join(args.results_dir, 'input', synset_id, '%s.ply' % model_id), pcd, write_ascii=True)
            os.makedirs(os.path.join(args.results_dir, 'output1', synset_id), exist_ok=True)
            pts_coord = mesh_out[0][:,0:3]
            pts_color = matplotlib.cm.Paired((np.argmax(mesh_out[0][:, 3:], -1) + 1)/11 - 0.5/11)[:,0:3]
            # save_pcd(os.path.join(args.results_dir, 'output1', synset_id, '%s.ply' % model_id), np.concatenate((pts_coord, pts_color), -1))
            pcd.points = Vector3dVector(pts_coord)
            pcd.colors = Vector3dVector(pts_color)
            write_point_cloud(os.path.join(args.results_dir, 'output1', synset_id, '%s.ply' % model_id), pcd, write_ascii=True)
            os.makedirs(os.path.join(args.results_dir, 'output2', synset_id), exist_ok=True)
            pts_coord = completion2[0][:,0:3]
            pts_color = matplotlib.cm.Paired((np.argmax(completion2[0][:, 3:], -1) + 1)/11 - 0.5/11)[:,0:3]
            # save_pcd(os.path.join(args.results_dir, 'output2', synset_id, '%s.ply' % model_id), np.concatenate((pts_coord, pts_color), -1))
            pcd.points = Vector3dVector(pts_coord)
            pcd.colors = Vector3dVector(pts_color)
            write_point_cloud(os.path.join(args.results_dir, 'output2', synset_id, '%s.ply' % model_id), pcd, write_ascii=True)
            #######
            os.makedirs(os.path.join(args.results_dir, 'regions', synset_id), exist_ok=True)
            val_min = np.min(completion2[0][:, 3:])
            val_max = np.max(completion2[0][:, 3:])
            for idx in range (3, 14):
                pts_color = matplotlib.cm.Oranges((completion2[0][:, idx]) / (val_max))[:,0:3]
                pcd.colors = Vector3dVector(pts_color)
                write_point_cloud(os.path.join(args.results_dir, 'regions', synset_id, '%s_%s.ply' % (model_id, idx)), pcd, write_ascii=True)
            os.makedirs(os.path.join(args.results_dir, 'gt', synset_id), exist_ok=True)
            pts_coord = complete[:,0:3]
            if args.experiment == 'shapenet':
                pts_color = matplotlib.cm.cool(complete[:,1])[:,0:3]
            elif args.experiment == 'suncg':
                pts_color = matplotlib.cm.Paired(complete[:,3] - 0.5/11)[:,0:3]
            # save_pcd(os.path.join(args.results_dir, 'gt', synset_id, '%s.ply' % model_id), np.concatenate((pts_coord, pts_color), -1))
            pcd.points = Vector3dVector(pts_coord)
            pcd.colors = Vector3dVector(pts_color)
            write_point_cloud(os.path.join(args.results_dir, 'gt', synset_id, '%s.ply' % model_id), pcd, write_ascii=True)
    sess.close()

    print('Average time: %f' % (total_time / len(model_list)))
    print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
    print('Average Earth mover distance: %f' % (total_emd / len(model_list)))
    writer.writerow([total_time / len(model_list), total_cd / len(model_list), total_emd / len(model_list)])
    print('Chamfer distance per category')
    for synset_id in cd_per_cat.keys():
        print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))
        writer.writerow([synset_id, np.mean(cd_per_cat[synset_id])])
    print('Earth mover distance per category')
    for synset_id in emd_per_cat.keys():
        print(synset_id, '%f' % np.mean(emd_per_cat[synset_id]))
        writer.writerow([synset_id, np.mean(emd_per_cat[synset_id])])
    csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default='data/shapenet/test.list')
    parser.add_argument('--data_dir', default='data/shapenet/test')
    parser.add_argument('--experiment', default='suncg')
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_emd')
    parser.add_argument('--results_dir', default='results/shapenet_pcn_emd')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    parser.add_argument('--num_channel', type=int, default=11)
    args = parser.parse_args()

    test(args)
