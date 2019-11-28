# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, inputs, npts, gt, alpha, num_channel):
        self.num_coarse = 1
        self.grid_size = 128
        self.grid_scale = 0.05
        self.channels = num_channel
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features, self.features_3d = self.create_encoder(inputs, npts)
        self.fold1, self.fold2, self.mesh, self.entropy = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.fold2, gt, alpha, self.entropy)
        self.outputs1 = self.fold1
        self.outputs2 = self.fold2
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.fold1, self.fold2, gt]
        self.visualize_titles = ['input', 'fold1', 'fold2', 'ground truth']

    def create_encoder(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features_reg = mlp_conv(inputs[:,:,0:3], [128, 256+11])
            tmp_s, tmp_u, tmp_v = tf.linalg.svd(features_reg[:,:,0:256])
            features_3d = tf.matmul(tmp_u, tf.matmul(tf.linalg.diag(tmp_s), tmp_v, adjoint_b=True))
            features = features_reg[:,:,0:256]
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
        return features, tf.concat([features_3d[:,:,0:3], features_reg[:,:,256:]], axis=2)

    def create_decoder(self, features):
        with tf.variable_scope('fold1_0', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])
            point_feat = tf.tile(tf.expand_dims(features, 1), [1, self.grid_size ** 2, 1])
            fold1_reg = mlp_conv(tf.concat([point_feat, grid_feat], axis=2), [512, 512+11])
            tmp_s, tmp_u, tmp_v = tf.linalg.svd(fold1_reg[:,:,0:512])
            fold1_3d = tf.matmul(tmp_u, tf.matmul(tf.linalg.diag(tmp_s), tmp_v, adjoint_b=True))
        with tf.variable_scope('fold1_1', reuse=tf.AUTO_REUSE):
            fold1 = mlp_conv(fold1_reg, [512, 3+11])
        with tf.variable_scope('fold2', reuse=tf.AUTO_REUSE):
            fold2 = mlp_conv(tf.concat([point_feat, fold1], axis=2), [512, 512, 3+11]) 
        mesh = fold2

        p_coar_feat = tf.nn.softmax(tf.round(fold1[:,:,3:3+self.channels]), -1)
        p_fine_feat = tf.nn.softmax(tf.round(fold2[:,:,3:3+self.channels]), -1)
        # p_coar_feat = tf.nn.softmax(fold1[:,:,3:3+self.channels], -1)
        # p_fine_feat = tf.nn.softmax(fold2[:,:,3:3+self.channels], -1)
        p_coar_samp = tf.reduce_mean(p_coar_feat, [1])
        p_fine_samp = tf.reduce_mean(p_fine_feat, [1])
        # entropy = -tf.reduce_mean(tf.reduce_sum(p_coar_feat * tf.log(p_coar_feat), [2]), [0, 1])
        # entropy -= tf.reduce_mean(tf.reduce_sum(p_fine_feat * tf.log(p_fine_feat), [2]), [0, 1])
        entropy = (tf.log(11.0) + tf.reduce_mean(tf.reduce_sum(p_coar_samp * tf.log(p_coar_samp), [1]), [0]))
        entropy += (tf.log(11.0) + tf.reduce_mean(tf.reduce_sum(p_fine_samp * tf.log(p_fine_samp), [1]), [0]))
        return fold1, fold2, mesh, entropy

    def create_loss(self, fold2, gt, alpha, entropy):
        loss = chamfer(fold2[:,:,0:3], gt[:,:,0:3])
        loss += 0.1*entropy
        """
        _, retb, _, retd = tf_nndistance.nn_distance(fold2[:,:,0:3], gt[:,:,0:3])
        for i in range(np.shape(gt)[0]):
            index = tf.expand_dims(retb[i], -1)
            sem_feat = tf.nn.softmax(fold2[i,:,3:], -1)
            sem_gt = tf.cast(tf.one_hot(tf.gather_nd(tf.cast(gt[i,:,3]*self.channels, tf.int32), index), self.channels), tf.float32)
            loss_sem = tf.reduce_mean(-tf.reduce_sum(
                        0.97 * sem_gt * tf.log(1e-6 + sem_feat) + (1 - 0.97) *
                        (1 - sem_gt) * tf.log(1e-6 + 1 - sem_feat), [1]))
            loss += 0.01 * loss_sem
        """

        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, update_loss
