# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, inputs, npts, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.channels = 11
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs, npts)
        self.coarse, self.fine, self.entropy = self.create_decoder(self.features, inputs, npts)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha, self.entropy)
        self.outputs1 = self.coarse
        self.outputs2 = self.fine
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs[:,:,0:3], [128, 256])
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
        return features

    def create_decoder(self, features, inputs, npts):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * (3+self.channels)])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3+self.channels])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size), tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3+self.channels])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3+self.channels])

            # fine = mlp_conv(feat, [512, 512, 3+self.channels]) + center
            fine = mlp_conv(feat, [512, 512, 3+self.channels])
            """
            fine *= [1,1,1,0,0,0,0,0,0,0,0,0,0,0]
            fine += center
            fine -= (center * [1,1,1,0,0,0,0,0,0,0,0,0,0,0])
            """

        p_coar_feat = tf.nn.softmax(tf.round(coarse[:,:,3:3+self.channels]), -1)
        p_fine_feat = tf.nn.softmax(tf.round(fine[:,:,3:3+self.channels]), -1)
        p_coar_samp = tf.reduce_mean(p_coar_feat, [1])
        p_fine_samp = tf.reduce_mean(p_fine_feat, [1])
        entropy = -tf.reduce_mean(tf.reduce_sum(p_coar_feat * tf.log(p_coar_feat), [2]), [0, 1])
        entropy -= tf.reduce_mean(tf.reduce_sum(p_fine_feat * tf.log(p_fine_feat), [2]), [0, 1])
        entropy += (2.7 + tf.reduce_mean(tf.reduce_sum(p_coar_samp * tf.log(p_coar_samp), [1]), [0]))
        entropy += (2.7 + tf.reduce_mean(tf.reduce_sum(p_fine_samp * tf.log(p_fine_samp), [1]), [0]))
        return coarse, fine, entropy

    def create_loss(self, coarse, fine, gt, alpha, entropy):
        loss_coarse = chamfer(coarse[:,:,0:3], gt[:,:,0:3])
        """
        _, retb, _, retd = tf_nndistance.nn_distance(coarse[:,:,0:3], gt[:,:,0:3])
        for i in range(np.shape(gt)[0]):
            index = tf.expand_dims(retb[i], -1)
            sem_feat = tf.nn.softmax(coarse[i,:,3:], -1)
            sem_gt = tf.cast(tf.one_hot(tf.gather_nd(tf.cast(gt[i,:,3]*self.channels, tf.int32), index), self.channels), tf.float32)
            loss_sem_coarse = tf.reduce_mean(-tf.reduce_sum(
                        0.97 * sem_gt * tf.log(1e-6 + sem_feat) + (1 - 0.97) *
                        (1 - sem_gt) * tf.log(1e-6 + 1 - sem_feat), [1]))
            loss_coarse += 0.0001 * loss_sem_coarse
        """
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine[:,:,0:3], gt[:,:,0:3])
        """
        _, retb, _, retd = tf_nndistance.nn_distance(fine[:,:,0:3], gt[:,:,0:3])
        for i in range(np.shape(gt)[0]):
            index = tf.expand_dims(retb[i], -1)
            sem_feat = tf.nn.softmax(fine[i,:,3:], -1)
            sem_gt = tf.cast(tf.one_hot(tf.gather_nd(tf.cast(gt[i,:,3]*self.channels, tf.int32), index), self.channels), tf.float32)
            loss_sem_fine = tf.reduce_mean(-tf.reduce_sum(
                        0.97 * sem_gt * tf.log(1e-6 + sem_feat) + (1 - 0.97) *
                        (1 - sem_gt) * tf.log(1e-6 + 1 - sem_feat), [1]))
            loss_fine += 0.0001 * loss_sem_fine
        """
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = loss_coarse + alpha * loss_fine 
        loss += 0.1*entropy
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]
