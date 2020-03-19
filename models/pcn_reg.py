# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, inputs, npts, gt, alpha, num_channel):
        self.num_coarse = 256
        self.grid_size = 8
        """
        self.num_coarse = 4096
        self.grid_size = 2
        """
        self.grid_scale = 0.05
        self.channels = num_channel
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.inputs_can, self.gt_can = self.canon_pose(gt, inputs, npts)
        self.features = self.create_encoder(inputs, npts)
        self.coarse, self.fine, self.mesh = self.create_decoder(self.features)
        self.canonical, _, _ = self.create_decoder(self.create_encoder(self.inputs_can, npts))
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs1 = self.coarse
        self.outputs2 = self.fine
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.gt_can, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'meshes', 'fine output', 'ground truth']

    def create_encoder(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs[:,:,0:3], [128, 256])
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
        return features

    def canon_pose(self, gt, inputs, npts):
        tmp_s, tmp_u, tmp_v = tf.linalg.svd(gt[:,:,:3])
        rot = tf.matmul(tf.linalg.diag(tmp_s), tmp_v, adjoint_b=True)
        rot = tmp_v
        gt_can = tf.concat([tf.matmul(gt[:,:,:3], rot), gt[:,:,3:]], axis=-1)
        inputs_can = tf.concat([tf.matmul(f, rot[idx,:,:]) for idx, f in enumerate(tf.split(inputs, npts, axis=1))], axis=1)
        return inputs_can, gt_can

    """
    def create_encoder_sp(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            inputs = [f for f in tf.split(inputs, npts, axis=1)]
            inputs = tf.concat(inputs, axis=0)
            inputs_ord = point_softpool(inputs, npts_output=self.num_coarse, orders=2)
            features = mlp_conv(inputs_ord, [128, 256, 512, 32])
        return features
    """

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse*self.channels])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, self.channels])
            coarse_ord = point_softpool(coarse, npts_output=self.num_coarse, orders=self.channels)
            coarse = mlp_conv_act(coarse_ord, [512, 512, 3], act_dim=self.channels)
            # learned_label = tf.nn.softmax(coarse[:,:,3:], axis=-1)
            coarse = tf.concat([coarse[:,:,:3], coarse_ord[:,:,:]], axis=-1)

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size), tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3+self.channels])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, global_feat, point_feat], axis=2)

            regions = tf.tile(tf.expand_dims(coarse_ord, 2), [1, 1, self.grid_size ** 2, 1])
            regions = tf.reshape(regions, [-1, self.num_fine, self.channels])

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3+self.channels])

            fine = mlp_conv_act(feat, [512, 512, 3], act_dim=self.channels) # + center
            fine = tf.concat([fine[:,:,:3], regions], axis=-1)
            
            mesh = fine

        return coarse, fine, mesh

    """
    def create_decoder_sp(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            feature_ord = point_softpool(features, npts_output=self.num_coarse, orders=self.channels)
            coarse = mlp_conv_act(feature_ord, [512, 512, 3], act_dim=self.channels)

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size), tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3+self.channels])

            feat = tf.concat([grid_feat, point_feat], axis=2)

            regions = tf.tile(feature_ord, [1, self.grid_size ** 2, 1])
            regions = tf.reshape(regions, [-1, self.num_fine, self.channels])

            fine = mlp_conv_act(feat, [512, 512, 3], act_dim=self.channels) # + center
            fine = tf.concat([fine[:,:,:3], regions], axis=-1)
            
            mesh = fine

        p_coar_feat = tf.nn.softmax(coarse[:,:,3:3+self.channels], -1)
        p_fine_feat = tf.nn.softmax(fine[:,:,3:3+self.channels], -1)
        p_regions_feat = tf.nn.softmax(regions, -1)
        p_coar_samp = tf.reduce_mean(p_coar_feat, [1])
        p_fine_samp = tf.reduce_mean(p_fine_feat, [1])
        # entropy = tf.nn.relu(tf.log(self.channels*1.0) + tf.reduce_mean(tf.reduce_sum(p_coar_samp * tf.log(p_coar_samp), [1]), [0]))
        # entropy += tf.nn.relu(tf.log(self.channels*1.0) + tf.reduce_mean(tf.reduce_sum(p_fine_samp * tf.log(p_fine_samp), [1]), [0]))
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p_regions_feat, logits=p_fine_feat))
        return coarse, fine, mesh, entropy
    """

    def create_loss(self, coarse, fine, gt, alpha):
        p_coar_feat = tf.nn.softmax(coarse[:,:,3:3+self.channels], -1)
        p_fine_feat = tf.nn.softmax(fine[:,:,3:3+self.channels], -1)
        p_can_feat = tf.nn.softmax(self.canonical[:,:,3:3+self.channels], -1)
        p_coar_samp = tf.reduce_mean(p_coar_feat, [1])
        p_fine_samp = tf.reduce_mean(p_fine_feat, [1])
        # entropy = tf.nn.relu(tf.log(self.channels*1.0) + tf.reduce_mean(tf.reduce_sum(p_coar_samp * tf.log(p_coar_samp), [1]), [0]))
        # entropy += tf.nn.relu(tf.log(self.channels*1.0) + tf.reduce_mean(tf.reduce_sum(p_fine_samp * tf.log(p_fine_samp), [1]), [0]))
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p_can_feat, logits=p_coar_feat))
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
            loss_coarse += 0.01 * loss_sem_coarse
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
            loss_fine += 0.01 * loss_sem_fine
        """
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = alpha * loss_coarse + loss_fine # + entropy
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]
