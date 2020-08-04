import tensorflow as tf
import numpy as np
import math
import scipy.io as sio
import random
import scipy.misc
import os
import tensorflow.contrib.layers as ly
from numpy import *
import numpy.matlib
import scipy.ndimage
import csv


class Network(object):
    def __init__(self, tf_x, tf_y, num_categories, lr=0.001, hidden_layers=None, num=10, is_adam=True, lambda_r=0.5, lambda_s=0.001, lambda_c=0.5, dim=1, reuse=False):
        """
        build the RDOSR network
        :param tf_x: tf.placeholder with shape (batch_size, height, 1, 1), input data
        :param tf_y: tf.placeholder with shape (batch_size, num_categories), one-hot label
        :param num_categories: int, number of categories
        :param lr: float, learning rate
        :param hidden_layers: a list or tuple of int, the channel number of each hidden layer,
        :param reuse: bool, whether reuse the graph
        :param is_train: bool, whether in training mode or testing mode
        """
        # initialize the input and weights matrices
        self.tf_x, self.tf_y = tf_x, tf_y
        self.num_categories = num_categories
        self.initlr_F = lr
        self.initlr_EDC = lr
        self.hidden_layers = hidden_layers
        self.num = num
        self.is_adam = is_adam
        self.lambda_r = lambda_r
        self.lambda_s = lambda_s
        self.lambda_c = lambda_c
        self.dim = dim

        with tf.variable_scope('var_decoder') as scope:
            self.wdecoder = {
                'lr_decoder_w1': tf.Variable(tf.truncated_normal([self.num, self.num],stddev=0.1)),
                'lr_decoder_w2': tf.Variable(tf.truncated_normal([self.num, self.num_categories], stddev=0.1)),
            }

        # build the network	
        self.opt_F, self.loss_F, self.opt_EDC, self.loss_EDC, self.loss_euc, self.loss_sparse, self.accuracy_F = self.build_graph(reuse=reuse)

    def compute_latent_vars_break(self, i, remaining_stick, v_samples):
        # compute stick segment
        stick_segment = v_samples[:, i] * remaining_stick
        remaining_stick *= (1 - v_samples[:, i])
        return (stick_segment, remaining_stick)

    def construct_vsamples(self,uniform,wb,hsize):
        concat_wb = wb
        for iter in range(hsize - 1):
            concat_wb = tf.concat([concat_wb, wb], 1)
        v_samples = uniform ** (1.0 / concat_wb)
        return v_samples

    def encoder_uniform(self,x,reuse=False):
        layer_1 = x   #shape: [N,L], L is the number of bands
        with tf.variable_scope('var_uniform') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_11 = tf.contrib.layers.fully_connected(layer_1, self.hidden_layers[0], activation_fn=None)
            stack_layer_11 = tf.concat([x,layer_11], 1)
            layer_12 = tf.contrib.layers.fully_connected(stack_layer_11, self.hidden_layers[1], activation_fn=None)
            stack_layer_12 = tf.concat([stack_layer_11, layer_12], 1)
            layer_13 = tf.contrib.layers.fully_connected(stack_layer_12, self.hidden_layers[2], activation_fn=None)
            stack_layer_13 = tf.concat([stack_layer_12, layer_13], 1)
            layer_14 = tf.contrib.layers.fully_connected(stack_layer_13, self.hidden_layers[3], activation_fn=None)
            stack_layer_14 = tf.concat([stack_layer_13, layer_14], 1)

            uniform = tf.contrib.layers.fully_connected(stack_layer_14, self.num, activation_fn=None)
        return layer_1, uniform

    def encoder_beta(self,x,reuse=False):
        with tf.variable_scope('var_beta') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_21 = tf.contrib.layers.fully_connected(x, self.hidden_layers[0], activation_fn=None)
            stack_layer_21 = tf.concat([x,layer_21], 1)
            layer_22 = tf.contrib.layers.fully_connected(stack_layer_21, self.hidden_layers[1], activation_fn=None)
            stack_layer_22 = tf.concat([layer_22, stack_layer_21], 1)
            layer_32 = tf.contrib.layers.fully_connected(stack_layer_22, 1, activation_fn=None)
        return layer_32

    def encoder_vsamples(self, x, hsize, reuse=False):
        layer1, uniform = self.encoder_uniform(x,reuse)
        uniform = tf.nn.sigmoid(uniform)
        wb = self.encoder_beta(layer1,reuse)
        wb = tf.nn.softplus(wb)
        v_samples = self.construct_vsamples(uniform,wb,hsize)
        return v_samples, uniform, wb

    def construct_stick_break(self,vsample, dim, stick_size):
        size = dim
        size = int(size)
        remaining_stick = tf.ones(size, )
        for i in range(stick_size):
            [stick_segment, remaining_stick] = self.compute_latent_vars_break(i, remaining_stick, vsample)
            if i == 0:
                stick_segment_sum_lr = tf.expand_dims(stick_segment, 1)
            else:
                stick_segment_sum_lr = tf.concat([stick_segment_sum_lr, tf.expand_dims(stick_segment, 1)],1)
        return stick_segment_sum_lr

    def encoder(self, x, dim, reuse=False):
        v_samples,uniform, wb = self.encoder_vsamples(x, self.num, reuse)
        stick_segment_sum_lr = self.construct_stick_break(v_samples, dim, self.num)
        return stick_segment_sum_lr

    def decoder(self, x):
        layer_1 = tf.matmul(x, self.wdecoder['lr_decoder_w1'])
        layer_2 = tf.matmul(layer_1, self.wdecoder['lr_decoder_w2'])
        return layer_2

    def generator(self, x, dim, reuse=False):
        encoder_op = self.encoder(x, dim, reuse)
        decoder_op = self.decoder(encoder_op)
        return decoder_op

    def classifier_C(self, x, dim, reuse=False):
        encoder_x = self.encoder(x, dim, reuse=True)
        with tf.variable_scope('var_cls_c') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_cls = tf.contrib.layers.fully_connected(encoder_x, self.num_categories, activation_fn=None)
        return layer_cls

    def classifier_F(self, x, dim, reuse=False):
        layer_1 = x   #shape: [N,103], 103 is number of bands
        with tf.variable_scope('var_cls_f') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_11 = tf.contrib.layers.fully_connected(layer_1,512)
            layer_12 = tf.contrib.layers.fully_connected(layer_11,1024)
            layer_13 = tf.contrib.layers.fully_connected(layer_12,512)
            layer_14 = tf.contrib.layers.fully_connected(layer_13,32)
            layer_5 = tf.contrib.layers.fully_connected(layer_14, self.num_categories, activation_fn=None)
        return layer_5,layer_14

    def build_graph(self, reuse=False):
	
        #--- Classifier F
        # classification loss
        output_F, input_feature = self.classifier_F(self.tf_x, self.dim, reuse=reuse)
        sparsity = tf.abs(output_F)
        sparse_loss =tf.reduce_mean(tf.reduce_sum(sparsity,-1))

        loss_F = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.tf_y, logits=output_F))\
                      +0.1*sparse_loss

        # optimizaer
        theta_F = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_cls_f')
        counter_F = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_F = ly.optimize_loss(loss=loss_F, learning_rate=self.initlr_F,
                                 optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_F,global_step=counter_F)

        correct_prediction = tf.equal(tf.argmax(output_F, axis=1), tf.argmax(self.tf_y, axis=1))
        accuracy_F = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        #--- Autoencoder + Dirichlet + Classifier C
	# reconstruction
        input_ae_,input_feature = self.classifier_F(self.tf_x, self.dim, reuse=True)
        input_ae = input_ae_/10
        output_ae = self.generator(input_ae, self.dim, reuse=reuse)
        reconstruction_error = output_ae - input_ae
        loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(reconstruction_error, 2),0))/100

        # spatial sparse
        eps = 0.00000001
        s_j = self.encoder(input_ae, self.dim, reuse=True)
        s_base = tf.reduce_sum(s_j, 1, keepdims=True)
        sparse = tf.div(s_j, (s_base + eps))
        loss_sparse = tf.reduce_mean(-tf.multiply(sparse, tf.log(sparse + eps)))

        # classification loss
        output_C = self.classifier_C(input_ae, self.dim, reuse=reuse)
        loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.tf_y, logits=output_C))

        # total loss
        loss_EDC = self.lambda_r*loss_euc + self.lambda_s*loss_sparse + self.lambda_c*loss_C

        # optimizer
        theta_basic_decoder = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_decoder')
        theta_uniform = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_uniform')
        theta_beta = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_beta')
        theta_cls_c = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_cls_c')
        counter_EDC = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_EDC = ly.optimize_loss(loss=loss_EDC, learning_rate=self.initlr_EDC,
                                 optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_basic_decoder+theta_uniform+theta_beta+theta_cls_c, global_step=counter_EDC)

        return opt_F, loss_F, opt_EDC, loss_EDC, loss_euc, loss_sparse, accuracy_F

