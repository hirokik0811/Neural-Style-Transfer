'''
Created on Dec 21, 2017

@author: kwibu
'''
import numpy as np
import tensorflow as tf

class VGG:
    def __init__(self, input_w = 224, input_h = 224, out_size = 10,
                conv_filter_size = 3,
                conv_channels = [[64, 64], [128, 128], [256, 256, 256, 256],
                                [512, 512, 512, 512], [512, 512, 512, 512]],
                fc_neurons = [4096, 4096, 1000],
                pooling = 'avg'):
        """VGG Network(configuration E)
            19 weight layers: 2, 4, 4, 4 conv layers each followed by a pooling layer 
            and 3 fc layers.
            if pooling == 'avg', use average pooling (recommended for this NST experiment).
            if pooling == 'max', use max pooling. """
        self.input_w = input_w
        self.input_h = input_h
        self.out_size = out_size
        self.conv_filter_size = conv_filter_size
        self.conv_channels = conv_channels
        self.fc_neurons = fc_neurons
        self.pooling = pooling
        self.reuse = False
    
    def network(self, X, keep_prob):
        """
        X: input batch_size*w*h*3 tensor
        training: true if it is for training
        out_layer: for NST, the layer number used for output
        """
        prev_layer = 'input'
        model = {'input': X}
        def weight_variable(shape, name=""):
            return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.02))
        def bias_variable(shape, name=""):
            return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer(tf.float32))
        with tf.variable_scope("VGG-Net", reuse = self.reuse):
            with tf.variable_scope("ConvLayers"):
                input = model['input']
                for i in range(len(self.conv_channels)):
                    with tf.variable_scope("ConvLayer%d"%i):
                        for j in range(len(self.conv_channels[i])):
                            input = model[prev_layer]
                            w_conv = weight_variable([self.conv_filter_size,
                                        self.conv_filter_size, input.shape[3], self.conv_channels[i][j]],
                                        name="weight_conv%d-%d"%(i, j))
                            b_conv = bias_variable([self.conv_channels[i][j]],
                                        name="bias_conv%d-%d"%(i, j))
                            model['conv%d-%d'%(i, j)] = \
                                        tf.nn.conv2d(input, filter = w_conv,
                                                     strides = [1, 1, 1, 1],
                                                     padding = 'SAME', name = "conv3-%d"%self.conv_channels[i][j])+b_conv
                            model['relu%d-%d'%(i, j)] = tf.nn.relu(model['conv%d-%d'%(i, j)], name='relu%d-%d'%(i, j))
                            prev_layer = 'relu%d-%d'%(i, j)
                        if self.pooling == 'avg':
                            model['avg%d'%i] = tf.nn.avg_pool(model['relu%d-%d'%(i, j)], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                                 padding='SAME', name="average_pooling%d"%i)
                            prev_layer = 'avg%d'%i
                        elif self.pooling == 'max':
                            model['max%d'%i] = tf.nn.max_pool(model['relu%d-%d'%(i, j)], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                                 padding='SAME', name="max_pooling%d"%i)
                            prev_layer = 'max%d'%i
            with tf.variable_scope("FCLayers"):
                input = model[prev_layer]
                model['reshaped'] = tf.reshape(input, [-1, input.shape[1]*input.shape[2]*input.shape[3]])
                prev_layer = 'reshaped'
                for i in range(len(self.fc_neurons)):
                    with tf.variable_scope("FCLayer%d"%i):
                        input = model[prev_layer]
                        w_fc = weight_variable([input.shape[1], self.fc_neurons[i]], name="weight_fclayer%d"%i)
                        b_fc = bias_variable([self.fc_neurons[i]], name="bias_fclayer%d"%i)
                        model['relu_fc%d'%i] = tf.nn.relu(tf.matmul(input, w_fc)+b_fc)
                        model['dropout%d'%i] = tf.nn.dropout(model['relu_fc%d'%i], keep_prob=keep_prob, name = "dropout%d"%i)
                        prev_layer = 'dropout%d'%i
                model['output'] = tf.nn.softmax(model[prev_layer], name = "softmax")
        self.reuse = True
        return model
        
    def train(self, x, y, learning_rate = 0.001, momentum_alpha = 0.9):
        hout = self.network(x, keep_prob=0.5)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=hout))
        opt = tf.train.MomentumOptimizer(learning_rate, momentum_alpha).minimize(loss)
        return opt
    
    def accuracy(self, x, y):
        hout = self.network(x, keep_prob=1.0)
        correct_predictions = tf.equal(tf.argmax(hout, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy
