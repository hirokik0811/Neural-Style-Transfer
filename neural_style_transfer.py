'''
Created on Dec 22, 2017

@author: kwibu
'''
import scipy.io
import tensorflow as tf
import numpy as np
import cv2
from VGG import VGG
from pretrained_vgg import PretrainedVGG
from dataset_constructor import dataset
EPOCHS_FOR_VGG = 10000
EPOCHS_FOR_NST = 10000
INPUT_W = 400
INPUT_H = 300
BATCH_SIZE = 256
OUTPUT_SIZE = 10
CONV_FILTER_SIZE = 3


class NeuralStyleTransfer:
    def __init__(self, input_w = 224, input_h = 224):
        self.input_w = input_w
        self.input_h = input_h
        
    def train(self, model, y_style, y_content, weights = [1, 1, 1, 1, 1], 
              learning_rate = 0.001, alpha=10, beta=1):
        """
        sess: tensorflow session running
        model: the pretrained neural network that is to be used for neural style transfer
        y_style: the style image
        y_content: the content image
        learning_rate: learning rate for Adam
        alpha, beta: |log10(alpha/beta)| = 3 or 4 according to the paper
        """
        x = tf.get_variable(name="input_x", shape=[1, self.input_w, self.input_h, 3],
                            dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=128, stddev=96))
        self.x = x
        x_model = model.network(x)
        y_style_model = model.network(y_style)
        y_content_model = model.network(y_content)
        x_outputs = [x_model['conv1_2'], x_model['conv2_2'], x_model['conv3_4'], x_model['conv4_4'], x_model['conv5_4']]
        y_style_outputs = [y_style_model['conv1_2'], y_style_model['conv2_2'], 
                           y_style_model['conv3_4'], y_style_model['conv4_4'], y_style_model['conv5_4']]
        y_content_outputs = [y_content_model['conv1_2'], y_content_model['conv2_2'],
                             y_content_model['conv3_4'], y_content_model['conv4_4'], y_content_model['conv5_4']]
        style_loss = 0
        content_loss = 0
        def activation(x, n, m):
            return tf.reshape(x, [-1, m, n])
        def gram_matrix(x, n, m):
            fx = activation(x, n, m)
            return tf.matmul(tf.transpose(fx, [0, 2, 1]), fx)
        for x_out, y_s_out, y_c_out, w in zip(x_outputs, y_style_outputs, y_content_outputs, weights):
            n = int(x_out.shape[3])
            m = int(x_out.shape[1]*x_out.shape[2])
            gx = gram_matrix(x_out, n, m)
            gy_style = gram_matrix(y_s_out, n, m)
            style_loss += (w/(4*n**2*m**2))*tf.reduce_sum(tf.pow(gx-gy_style, 2))
            fx = activation(x_out, n, m)
            fy_content = activation(y_c_out, n, m)
            content_loss += (w/2)*tf.reduce_sum(tf.pow(fx-fy_content, 2))
        loss = alpha*style_loss + beta*content_loss
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[x])
        return train_op

"""
x = tf.placeholder(tf.float32, [None, INPUT_W, INPUT_H, 3])
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
keep_prob = tf.placeholder(tf.float32)
vgg = VGG(input_w=INPUT_W, input_h=INPUT_H, conv_filter_size=CONV_FILTER_SIZE)
train_vgg_op = vgg.train(x, y, learning_rate=0.001, momentum_alpha=0.9)
accuracy_op = vgg.accuracy(x, y)
data = dataset([np.load('D:\DATASETS\Characters\ICDAR2003\data')])
"""
vgg_matdata = scipy.io.loadmat("D:\MODELS\VGG\imagenet-vgg-verydeep-19.mat")
vgg = PretrainedVGG(vgg_matdata)

y_style = tf.placeholder(tf.float32, [1, INPUT_W, INPUT_H, 3])
y_content = tf.placeholder(tf.float32, [1, INPUT_W, INPUT_H, 3])

nst = NeuralStyleTransfer(INPUT_W, INPUT_H)
train_nst_op = nst.train(vgg, y_style, y_content, weights=[0.5, 1, 1.5, 4, 5], learning_rate=1.0, alpha=10000, beta=1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)


saver = tf.train.Saver()

"""
for i in range(EPOCHS_FOR_VGG):
    batch = data.next_batch(BATCH_SIZE)
    sess.run(train_vgg_op, feed_dict={x:batch[0], y:data[1], keep_prob:0.5})
    if i%100 == 0:
        print(sess.run(accuracy_op, feed_dict={x:data[0], y:data[1], keep_prob:1.0}))
        saver.save(sess, "D:\CHECK_POINTS\VGG_NET(19Layers)\vgg_19.ckpt")
"""
style_im = np.expand_dims(cv2.resize(cv2.imread('./eyes2.jpg'), (INPUT_H, INPUT_W)), axis = 0)
content_im = np.expand_dims(cv2.resize(cv2.imread('./myface.jpg'), (INPUT_H, INPUT_W)), axis = 0)

for i in range(EPOCHS_FOR_NST):
    sess.run(train_nst_op, feed_dict={y_style:style_im, y_content:content_im})
    if i%100 == 0:
        generated_im = sess.run(nst.x)
        #cv2.imshow("", generated_im[0])
        #cv2.waitKey(100)
        cv2.imwrite(r"D:\EXPERIMENT_RESULTS\NeuralStyleTransfer\ckpt%d.jpg"%i, generated_im[0])