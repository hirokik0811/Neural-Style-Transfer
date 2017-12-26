'''
Created on Dec 25, 2017

@author: kwibu
'''
import tensorflow as tf
class PretrainedVGG:
    def __init__(self, matdata):
        self.matdata = matdata
    def network(self, x, pooling = 'avg'):
        prev_layer = 'input'
        model = {'input': x}
        for layer in self.matdata['layers'][0]:
            layer_name = layer['name'][0][0][0]
            if layer['type'] == 'conv':
                if layer_name[0:1] == 'fc':
                    model[layer_name] = \
                    tf.matmul(model[prev_layer], layer['weights'][0][0][0][0])+\
                             tf.reshape(layer['weights'][0][0][0][1], [layer['weights'][0][0][0][1].shape[0]])
                else:
                    model[layer_name] = \
                    tf.nn.conv2d(model[prev_layer], filter=layer['weights'][0][0][0][0],
                                 strides=[1]+list(layer['stride'][0][0][0])+[1], padding='SAME',
                                 name=layer_name)+\
                                 tf.reshape(layer['weights'][0][0][0][1], [layer['weights'][0][0][0][1].shape[0]])
                
            elif layer['type'] == 'relu':
                model[layer_name] = \
                tf.nn.relu(model[prev_layer])
            elif layer['type'] == 'pool':
                if pooling == 'avg':
                    model[layer_name] = \
                    tf.nn.avg_pool(model[prev_layer], ksize=[1]+list(layer['pool'][0][0][0])+[1],
                                   strides=[1]+list(layer['stride'][0][0][0])+[1], padding='SAME',
                                   name=layer_name)
                elif pooling == 'max':
                    model[layer_name] = \
                    tf.nn.max_pool(model[prev_layer], ksize=[1]+list(layer['pool'][0][0][0])+[1],
                                   strides=[1]+list(layer['stride'][0][0][0])+[1], padding='SAME',
                                   name=layer_name)
            elif layer['type'] == 'fc':
                model[layer_name] = \
                tf.matmul(model[prev_layer], layer['weights'][0][0][0][0])+layer['weights'][0][0][0][1]
            elif layer['type'] == 'softmax':
                model[layer_name] = \
                tf.nn.softmax(model[prev_layer])
            prev_layer = layer_name
            
        return model
                