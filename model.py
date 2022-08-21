import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tfrecords as jpeg
import numpy as np
import cv2,os,shutil,random

def conv_layer(x_image, W_size, weight_name, b_size, bias_name, stride, padding):
    W_conv1 = tf.Variable(tf.random_normal(W_size, stddev=0.1), name=weight_name)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W_conv1)
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, stride, stride, 1], padding=padding) #54*54
    b_conv1 = tf.Variable(tf.random_normal(b_size, stddev=0.1), name=bias_name)
    h_conv1 = conv1 + b_conv1
    return h_conv1
    
def depthwise_conv_layer(x_image, W_size, weight_name, stride, padding):
    W_conv1 = tf.Variable(tf.random_normal(W_size, stddev=0.1), name=weight_name)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W_conv1)
    conv1 = tf.nn.depthwise_conv2d(x_image, W_conv1, strides=[1, stride, stride, 1], padding=padding) #54*54
    return conv1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

mode = 'predict'

bs = 64
num_training_samples = 2960
epoch = round(num_training_samples / bs)
num_batch = epoch * 100


if mode == 'test':
    X_Rays, label, filename_ = jpeg.read_and_decode('test.tfrecords')
    x, y_, filename = tf.train.batch([X_Rays, label, filename_],batch_size=1, capacity=16, num_threads=4)

if mode == 'train':
    X_Rays, label, filename_ = jpeg.read_and_decode('train.tfrecords')
    x, y_, filename = tf.train.shuffle_batch([X_Rays, label, filename_],batch_size=bs, capacity=4096, num_threads=16, min_after_dequeue=512)

if mode == 'validation':
    X_Rays, label, filename_ = jpeg.read_and_decode('validation.tfrecords')
    x, y_, filename = tf.train.batch([X_Rays, label, filename_],batch_size=1, capacity=16, num_threads=4)

y = tf.one_hot(y_, 2)

image = tf.reshape(x, [-1, 400, 400, 1])
h_conv1 = conv_layer(image, [3, 3, 1, 8], 'W_conv1', [8], 'b_conv1', 1, 'SAME')
r1 = tf.nn.relu(h_conv1)

#path 1x1
h_conv1x1_1 = conv_layer(r1, [1, 1, 8, 8], 'W_conv1x1_1', [8], 'b_conv1x1_1', 1, 'SAME')
h_conv1x1_1r = tf.nn.relu(h_conv1x1_1)
h_conv1x1_1p = tf.nn.max_pool(h_conv1x1_1r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
h_conv1x1_2 = conv_layer(h_conv1x1_1p, [1, 1, 8, 8], 'W_conv1x1_2', [8], 'b_conv1x1_2', 1, 'SAME')
h_conv1x1_2r = tf.nn.relu(h_conv1x1_2)
h_conv1x1_2p = tf.nn.max_pool(h_conv1x1_2r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
h_conv1x1_3 = conv_layer(h_conv1x1_2p, [1, 1, 8, 8], 'W_conv1x1_3', [8], 'b_conv1x1_3', 1, 'SAME')
h_conv1x1_3r = tf.nn.relu(h_conv1x1_3)
h_conv1x1_3p = tf.nn.max_pool(h_conv1x1_3r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

#path 3x3
h_conv3x3_1 = depthwise_conv_layer(r1, [3, 3, 8, 1], 'W_conv3x3_1', 1, 'SAME')
h_conv3x3_1r = tf.nn.relu(h_conv3x3_1)
h_conv3x3_1p = tf.nn.max_pool(h_conv3x3_1r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv3x3_2 = depthwise_conv_layer(h_conv3x3_1p, [3, 3, 8, 1], 'W_conv3x3_2', 1, 'SAME')
h_conv3x3_2r = tf.nn.relu(h_conv3x3_2)
h_conv3x3_2p = tf.nn.max_pool(h_conv3x3_2r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv3x3_3 = depthwise_conv_layer(h_conv3x3_2p, [3, 3, 8, 1], 'W_conv3x3_3', 1, 'SAME')
h_conv3x3_3r = tf.nn.relu(h_conv3x3_3)
h_conv3x3_3p = tf.nn.max_pool(h_conv3x3_3r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

concat = tf.concat([h_conv1x1_3p, h_conv3x3_3p], axis=3)
h_conv2 = conv_layer(concat, [3, 3, 16, 16], 'W_conv2', [16], 'b_conv2', 1, 'SAME')
r2 = tf.nn.relu(h_conv2)
pool2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv3 = conv_layer(pool2, [3, 3, 16, 16], 'W_conv3', [16], 'b_conv3', 1, 'SAME')
r3 = tf.nn.relu(h_conv3)
pool3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv4 = conv_layer(pool3, [3, 3, 16, 8], 'W_conv4', [8], 'b_conv4', 2, 'SAME')
# GAP = tf.nn.avg_pool(h_conv4, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='VALID')

flat = tf.reshape(h_conv4, [-1, 6*6*8])
W_fc = tf.Variable(tf.random_normal([6*6*8, 2], stddev=0.1, dtype=tf.float32), 'W_fc')
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W_fc)
b_fc = tf.Variable(tf.random_normal([2], stddev=0.1, dtype=tf.float32), 'b_fc')
fc_add = tf.matmul(flat, W_fc) + b_fc

y_conv = tf.nn.softmax(fc_add)
pred = tf.argmax(y_conv,1)

regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_add, labels=y)) + reg_term

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy, var_list=[var for var in tf.trainable_variables()]) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver(max_to_keep=5000, var_list=[var for var in tf.trainable_variables()])

with tf.Session() as sess:
    if mode == 'train':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        cnt = 1
        l = []
        for i in range(num_batch):
            _, loss = sess.run([train_step, cross_entropy])
            l.append(np.mean(loss))
            if (i+1) % epoch == 0:
                saver.save(sess, 'checkpoints/%d.ckpt'%(cnt))
                print('Save ckpt %d, Loss %g'%(cnt, np.mean(l)))
                cnt += 1
                l = []
    elif model == 'validation':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        for cnt in range(1,1+100):
            TP = 0.
            FN = 0.
            TN = 0.
            FP = 0.
            saver.restore(sess, 'checkpoints/%d.ckpt'%(cnt))
            for i in range(200):
                img_name, predict, gt = sess.run([filename ,pred, y_])
                if predict[0] == gt[0]:
                    if gt[0] == 0:
                        TN += 1                        
                    else:
                        TP += 1                        
                else:
                    if gt[0] == 0:
                        FP += 1                        
                    else:
                        FN += 1
            SPEC = TN/(TN+FP+1e-5)
            SEN = TP/(TP+FN+1e-5)
            print('\nEpoch = %d'%cnt)
            print('TP = %g FN = %g'%(TP,FN))
            print('TN = %g FP = %g'%(TN,FP))
            print('SEN = %f'%(SEN))
            print('SPEC = %f'%(SPEC))
    elif mode == 'test':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        
        cnt = 71
        saver.restore(sess, 'checkpoints/%d.ckpt'%(cnt))
        
        TP = 0.
        FN = 0.
        TN = 0.
        FP = 0.        
        for i in range(200):
            img_name, predict, gt = sess.run([filename ,pred, y_])
            if predict[0] == gt[0]:
                if gt[0] == 0:
                    TN += 1                        
                else:
                    TP += 1                        
            else:
                if gt[0] == 0:
                    FP += 1                        
                else:
                    FN += 1
        SPEC = TN/(TN+FP+1e-5)
        SEN = TP/(TP+FN+1e-5)
        print('TP = %g FN = %g'%(TP,FN))
        print('TN = %g FP = %g'%(TN,FP))
        print('SEN = %f'%(SEN))
        print('SPEC = %f'%(SPEC))