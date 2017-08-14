#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:12:47 2017

@author: geyi0530
"""

#coding:utf-8 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask,request,Response,make_response
import tensorflow as tf
import os
import wget
from PIL import Image,ImageFilter
import numpy as np
app = Flask(__name__)

#########fuwuqi

import tensorflow as tf

##using flask
import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask import Flask
from redis import Redis, RedisError
import socket

#pac for cassandra
from cassandra.cluster import Cluster
import re

#from app import app


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
user_id=0

@app.route("/")
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    
def prepareImage(img_path):
	im = Image.open(img_path).convert('L')
	width = float(im.size[0])
	height = float(im.size[1])
	newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
	if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
		nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
		if (nheight == 0): #rare case but minimum is 1 pixel
			nheight = 1  
        # resize and sharpen
		img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
		newImage.paste(img, (4, wtop)) #paste resized image on white canvas
	else:
        #Height is bigger. Heigth becomes 20 pixels. 
		nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
		if (nwidth == 0): #rare case but minimum is 1 pixel
			nwidth = 1
         # resize and sharpen
		img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
		newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    #newImage.save("sample.png")

	tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
	tva = [ (255-x)*1.0/255.0 for x in tv] 
	return tva

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
   
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    """
    Load the model2.ckpt file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.

    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('my-test-model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)

@app.route('/img',methods=['POST'])
def process():
    f = request.files['file']
    basic_path = '/app/'
    f.save(os.path.join(basic_path,'picresult.jpg'))
    imvalue = prepareImage(basic_path+'picresult.jpg')
    predint = predictint(imvalue)[0]
    result = str(np.asscalar(predint))
    return make_response(result+'\n')
    cassandra(result)

@app.route('/',methods=['POST','GET'])
def init():
    print (request.form['test'])
    return make_response()

    
if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=80)
    process()

def cassandra(res):
    imageH=imgToHex(basic_path,'picresult.jpg')
    user_id=user_id+1
    #res is alreday string value
    data='('+user_id+','+imageH+','+res+')'
    IP='0.0.0.0'
    cluster = Cluster([IP])
    session= cluster.connect()
    table='icnn (id,image,result)'
    session.execute('USE cnn;') 
    session.execute('INSERT INTO '+table+'VALUES'+data+';')
    cluster.shutdown()
    
    
def imgToHex(file):
    string = ''
    with open(file, 'rb') as f:
        binValue = f.read(1)
        while len(binValue) != 0:
            hexVal = hex(ord(binValue))
            string += '\\' + hexVal
            binValue = f.read(1)
    #string = re.sub('0x', 'x', string) # Replace '0x' with 'x' for your needs
    return string