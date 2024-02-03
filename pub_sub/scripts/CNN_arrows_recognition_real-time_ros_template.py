# -*- coding: utf-8 -*-



#Load ROS packages ]import rospy
import rospy
from geometry_msgs.msg import Twist
from demo_programs.msg import prox_sensor
from geometry_msgs.msg import Pose
from demo_programs.msg import line_sensor
import math



# Load required packages
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import numpy.ma as ma
from PIL import Image                                                            
import glob
import os
from tensorflow.keras import models
import random
import time


angle = 0

# packages to connect to ROS 
rospy.init_node("control")
# Define publisher for robot movement
pub=rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
# Use twist message for robot movement 
msg = Twist()


rospy.Subscriber("/cop/pose", Pose, callback_func)



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

# Load testing databse 
imageFolderPath = r'/u/r/nep27/Documents/Arrows/Database_arrows'
imageFolderTestingPath = imageFolderPath + r'/validation'
imageTestingPath = []


for i in range(len(namesList)):
    testingLoad = imageFolderTestingPath + '/' + namesList[i] + '/*.jpg'
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)

    
 #Resize images   
updateImageSize = [128, 128]
tempImg = Image.open(imageTestingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size


x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')
    
# create space to load testing labels
y_test = np.zeros((len(x_test),));

countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
y_test = tf.keras.utils.to_categorical(y_test, len(namesList));



# Load trained CNN
model = load_model(r'/rosdata/ros_ws_loc/src/pub_sub/scripts/modelV1.h5')



while(True):
    # Show testing image 
    plt.ion() # Open images as non-blocking 
    random_image = random.randint(0, x_test.shape[0]+1)
    test_image= x_test[random_image, :, :]
    plt.imshow(test_image.squeeze())
    plt.show()
    plt.pause(0.001)

    test_label=y_test[random_image]
    test_image = test_image.reshape((1, 96,128,1))

    output= model.predict_on_batch(test_image)
   
    print('Actual digits', test_label, '---Predicted digits', np.argmax(output, axis=1))
    pub=rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
    
    #End program in user inputs 'yes'
    if (input('stop?') == 'yes'):
        break
     #logic for controlling robot using CNN output classification    
    if np.argmax(output, axis=1)[0] == 0:
        rospy.loginfo('up %s')
        msg.linear.x = 0.15
        msg.angular.z = 0
        pub.publish(msg)
        time.sleep(5)
        
        
    if np.argmax(output, axis=1)[0] == 1:
        rospy.loginfo('down %s ')
        msg.linear.x = -0.15
        msg.angular.z = 0
        pub.publish(msg)
        time.sleep(5)
  
        
    if np.argmax(output, axis=1)[0] == 2:
        rospy.loginfo('left %s ')    
        msg.angular.z = -0.3
        msg.linear.x = 0
        pub.publish(msg)
        time.sleep(7)
        msg.angular.z = 0
        msg.linear.x = 0.15
        pub.publish(msg)

        
    if np.argmax(output, axis=1)[0] == 3:
     
        rospy.loginfo('right %s ')
        msg.angular.z = 0.3
        msg.linear.x = 0
        pub.publish(msg)
        time.sleep(6.5)
        msg.angular.z = 0
        msg.linear.x = 0.15
        pub.publish(msg)
    
    
    # write here the required code to do the following:
    # select an arrow image randmonly from the test dataset
    # select the corresponding class label
    # prepare the image with the correct shape for the CNN
    # use the input image for prediction with the pre-trained model
    # use the predicted output to control a mobile robot in CoppeliaSim via ROS
    # show int the terminal (or plots) the actual and predicted arrow
    # repeate the process until stopped by the user
 

#while not rospy.is_shutdown():
    #pub.publish(msg)
    
    
print('well done!!')