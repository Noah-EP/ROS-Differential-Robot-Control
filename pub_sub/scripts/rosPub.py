#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from demo_programs.msg import prox_sensor
from geometry_msgs.msg import Pose
from demo_programs.msg import line_sensor
import math
rospy.init_node("control")
pub=rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
angle = 0
line = False
finishedMaze = False

def euler_from_quaternion(x, y, z, w):  #Convert from quaternion form to euler 
        """
        Used code from source: https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
     
        """
       #  t0 = +2.0 * (w * x + y * z)
       #  t1 = +1.0 - 2.0 * (x * x + y * y)       
       #  roll_x = math.atan2(t0, t1)
     
       #  t2 = +2.0 * (w * y - z * x)
       #  t2 = +1.0 if t2 > +1.0 else t2
       #  t2 = -1.0 if t2 < -1.0 else t2
       #  pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z



def callback_func2(message): # Get quaternion position from pose topic and convert to angular position
    global angle
    posX = message.orientation.x
    posY = message.orientation.y
    posZ = message.orientation.z
    posW = message.orientation.w
    euler = euler_from_quaternion(posX, posY, posZ, posW)
    angle = euler*180/math.pi
 
    
def callback_func3(message): #Get status of line sensor 
    global line
    line = message.line_middle 
    
def callback_func(message): #Get proximity sensor data and perform logic operations to decide robot maneuver
    global angle
    global line
    global finishedMaze
    global escapeCorner
    
    
    #Prox sensor data
    front = message.prox_front
    frontL = message.prox_front_left
    frontR = message.prox_front_right
    frontRR = message.prox_front_right_right
    frontLL= message.prox_front_left_left 
   
       
   
    if (line == True): #Stop robot if it has reached line 
            finishedMaze = True
            msg.linear.x = 0
            msg.angular.z= 0
    if finishedMaze ==False: 

        if (front >0):
         #Decide whether to turn left or right if front sensors are detecting an object      
                
            if (frontL > frontR and frontR > 0):
                msg.angular.z= -0.3
                msg.linear.x = 0
                
            if (frontR > frontL and frontL> 0):
                msg.angular.z= 0.3
                msg.linear.x = 0

        
        
        
        if (front == 0 and frontL == 0 and frontR == 0 and  frontLL > 0):
         
            msg.linear.x = 0.2
            msg.angular.z= 0.3  
        
        
        if (front == 0 and frontL == 0 and frontR == 0 and frontLL == 0 and frontRR > 0):
            
            msg.linear.x = 0.1
            msg.angular.z= -0.3
              
        if (front == 0 and frontL == 0 and frontR == 0 and frontLL > 0 and frontRR == 0):
            msg.linear.x = 0.1
            msg.angular.z= 0.3
              
        if (front >0 and frontL >0 and frontR >0 and frontLL >0 and frontRR ==0):
            msg.linear.x = 0
            msg.angular.z= 0.3       
    
        if (front >0 and frontL >0 and frontR >0 and frontLL==0 and frontRR > 0):
            msg.linear.x = 0
            msg.angular.z=-0.3    
        
        if (front >0 and frontL >0 and frontR >0 and frontLL>0 and frontRR > 0):
            if (frontLL > frontRR):
                msg.angular.z=0.3    
            if (frontRR> frontLL):
                msg.angular.z=-0.3 

        
        #If no objects detected, use angular position to point robot towards end of maze
        if (front == 0 and frontL == 0 and frontR == 0 and frontLL == 0 and frontRR == 0):
            if (angle < 180 and angle > 0):
                msg.linear.x = 0.2
                msg.angular.z= -0.4 
                
                
            if (angle < 0 and angle > -180):
                msg.linear.x = 0.2
                msg.angular.z= 0.4 
    
#Subscribe to sensor topics  
rospy.Subscriber("/cop/prox_sensors", prox_sensor, callback_func, queue_size= 1)
rospy.Subscriber("/cop/pose", Pose, callback_func2)
rospy.Subscriber("/cop/line_sensors", line_sensor, callback_func3)
msg = Twist()

msg2=prox_sensor()

while not rospy.is_shutdown():
    pub.publish(msg) #Publish velocity messages
    
    rospy.sleep(1)