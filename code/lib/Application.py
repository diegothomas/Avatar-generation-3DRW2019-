#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:24:42 2017

@author: diegothomas
"""

# File to handle program main loop
import cv2
from math import cos, sin, pi
import numpy as np
import Tkinter as tk
import imp
import time
from PIL import Image
from PIL import ImageTk

from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )
RGBD = imp.load_source('RGBD', APP_ROOT + '/RGBD.py')
GPU = imp.load_source('GPUManager', APP_ROOT + '/GPUManager.py')
TrackManager = imp.load_source('Tracker', APP_ROOT + '/Tracker.py')
BSManager = imp.load_source('BSManager', APP_ROOT + '/BSManager.py')
FacialExpression = imp.load_source('FacialExpression', APP_ROOT + '/FacialExpression.py')

VERBOSE = False

'''
    Manager for the whole application
'''
class Application(tk.Frame):
    
    ''' Function to handle keyboard inputs '''
    def key(self, event):
        # 3D transformation that will take the deplacement due to keyboard input
        # Initialise as a 4x4 identity matrix
        Transfo = np.array([[1., 0., 0., 0.], 
                            [0., 1., 0., 0.], 
                            [0., 0., 1., 0.], 
                            [0., 0., 0., 1.]])
        
        if (event.keysym == 'Escape'): # the escape key is hit => exit the application
            self.root.destroy()
        
        '''
            'd' = shift to the right
            'a' = shift to the left
            'w' = move forward
            's' = move backward
            'q' = move up
            'e' = move down
            'c' = switch color mode
            'r' = Run KinectFusion
            'b' = Record result
        '''
        if (event.keysym == 'd'):
            Transfo[0,3] = 0.01
        if (event.keysym == 'a'):
            Transfo[0,3] = -0.01
        if (event.keysym == 'w'):
            Transfo[1,3] = -0.01
        if (event.keysym == 's'):
            Transfo[1,3] = 0.01
        if (event.keysym == 'e'):
            Transfo[2,3] = -0.01
        if (event.keysym == 'q'):
            Transfo[2,3] = 0.01
        if (event.keysym == 'c'):
            self.color_tag = (self.color_tag+1) %3
        if (event.keysym == 'r'):
            self.Run = True
            self.RunApp()
        if (event.keysym == 'b'):
            self.Stop = True
        if (event.keysym == 'g'):
            self.BS.ChangeLandmark(self.landmark, 1)
        if (event.keysym == 'h'):
            self.BS.ChangeLandmark(self.landmark, -1)
            
        # if the application is still running: do some drawing
        if (event.keysym != 'Escape'):
            # Update the current global camera pose matrix
            self.Camera_Pose = np.dot(self.Camera_Pose, Transfo)
            #self.res = self.RGBD.Draw_GPU(self.Camera_Pose,1,self.color_tag)
            #self.res = 0.5*self.res + 0.5*self.BS.DrawBump(self.Camera_Pose, self.intrinsic, self.color_tag)
            self.res = self.BS.DrawBump(self.Camera_Pose, self.intrinsic, self.color_tag)
            self.res = Image.fromarray(self.res.astype(np.uint8), 'RGB')
            self.res = ImageTk.PhotoImage(self.res)
            self.canvas.create_image(320,240,image = self.res)
                
            #self.BS.BlendShapes[0].DrawMesh(self.Pose, self.intrinsic, self.RGBD.Size, self.canvas)
            #if (not event.keysym == 'g' and not event.keysym == 'h'):
            #    self.landmark += 1
            self.BS.DrawLandmarks(self.RGBD, self.Camera_Pose, self.intrinsic, self.canvas, self.landmark)
            
    def SaveResult(self):
        # Save the 3D mesh on the disk
        start_time = time.time()
        self.BS.Save("../../../Results/WrinkleMe")
        elapsed_time = time.time() - start_time
        print "SaveToPly: %f" % (elapsed_time)
        
            
    def RunApp(self):
        start_time = time.time()
        """ Read new image """
        start_time = time.time()
        if ((not self.GetNewData()) or self.Stop):
            self.Stop = True
            self.SaveResult()
            return
        elapsed_time = time.time() - start_time
        print "Get New Data: %f" % (elapsed_time)
        
        """ Detect Face """
        self.RGBD.DetectFace()
        
        """ Track """
        start_time = time.time()
        self.Pose = self.Tracker.RegisterBumpToRGBD_GPU(self.BS, self.RGBD, self.Pose)
        elapsed_time = time.time() - start_time
        print "Register: %f" % (elapsed_time)
        
        """ Track Facial Expression """
        start_time = time.time()
        self.ExpressionTracker.EstimateBSCoeff(self.BS, self.RGBD, self.Pose)
        elapsed_time = time.time() - start_time
        print "EstimateBSCoeff: %f" % (elapsed_time)
        
        """ Update Bump Image """
        start_time = time.time()
        self.BS.BumpImage(self.RGBD, self.Pose)
        elapsed_time = time.time() - start_time
        print "Bump mapping: %f" % (elapsed_time)
        
        #start_time = time.time()
        Transfo = np.array([[1., 0., 0., 0.], 
                            [0., 1., 0., 0.], 
                            [0., 0., 1., 0.], 
                            [0., 0., 0., 1.]], dtype = np.float32)
        #self.res = self.RGBD.Draw_GPU(Transfo,1,self.color_tag).astype(np.uint8)
        #self.res = 0.5*self.res + 0.5*self.BS.DrawBump(self.Pose, self.intrinsic, self.color_tag)
        self.res = self.BS.DrawBump(self.Pose, self.intrinsic, self.color_tag)
        #cv2.imwrite('../../../Results/WrinkleMe/Output/Output_'+str(self.RGBD.index)+'.png',self.res.astype(np.uint8))
        #elapsed_time = time.time() - start_time
        #print "Draw: %f" % (elapsed_time)
        self.res = Image.fromarray(self.res.astype(np.uint8), 'RGB')
        self.res = ImageTk.PhotoImage(self.res)
        self.canvas.create_image(320,240,image =  self.res)
        self.BS.DrawLandmarks(self.RGBD, self.Pose, self.intrinsic, self.canvas, self.landmark)
        
        elapsed_time = time.time() - start_time
        if (VERBOSE):
            print "Application: %f" % (elapsed_time)
                
        self.root.after(1, self.RunApp)
        
    ''' Function to handle mouse press event '''
    def mouse_press(self, event):
        self.x_init = event.x
        self.y_init = event.y
    
    ''' Function to handle mouse motion events '''
    def mouse_motion(self, event):
        if (event.y < self.Size[0]):
            delta_x = event.x - self.x_init
            delta_y = event.y - self.y_init
            
            angley = 0.
            if (delta_x > 0.):
                angley = -0.01
            elif (delta_x < 0.):
                angley = 0.01 #pi * 2. * delta_x / float(self.Size[0])
            RotY = np.array([[cos(angley), 0., sin(angley), 0.], \
                             [0., 1., 0., 0.], \
                             [-sin(angley), 0., cos(angley), 0.], \
                             [0., 0., 0., 1.]])
            self.Camera_Pose = np.dot(self.Camera_Pose, RotY)
            
            anglex = 0.
            if (delta_y > 0.):
                anglex = 0.01
            elif (delta_y < 0.):
                anglex = -0.01 # pi * 2. * delta_y / float(self.Size[0])
            RotX = np.array([[1., 0., 0., 0.], \
                            [0., cos(anglex), -sin(anglex), 0.], \
                            [0., sin(anglex), cos(anglex), 0.], \
                            [0., 0., 0., 1.]])

            self.Camera_Pose = np.dot(self.Camera_Pose, RotX)
            self.res = self.RGBD.Draw_GPU(self.Camera_Pose,1,self.color_tag).astype(np.uint8)
            self.res = Image.fromarray(self.res, 'RGB')
            self.res = ImageTk.PhotoImage(self.res)
            self.canvas.create_image(320,240,image = self.res)
            
            
        self.x_init = event.x
        self.y_init = event.y
        
    def GetNewData(self):
        self.RGBD.depthname = self.path + '/Depth_'+str(self.RGBD.index)+'.tiff'
        self.RGBD.colorname = self.path + '/RGB_'+str(self.RGBD.index)+'.tiff'
        self.RGBD.index += 1
        
        if (not self.RGBD.ReadFromDisk()):
            return False
        
        self.RGBD.BilateralFilter_GPU(2,0.02,3)
        
        self.RGBD.Vmap_GPU(0)
        self.RGBD.ReProj_depth(self.D2RGB)
        #self.RGBD.Vmap_GPU(1)
        
        self.RGBD.NMap_GPU()
        
        self.GPUManager.queue.finish()
        
        return True
            
        

    ''' Constructor function '''
    def __init__(self, path, GPUManager, master=None):
        # link with the global Tkinter app
        self.root = master
        # Initialise path to load data
        self.path = path
        # Initialise GPU manager class
        self.GPUManager = GPUManager
        # Tag do draw or not colors
        self.color_tag = 2
        # Initialise camera pose (transfo from ref to curr frame) to 4x4 identity matrix
        self.Pose = np.array([[1., 0., 0., 0.], 
                              [0., 1., 0., 0.], 
                              [0., 0., 1., 0.], 
                              [0., 0., 0., 1.]], dtype = np.float32)
    
        self.Camera_Pose = np.array([[1., 0., 0., 0.], 
                              [0., 1., 0., 0.], 
                              [0., 0., 1., 0.], 
                              [0., 0., 0., 1.]], dtype = np.float32)
    
        self.Run = False
        self.Stop = False
        self.landmark = 0
 
        # Initialise tk rendering object
        tk.Frame.__init__(self, master)
        self.pack()

        # Read the calibration file to get camera intrinsic parameters
        calib_file = open(self.path + '/Calib.txt', 'r')
        calib_data = calib_file.readlines()
        self.Size = [int(calib_data[0]), int(calib_data[1])]
        self.intrinsicRGB = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                                   [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                                   [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)
    
        self.intrinsic = np.array([[float(calib_data[12]), float(calib_data[13]), float(calib_data[14])], \
                                   [float(calib_data[15]), float(calib_data[16]), float(calib_data[17])], \
                                   [float(calib_data[18]), float(calib_data[19]), float(calib_data[20])]], dtype = np.float32)
    
        self.D2RGB = np.array([[float(calib_data[21]), float(calib_data[22]), float(calib_data[23]), float(calib_data[24])], 
                              [float(calib_data[25]), float(calib_data[26]), float(calib_data[27]), float(calib_data[28])], 
                              [float(calib_data[29]), float(calib_data[30]), float(calib_data[31]), float(calib_data[32])], 
                              [float(calib_data[33]), float(calib_data[34]), float(calib_data[35]), float(calib_data[36])]], dtype = np.float32)
    
    
        self.fact = float(calib_data[11])
        print "Input image size: \n", self.Size
        print "Camera intrinsic matrix: \n", self.intrinsic
        print "Input depth encoding factor: ", self.fact
        
        # Visualisation camera image
        self.canvas = tk.Canvas(self, bg="black", height=self.Size[0], width=self.Size[1])
        self.canvas.pack()    

        # Create the object that will manage the RGBD data
        self.RGBD = RGBD.RGBD(self.GPUManager, self.path + '/Depth_10.tiff', self.path + '/RGB_10.tiff', self.intrinsic, (self.Size[0], self.Size[1], 3), self.fact)
        self.RGBD.index = 10
        self.RGBD.intrinsicRGB = self.intrinsicRGB
        
        # Create the object that will manage the tracking
        self.Tracker = TrackManager.Tracker(self.GPUManager, self.RGBD.Size, self.intrinsic, 0.02, 60.0 / 180.0 * pi, 1, [6,1,0,0], 0.00001)
        
        start_time = time.time()
        self.RGBD.ReadFromDisk()
        elapsed_time = time.time() - start_time
        print "ReadFromDisk: %f" % (elapsed_time)
        
        start_time = time.time()
        self.RGBD.BilateralFilter_GPU(2,0.02,3)
        elapsed_time = time.time() - start_time
        print "BilateralFilter: %f" % (elapsed_time)
        
        start_time = time.time()
        self.RGBD.Vmap_GPU(0)
        self.RGBD.ReProj_depth(self.D2RGB)
        #self.RGBD.Vmap_GPU(1)
        elapsed_time = time.time() - start_time
        print "Vmap: %f" % (elapsed_time)
        
        start_time = time.time()
        self.RGBD.NMap_GPU()
        elapsed_time = time.time() - start_time
        print "NMap: %f" % (elapsed_time)
        
        start_time = time.time()
        self.RGBD.DetectFace()
        elapsed_time = time.time() - start_time
        print "DetectFace: %f" % (elapsed_time)
        
        
        """ Initialise 3D model """ 
        self.BS = BSManager.BSMng(self.GPUManager, '../data/blendshapes/', self.D2RGB)
        self.BS.LoadBS()
        start_time = time.time()
        self.BS.LoadAffineTransfo()
        print "ComputeAffineTransfo: %f" % (elapsed_time)
        
        print len(self.BS.BlendShapes)
        
        # Create the object that will manage the tracking
        self.ExpressionTracker = FacialExpression.FacialExpression(self.GPUManager, self.BS.Size, self.RGBD.Size, self.intrinsic, 0.01, 40.0 / 180.0 * pi, 1, [6,1,0,0], 0.00001)
            
        start_time = time.time()
        self.BS.Rescale(self.RGBD)
        elapsed_time = time.time() - start_time
        print "Rescale: %f" % (elapsed_time)
        
        start_time = time.time()
        self.BS.AlignToFace(self.RGBD)
        elapsed_time = time.time() - start_time
        print "AlignToFace: %f" % (elapsed_time)
        
        start_time = time.time()
        self.BS.ElasticRegistration(self.RGBD)
        elapsed_time = time.time() - start_time
        print "ElasticRegistration: %f" % (elapsed_time)
        
        start_time = time.time()
        self.BS.PreProcessing()
        elapsed_time = time.time() - start_time
        print "PreProcessing: %f" % (elapsed_time)
        
        # Generate first bump image
        start_time = time.time()
        self.BS.BumpImage(self.RGBD, self.Pose)
        self.BS.SetLandmarks(self.RGBD)
        elapsed_time = time.time() - start_time
        print "BumpImage: %f" % (elapsed_time)
        
        self.res = self.BS.DrawBump(self.Camera_Pose, self.intrinsic, 1).astype(np.uint8)
        self.res = Image.fromarray(self.res, 'RGB')
        self.res = ImageTk.PhotoImage(self.res)
        self.canvas.create_image(320,240,image =  self.res)
        #self.BS.BlendShapes[0].DrawMesh(self.Camera_Pose, self.intrinsic, self.RGBD.Size, self.canvas)
        self.BS.DrawLandmarks(self.RGBD, self.Camera_Pose, self.intrinsic, self.canvas, 0)
        
        ################TO DO#######################
        ## Implement the function that draw the input on the visualization image plane
        '''start_time = time.time()
        self.res = self.RGBD.Draw_GPU(self.Pose,1,self.color_tag).astype(np.uint8)
        elapsed_time = time.time() - start_time
        self.res = Image.fromarray(self.res, 'RGB')
        self.res = ImageTk.PhotoImage(self.res)
        self.canvas.create_image(320,240,image =  self.res)
        print "Draw: %f" % (elapsed_time)'''
        ################################################              
        
        #enable keyboard and mouse monitoring
        self.root.bind("<Key>", self.key)
        self.root.bind("<Button-1>", self.mouse_press)
        self.root.bind("<B1-Motion>", self.mouse_motion)

        self.w = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.w.pack()
        
        self.root.mainloop()





