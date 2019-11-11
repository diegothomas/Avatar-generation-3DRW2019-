#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:24:42 2017

@author: diegothomas
"""

# Define functions to manipulate RGB-D data
import cv2
import numpy as np
from numpy import linalg as LA
import imp
from os import path
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import dlib
from imutils import face_utils

APP_ROOT = path.dirname( path.abspath( __file__ ) )
KernelsOpenCL = imp.load_source('RGBDKernels', APP_ROOT + '/RGBDKernels.py')

def normalized_cross_prod(a,b):
    res = np.zeros(3, dtype = "float")
    if (LA.norm(a) == 0.0 or LA.norm(b) == 0.0):
        return res
    a = a/LA.norm(a)
    b = b/LA.norm(b)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = -a[0]*b[2] + a[2]*b[0]
    res[2] = a[0]*b[1] - a[1]*b[0]
    if (LA.norm(res) > 0.0):
        res = res/LA.norm(res)
    return res


def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

def division_by_norm(mat,norm):
    """This fonction divide a n by m by p=3 matrix, point by point, by the norm made through the p dimension>
    It ignores division that makes infinite values or overflow to replace it by the former mat values or by 0"""
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            mat[:,:,i] = np.true_divide(mat[:,:,i],norm)
            mat[:,:,i][mat[:,:,i] == np.inf] = 0
            mat[:,:,i] = np.nan_to_num(mat[:,:,i])
    return mat
                
def normalized_cross_prod_optimize(a,b):
    #res = np.zeros(a.Size, dtype = "float")
    norm_mat_a = np.sqrt(np.sum(a*a,axis=2))
    norm_mat_b = np.sqrt(np.sum(b*b,axis=2))
    #changing every 0 to 1 in the matrix so that the division does not generate nan or infinite values
    norm_mat_a = in_mat_zero2one(norm_mat_a)
    norm_mat_b = in_mat_zero2one(norm_mat_b)
    # compute a/ norm_mat_a
    a = division_by_norm(a,norm_mat_a)
    b = division_by_norm(b,norm_mat_b)
    #compute cross product with matrix
    res = np.cross(a,b)
    #compute the norm of res using the same method for a and b 
    norm_mat_res = np.sqrt(np.sum(res*res,axis=2))
    norm_mat_res = in_mat_zero2one(norm_mat_res)
    #norm division
    res = division_by_norm(res,norm_mat_res)
    return res

'''
    The main class that manage RGB-D data
'''
class RGBD():

    ''' Constructor '''
    def __init__(self, GPUManager, depthname, colorname, intrinsic, size, fact):
        self.depthname = depthname
        self.colorname = colorname
        self.intrinsic = intrinsic
        self.fact = fact
        self.GPUManager = GPUManager
        # index of current input image
        self.index = 0
        self.Size = size
        print self.Size
        
        self.intrinsicRGB = self.intrinsic
        self.Calib  = np.array([[1., 0., 0., 0.], 
                              [0., 1., 0., 0.], 
                              [0., 0., 1., 0.], 
                              [0., 0., 0., 1.]], dtype = np.float32)
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.bilateral_filter = ElementwiseKernel(self.GPUManager.context, 
                                               "float *depth, float *depth_out, int d, float sigma_r, float sigma_d, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_BilateralFilter,
                                               "bilateral_filter")
               
        self.depth_to_vmap = ElementwiseKernel(self.GPUManager.context, 
                                               "float *depth, float *vmap, float *intrinsic, int nbColumns",
                                               KernelsOpenCL.Kernel_VMap,
                                               "depth_to_vmap")
        
        self.vmap_to_nmap = ElementwiseKernel(self.GPUManager.context, 
                                               "float *vmap, float *nmap, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_NMap,
                                               "vmap_to_nmap")
        
        self.draw_vmap = ElementwiseKernel(self.GPUManager.context, 
                                               "unsigned int *res, float *vmap, float *nmap, unsigned char *color, float *Pose, float *Intrinsic, int color_flag, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_Draw,
                                               "draw_vmap")
        
        self.reproj_depth = ElementwiseKernel(self.GPUManager.context, 
                                               "int *RGB2Depth, unsigned char *color, unsigned char *color_buff, float *vmap, float *Pose, float *Intrinsic, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_Reproj,
                                               "reproj_depth")
        
        self.transform_vnmap = ElementwiseKernel(self.GPUManager.context, 
                                               "float *vmap, float *nmap, float *Pose",
                                               KernelsOpenCL.Kernel_Transform,
                                               "transform_vnmap")
        # Allocate on CPU 
        self.draw_result = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint32)
        
        # Allocate on GPU 
        intrinsic_curr = np.array([self.intrinsic[0,0], self.intrinsic[1,1], self.intrinsic[0,2], self.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        self.Pose_d = cl.array.zeros(self.GPUManager.queue, (4,4), np.float32)
        
        self.depth_raw_d = cl.array.zeros(self.GPUManager.queue, (self.Size[0], self.Size[1]), np.float32)
        self.depth_d = cl.array.zeros(self.GPUManager.queue, (self.Size[0], self.Size[1]), np.float32)
        self.depth_d_buff = cl.array.zeros(self.GPUManager.queue, (self.Size[0], self.Size[1]), np.float32)
        self.vmap_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.nmap_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.color_d = cl.array.zeros(self.GPUManager.queue, self.Size, dtype = np.uint8)
        self.color_d_buff = cl.array.zeros(self.GPUManager.queue, self.Size, dtype = np.uint8)
        self.draw_d = cl.array.zeros(self.GPUManager.queue, self.Size, dtype = np.uint32)
        self.RGB2Depth = cl.array.zeros(self.GPUManager.queue, (self.Size[0], self.Size[1]), np.int32)
        
    
    ''' Reads an input RGBD image from the disk '''
    def ReadFromDisk(self): #Read an RGB-D image from the disk
        self.color_image = cv2.imread(self.colorname)
        if (self.color_image is None):
            return False
        depth_raw = cv2.imread(self.depthname,-1)
        self.depth_image = np.ascontiguousarray(depth_raw.astype(np.float32) / self.fact)
        self.depth_raw_d.set(self.depth_image, queue = self.GPUManager.queue)
        self.color_d.set(self.color_image, queue = self.GPUManager.queue)
        self.index+=1
        return True
                                
    
    ''' Create the vertex image from the depth image and intrinsic matrice '''
    def Vmap(self): 
        d = self.depth_image[0:self.Size[0]][0:self.Size[1]]
        d_pos = d * (d > 0.0)#depth im
        x_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)#number of row?
        y_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)#number of column?
        # change the matrix so that the first row is on all rows for x respectively colunm for y.
        x_raw[0:-1,:] = ( np.arange(self.Size[1]) - self.intrinsic[0,2])/self.intrinsic[0,0]#x-cx/fx
        y_raw[:,0:-1] = np.tile( ( np.arange(self.Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()#(y-cy/fy )^T
        # multiply point by point d_pos and raw matrices
        x = d_pos * x_raw
        y = d_pos * y_raw
        self.Vtx = np.dstack((x, y,d))
        
    ''' ON GPU: Create the vertex image from the depth image and intrinsic matrice ''' 
    def Vmap_GPU(self, mode = 0):
        intrinsic_curr = np.array([self.intrinsic[0,0], self.intrinsic[1,1], self.intrinsic[0,2], self.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        if (mode == 0):
            self.depth_to_vmap(self.depth_d, self.vmap_d, self.intrinsic_d, self.Size[1])
        else:
            self.depth_to_vmap(self.depth_d_buff, self.vmap_d, self.intrinsic_d, self.Size[1])
        

    ''' Compute normals '''
    def NMap(self):
        self.Nmls = np.zeros(self.Size, np.float32)        
        nmle1 = normalized_cross_prod_optimize(self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])        
        nmle2 = normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle3 = normalized_cross_prod_optimize(self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle4 = normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
        norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
        norm_mat_nmle = in_mat_zero2one(norm_mat_nmle)
        #norm division 
        nmle = division_by_norm(nmle,norm_mat_nmle)
        self.Nmls[1:self.Size[0]-1][:,1:self.Size[1]-1] = nmle
        
    ''' On GPU: Compute normals '''
    def NMap_GPU(self):
        self.vmap_to_nmap(self.vmap_d, self.nmap_d, self.Size[0], self.Size[1])
        self.Vtx = self.vmap_d.get()
        self.Nmls = self.nmap_d.get()

    ''' Draw the RGBD image onto a virtual image plane with 
            camera pose: Pose
            scale: s
            color mode: color
    '''
    def Draw(self, Pose, s, color = 0) :
        result = np.zeros(self.Size, dtype = np.uint8)
        raw_ones = np.ones([self.Size[0],self.Size[1],1],dtype=np.float32)
        raw_pos = np.dstack((self.Vtx,raw_ones))#[x,y,z,1]
        VtxPos =  np.tensordot(Pose,raw_pos,(1,2)).transpose(1,2,0)#P*V
        Nmls_Pose = np.tensordot(Pose[:3,:3],self.Nmls,(1,2)).transpose(1,2,0)#Nmls Pose(R) change

        
        #Vtx -> depth
        fx = self.intrinsic[0,0]
        x_proj = np.ones([self.Size[0],self.Size[1]],np.float32)
        x_proj = (x_proj * fx) #+ self.intrinsic[0,2]  #x = 1*X #+ cx #make fx matrix

        fy = self.intrinsic[1,1]
        y_proj = np.ones([self.Size[0],self.Size[1]],np.float32)
        y_proj = (y_proj*fy) #+ self.intrinsic[1,2]    #y = 1*Y #+ cy# make fy matrix
                
        #negative or 0 -> 1 @ 0 division avoid
        mask_z = VtxPos[:,:,2] <= 0
        Vtx_mask = VtxPos[:,:,2]
        Vtx_mask[mask_z] = 1


        #projection matrix
        x_proj_int = x_proj*VtxPos[:,:,0]#x=fxX
        x_proj_int = x_proj_int/Vtx_mask#x= fxX/Z (Z!=0)
        x_proj_int = x_proj_int +  self.intrinsic[0,2]#x= fxX/Z + cy
        x_proj_int = np.floor(x_proj_int).astype(int)# x = (int)x 
        x_proj_int[x_proj_int >= 640] = 0#out of window
        x_proj_int[x_proj_int < 0] = 0#out of window

        y_proj_int = y_proj*VtxPos[:,:,1]#y=fyY
        y_proj_int = y_proj_int/Vtx_mask#y=fyY/Z
        y_proj_int = y_proj_int +  self.intrinsic[1,2]#y=fyY/z + cy
        y_proj_int = np.floor(y_proj_int).astype(int)#y = (int)y    
        y_proj_int[y_proj_int >= 480] = 0#out of window
        y_proj_int[y_proj_int < 0] = 0#out of window


        #rendering images        
        rend_Vtx = np.zeros([self.Size[0],self.Size[1]], dtype = np.float32)        
        rend_color = np.zeros(self.Size, dtype = np.uint8)        
        rend_Nmls = np.zeros(self.Size, dtype = np.float32)        
        
        rend_Vtx[y_proj_int,x_proj_int]=VtxPos[:,:,2]#[x,y]=[d]
        rend_color[y_proj_int,x_proj_int]=self.color_image
        rend_Nmls[y_proj_int,x_proj_int]=Nmls_Pose
        
        
        #change color type
        if (color == 0):#depth
            result = np.dstack(( rend_Vtx , rend_Vtx, rend_Vtx)) * 100 * 255
            result = result.astype(np.uint8)
        elif(color == 1) : #color
            result = rend_color
            result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        else : #normal image
            result = (rend_Nmls + 1) * 100  
            result = result.astype(np.uint8)
            


        return result
    
    ''' On GPU '''
    def Draw_GPU(self, Pose, s, color = 0):  
        self.Pose_d.set(Pose.astype(np.float32))
        intrinsic_curr = np.array([self.intrinsic[0,0], self.intrinsic[1,1], self.intrinsic[0,2], self.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        self.draw_d.set(self.draw_result)
        self.draw_vmap(self.draw_d, self.vmap_d, self.nmap_d, self.color_d_buff, self.Pose_d, self.intrinsic_d, color, self.Size[0], self.Size[1])
        return self.draw_d.get()
    
    
    ''' On GPU '''
    def ReProj_depth(self, Pose):  
        intrinsic_curr = np.array([self.intrinsicRGB[0,0], self.intrinsicRGB[1,1], self.intrinsicRGB[0,2], self.intrinsicRGB[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        self.Pose_d.set(Pose.astype(np.float32))
        self.RGB2Depth.set(np.zeros((self.Size[0], self.Size[1]), dtype = np.int32))
        #self.color_d_buff = self.color_d.copy()
        self.color_d_buff.set(np.zeros((self.Size[0], self.Size[1],3), dtype = np.uint8))
        self.reproj_depth(self.RGB2Depth, self.color_d_buff, self.color_d, self.vmap_d, self.Pose_d, self.intrinsic_d, self.Size[0], self.Size[1])


    
##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        self.depth_image = (self.depth_image[:,:] > 0.0) * cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)
        print self.depth_image.shape
        
    ''' On GPU '''
    def BilateralFilter_GPU(self, d, sigma_color, sigma_space):
        self.bilateral_filter(self.depth_raw_d, self.depth_d, d, sigma_color, sigma_space, self.Size[0], self.Size[1])


##################################################################
###################Transformation Funtion#######################
##################################################################
    def Transform(self, Pose):
        ## TO DO
        print "TO DO: Implement function Transform"
              
    ''' On GPU'''
    def Transform_GPU(self, Pose):
        self.Pose_d.set(Pose.astype(np.float32))
        self.transform_vnmap(self.vmap_d, self.nmap_d, self.Pose_d)
        self.Vtx = self.vmap_d.get()
        self.Nmls = self.nmap_d.get()
        

##################################################################
###################Face detection#######################
##################################################################
    def DetectFace(self):
        gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        
        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        
        if (len(rects) == 0):
            return False
        
        # Keep only th first face
        self.rect = rects[0]
        
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        self.shape = self.predictor(gray, self.rect)
        
        self.shape = face_utils.shape_to_np(self.shape)[17::]
        
        print len(self.shape), " facial features detected"

        
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in self.shape:
        #(x, y) = self.shape[13]
            cv2.circle(self.color_image, (x, y), 1, (0, 0, 255), -1)
    	  
        # show the frame
        cv2.imshow("Frame", self.color_image)
        #cv2.imwrite('../../../Results/WrinkleMe/Input/Input_RGB_'+str(self.index)+'.png',self.color_image)
        cv2.waitKey(1) & 0xFF
        
        
        return True