#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:27:58 2017

@author: diegothomas
"""


import imp
from os import path
import numpy as np
from numpy import linalg as LA
from scipy import optimize as Opt
from math import sin, cos, acos
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
from pyopencl.array import dot
import time



APP_ROOT = path.dirname( path.abspath( __file__ ) )
RGBD = imp.load_source('RGBD', APP_ROOT + '/RGBD.py')
BSManager = imp.load_source('BSManager', APP_ROOT + '/BSManager.py')
KernelsOpenCL = imp.load_source('ExpressionKernel', APP_ROOT + '/ExpressionKernel.py')

class FacialExpression():
    
    # Constructor
    def __init__(self, GPUManager, SizeBS, Size, Intrinsic, thresh_dist, thresh_norm, lvl, max_iter, thresh_conv):
        self.thresh_dist = thresh_dist
        self.thresh_norm = thresh_norm
        self.lvl = lvl
        self.max_iter = max_iter
        self.thresh_conv = thresh_conv
        self.Size = SizeBS
        self.intrinsic = Intrinsic
        self.GPUManager = GPUManager
        self.NB_BS = 28
        
        self.Matrix_B = ElementwiseKernel(self.GPUManager.context, 
                                          """float *BumpImage, int * LabelsMask, float *RGBMapBump, float *NMapBump,  
                                          float *VerticesBS, float *BlendshapeCoeff, float *VMap, float *NMap, float *RGBMap, 
                                          float *Pose, float *calib_depth, float *calib_rgb, double *buf,
                                          float distThres, float angleThres, int n_bump, int m_bump, int n_row, int m_col""",
                                          KernelsOpenCL.Kernel_Matrix_B,
                                          "Matrix_B")
        
        self.my_dot = ReductionKernel(self.GPUManager.context, np.float32, neutral="0",
                                reduce_expr="a+b", map_expr="x[i]*y[i]",
                                arguments="__global float *x, __global float *y")
        
        
        self.my_slice = ElementwiseKernel(self.GPUManager.context,
                                          """float *Buffer_slice, float *Buffer, int col""",
                                          """ Buffer_slice[i] =  Buffer[28*i + col]""",
                                          "my_slice")
        
        
        self.MatMult = ElementwiseKernel(self.GPUManager.context,
                                          """float *output, float *A, float *B, int NB_BS, int k""",
                                          """ val = 0.0;
                                        		#pragma unroll
                                        		for (int l = 0; l < NB_BS-1; l++) {
                                        			val = val + A[k*(NB_BS-1) + l] * B[NB_BS*i + l];
                                        		}
                                        		output[(NB_BS-1)*i + k] = val;""",
                                          "MatMult")
        
        self.Buffer_d = cl.array.zeros(self.GPUManager.queue, (self.Size[0]*self.Size[1], self.NB_BS), dtype = np.float64)
        self.A_inverse_d = cl.array.zeros(self.GPUManager.queue, (self.NB_BS, self.NB_BS), dtype = np.float32)
        
        self.Buffer = []
        self.Mat1 = []
        for i in range(self.NB_BS): 
            self.Buffer.append(cl.array.zeros(self.GPUManager.queue, (self.Size[0]*self.Size[1], 1), dtype = np.float32))
            self.Mat1.append(cl.array.zeros(self.GPUManager.queue, (self.Size[0]*self.Size[1], 1), dtype = np.float32))
        
        self.Pose_d = cl.array.zeros(self.GPUManager.queue, (4,4), np.float32)
        self.intrinsic_d = cl.array.zeros(self.GPUManager.queue, 4, np.float32)
        self.intrinsic_RGB_d = cl.array.zeros(self.GPUManager.queue, 4, np.float32)
        
    def EstimateBSCoeff(self, BS, rgbd, Pose):
        
        self.Pose_d.set(Pose.astype(np.float32))
        self.BlendshapeCoeff_d = BS.BlendshapeCoeff_d.copy()
        
        intrinsic_depth = np.array([rgbd.intrinsic[0,0], rgbd.intrinsic[1,1], rgbd.intrinsic[0,2], rgbd.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_depth)
        intrinsic_rgb = np.array([rgbd.intrinsicRGB[0,0], rgbd.intrinsicRGB[1,1], rgbd.intrinsicRGB[0,2], rgbd.intrinsicRGB[1,2]])
        self.intrinsic_RGB_d = cl.array.to_device(self.GPUManager.queue, intrinsic_rgb)
        
        converged = False
        
        b = np.zeros(self.NB_BS-1, np.float32)
        A = np.zeros((self.NB_BS-1,self.NB_BS-1), np.float32)
        
        rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
        
        landmark = np.ones(4, dtype = np.float32)
        landmark_mesh = np.ones(4, dtype = np.float32)
        landmark_mesh_nmle = np.zeros(3, dtype = np.float32)
        
        Smooth_cnst = np.zeros((self.NB_BS-1,self.NB_BS), np.float64)
        for i in range(self.NB_BS-1):
            Smooth_cnst[i,i] = 0.5
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                fact = 2**(self.lvl-l)
                
                ''' 1. Compute matrix B **************************'''
                # Component from the geometry
                self.Matrix_B(BS.BumpImage_d, BS.labels_d, BS.RGBBump_d, BS.NMapBump_d, BS.VerticesBS_d, self.BlendshapeCoeff_d,
                              rgbd.vmap_d, rgbd.nmap_d, rgbd.color_d, self.Pose_d, 
                              self.intrinsic_d, self.intrinsic_RGB_d, self.Buffer_d,
                              self.thresh_dist, self.thresh_norm, BS.Size[0], BS.Size[1], rgbd.Size[0], rgbd.Size[1])
                
                tmp_Mat = np.vstack((self.Buffer_d.get(),Smooth_cnst))
                
                Landmark_cnst = np.zeros((3*51,self.NB_BS), np.float64)
                 # Add correspondences from facial features
                for k in range(51):
                    idx = rgb2depth[rgbd.shape[k][1], rgbd.shape[k][0]]
                    if (idx == 0):
                        continue
                    
                    landmark[0:3] = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1],:]
                    
                    landmark_mesh[0:3] = BS.BlendShapes[0].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3]
                    #landmark_mesh_nmle[0:3] = BS.BlendShapes[0].Normales[BSManager.FACIAL_LANDMARKS[k],0:3]
                    
                    for bs_idx in range(1,28):
                        landmark_mesh[0:3] = landmark_mesh[0:3] + (BS.BlendShapes[bs_idx].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3]-BS.BlendShapes[0].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3])*BS.BlendshapeCoeff[bs_idx]
                    #    landmark_mesh_nmle[0:3] = landmark_mesh_nmle[0:3] + (BS.BlendShapes[bs_idx].Normales[BSManager.FACIAL_LANDMARKS[k],0:3]-BS.BlendShapes[0].Normales[BSManager.FACIAL_LANDMARKS[k],0:3])*BS.BlendshapeCoeff[bs_idx]
                    
                    landmark_mesh = np.dot(Pose, landmark_mesh)
                    #landmark_mesh_nmle = np.dot(Pose[0:3,0:3], landmark_mesh_nmle)
                    
                    #if (landmark_mesh_nmle[2] > 0.0):
                        #print ("non visible landmark", k)
                    #    continue
                    
                    dist = LA.norm(landmark[0:3]-landmark_mesh[0:3])
                    if (dist > 0.1):
                        #print "too far"
                        continue
                    
                    for bs_idx in range(1,self.NB_BS):
                        tmp_v = (BS.BlendShapes[bs_idx].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3]-BS.BlendShapes[0].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3])
                        tmp_v = np.dot(Pose[0:3,0:3], tmp_v)
                        Landmark_cnst[3*k:3*k+3,bs_idx-1] = 10.0*tmp_v
                    
                    Landmark_cnst[3*k:3*k+3,self.NB_BS-1] = -10.0*(landmark_mesh[0:3] - landmark[0:3])
                    
                
                tmp_Mat = np.vstack((tmp_Mat,Landmark_cnst))    
                
                res = Opt.lsq_linear(tmp_Mat[::,0:self.NB_BS-1], tmp_Mat[::,self.NB_BS-1], bounds=(-0.2, 0.2))
                #print res['x']
                #print res['cost']
                
                BS.BlendshapeCoeff[1::] = BS.BlendshapeCoeff[1::] + res['x'].astype(np.float32) #BS.BlendshapeCoeff[1::] + 
                BS.BlendshapeCoeff[BS.BlendshapeCoeff < 0.0] = 0.0
                BS.BlendshapeCoeff[BS.BlendshapeCoeff > 1.0] = 1.0
                self.BlendshapeCoeff_d.set(BS.BlendshapeCoeff)
        
        print BS.BlendshapeCoeff
        BS.BlendshapeCoeff_d.set(BS.BlendshapeCoeff)
                
                