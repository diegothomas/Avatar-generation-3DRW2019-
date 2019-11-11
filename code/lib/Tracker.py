#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:24:42 2017

@author: diegothomas
"""

import imp
from os import path
import numpy as np
from numpy import linalg as LA
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
KernelsOpenCL = imp.load_source('TrackerKernel', APP_ROOT + '/TrackerKernel.py')

mf = cl.mem_flags

def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

def Exponential(qsi):
    theta = LA.norm(qsi[3:6])
    res = np.identity(4)
    
    if (theta != 0.):
        res[0,0] = 1.0 + sin(theta)/theta*0.0 + (1.0 - cos(theta)) / (theta*theta) * (-qsi[5]*qsi[5] - qsi[4]*qsi[4])
        res[1,0] = 0.0 + sin(theta)/theta*qsi[5] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[4])
        res[2,0] = 0.0 - sin(theta)/theta*qsi[4] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[5])
        
        res[0,1] = 0.0 - sin(theta)/theta*qsi[5] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[4])
        res[1,1] = 1.0 + sin(theta) / theta*0.0 + (1.0 - cos(theta))/(theta*theta) * (-qsi[5]*qsi[5] - qsi[3]*qsi[3])
        res[2,1] = 0.0 + sin(theta)/theta*qsi[3] + (1.0 - cos(theta))/(theta*theta) * (qsi[4]*qsi[5])
        
        res[0,2] = 0.0 + sin(theta) / theta*qsi[4] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[5])
        res[1,2] = 0.0 - sin(theta)/theta*qsi[3] + (1.0 - cos(theta))/(theta*theta) * (qsi[4]*qsi[5])
        res[2,2] = 1.0 + sin(theta)/theta*0.0 + (1.0 - cos(theta))/(theta*theta) * (-qsi[4]*qsi[4] - qsi[3]*qsi[3])
        
        skew = np.zeros((3,3), np.float32)
        skew[0,1] = -qsi[5]
        skew[0,2] = qsi[4]
        skew[1,0] = qsi[5]
        skew[1,2] = -qsi[3]
        skew[2,0] = -qsi[4]
        skew[2,1] = qsi[3]
        
        V = np.identity(3) + ((1.0-cos(theta))/(theta*theta))*skew + ((theta - sin(theta))/(theta*theta))*np.dot(skew,skew)
        
        res[0,3] = V[0,0]*qsi[0] + V[0,1]*qsi[1] + V[0,2]*qsi[2]
        res[1,3] = V[1,0]*qsi[0] + V[1,1]*qsi[1] + V[1,2]*qsi[2]
        res[2,3] = V[2,0]*qsi[0] + V[2,1]*qsi[1] + V[2,2]*qsi[2]
    else:
        res[0,3] = qsi[0]
        res[1,3] = qsi[1]
        res[2,3] = qsi[2]
        
    return res

def Logarithm(Mat):
    trace = Mat[0,0]+Mat[1,1]+Mat[2,2]
    theta = acos((trace-1.0)/2.0)
    
    qsi = np.array([0.,0.,0.,0.,0.,0.])
    if (theta == 0.):
        qsi[3] = qsi[4] = qsi[5] = 0.0
        qsi[0] = Mat[0,3]
        qsi[1] = Mat[1,3]
        qsi[2] = Mat[2,3]
        return qsi
    
    R = Mat[0:3,0:3]
    lnR = (theta/(2.0*sin(theta))) * (R-np.transpose(R))
    
    qsi[3] = (lnR[2,1] - lnR[1,2])/2.0
    qsi[4] = (lnR[0,2] - lnR[2,0])/2.0
    qsi[5] = (lnR[1,0] - lnR[0,1])/2.0
    
    theta = LA.norm(qsi[3:6])

    skew = np.zeros((3,3), np.float32)
    skew[0,1] = -qsi[5]
    skew[0,2] = qsi[4]
    skew[1,0] = qsi[5]
    skew[1,2] = -qsi[3]
    skew[2,0] = -qsi[4]
    skew[2,1] = qsi[3]
    
    V = np.identity(3) + ((1.0 - cos(theta))/(theta*theta))*skew + ((theta-sin(theta))/(theta*theta))*np.dot(skew,skew)
    V_inv = LA.inv(V)
    
    qsi[0] = V_inv[0,0]*Mat[0,3] + V_inv[0,1]*Mat[1,3] + V_inv[0,2]*Mat[2,3]
    qsi[1] = V_inv[1,0]*Mat[0,3] + V_inv[1,1]*Mat[1,3] + V_inv[1,2]*Mat[2,3]
    qsi[2] = V_inv[2,0]*Mat[0,3] + V_inv[2,1]*Mat[1,3] + V_inv[2,2]*Mat[2,3]
    
    return qsi
    

class Tracker():

    # Constructor
    def __init__(self, GPUManager, Size, Intrinsic, thresh_dist, thresh_norm, lvl, max_iter, thresh_conv):
        self.thresh_dist = thresh_dist
        self.thresh_norm = thresh_norm
        self.lvl = lvl
        self.max_iter = max_iter
        self.thresh_conv = thresh_conv
        self.Size = Size
        self.intrinsic = Intrinsic
        self.GPUManager = GPUManager
        
        self.GICP = ElementwiseKernel(self.GPUManager.context, 
                                               """float *vmap, float *nmap, float *vmap2, float *nmap2, float *Buffer_1, 
                                               float *Buffer_2, float *Buffer_3, float *Buffer_4, float *Buffer_5, 
                                               float *Buffer_6, float *Buffer_B, float *Pose, float *Intrinsic, 
                                               float thresh_dis, float thresh_norm, int nbLines, int nbColumns""",
                                               KernelsOpenCL.Kernel_GICP,
                                               "GICP")
        
        self.GICP_M2Depth = ElementwiseKernel(self.GPUManager.context, 
                                               """float *Vertices, float *Normales, float *vmap2, float *nmap2, float *Buffer_1, 
                                               float *Buffer_2, float *Buffer_3, float *Buffer_4, float *Buffer_5, 
                                               float *Buffer_6, float *Buffer_B, float *Pose, float *Intrinsic, 
                                               float thresh_dis, float thresh_norm, int nbLines, int nbColumns, int fact, int nbVertices""",
                                               KernelsOpenCL.Kernel_GICP_M2Depth,
                                               "GICP_M2Depth")
        
        self.my_dot = ReductionKernel(self.GPUManager.context, np.float32, neutral="0",
                                reduce_expr="a+b", map_expr="x[i]*y[i]",
                                arguments="__global float *x, __global float *y")
        
        intrinsic_curr = np.array([self.intrinsic[0,0], self.intrinsic[1,1], self.intrinsic[0,2], self.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        self.Pose_d = cl.array.zeros(self.GPUManager.queue, (4,4), np.float32)
        
        self.Buffer_d_1 = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.Buffer_d_2 = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.Buffer_d_3 = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.Buffer_d_4 = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.Buffer_d_5 = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.Buffer_d_6 = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.Buffer_d = [self.Buffer_d_1, self.Buffer_d_2, self.Buffer_d_3, self.Buffer_d_4, self.Buffer_d_5, self.Buffer_d_6]
        self.Buffer_B_d = cl.array.zeros(self.GPUManager.queue, (1, 1), dtype = np.float32)
        self.buff_size = 1
        
    """
    Function that estimate the relative rigid transformation between two input RGB-D images
    """
    def RegisterRGBD_GPU(self, Image1, Image2):
        res = np.identity(4)
        b = np.zeros(6, np.float32)
        A = np.zeros((6,6), np.float32)
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                self.Pose_d.set(res.astype(np.float32))
                
                self.GICP(Image1.vmap_d, Image1.nmap_d, Image2.vmap_d, Image2.nmap_d,
                          self.Buffer_d_1, self.Buffer_d_2, self.Buffer_d_3, self.Buffer_d_4, self.Buffer_d_5,
                          self.Buffer_d_6, self.Buffer_B_d, self.Pose_d, self.intrinsic_d,
                          self.thresh_dist, self.thresh_norm, self.Size[0], self.Size[1])
                
                for i in range(6):
                    for j in range(i,7):
                        if (j == 6):
                            value = self.my_dot(self.Buffer_d[i],self.Buffer_B_d).get()
                            value = value.reshape((1,1))
                            b[i] = value[0,0]
                        else:
                            value = self.my_dot(self.Buffer_d[i],self.Buffer_d[j]).get()
                            value = value.reshape((1,1))
                            A[i,j] = A[j,i] = value[0,0]
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
        print res
        
                
    def RegisterRGBD_CPU(self, Image1, Image2):
        
        res = np.identity(4)
        
        column_index_ref = np.array([np.array(range(Image1.Size[1])) for _ in range(Image1.Size[0])])
        line_index_ref = np.array([x*np.ones(Image1.Size[1], np.int) for x in range(Image1.Size[0])])
        Indexes_ref = column_index_ref + Image1.Size[1]*line_index_ref
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                #nbMatches = 0
                #row = np.array([0.,0.,0.,0.,0.,0.,0.])
                #Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((Image1.Size[0]*Image1.Size[1], 6), dtype = np.float32)
                Buffer_B = np.zeros((Image1.Size[0]*Image1.Size[1], 1), dtype = np.float32)
                stack_pix = np.ones((Image1.Size[0], Image1.Size[1]), dtype = np.float32)
                stack_pt = np.ones((np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1)), dtype = np.float32)
                pix = np.zeros((Image1.Size[0], Image1.Size[1],2), dtype = np.float32)
                pix = np.dstack((pix,stack_pix))
                pt = np.dstack((Image1.Vtx[ ::l, ::l, :],stack_pt))
                pt = np.dot(res,pt.transpose(0,2,1)).transpose(1,2,0)
                nmle = np.zeros((Image1.Size[0], Image1.Size[1],Image1.Size[2]), dtype = np.float32)
                nmle[ ::l, ::l,:] = np.dot(res[0:3,0:3],Image1.Nmls[ ::l, ::l,:].transpose(0,2,1)).transpose(1,2,0)
                
                #if (pt[2] != 0.0):
                lpt = np.dsplit(pt,4)
                lpt[2] = in_mat_zero2one(lpt[2])
                
                # if in 1D pix[0] = pt[0]/pt[2]
                pix[ ::l, ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1))
                # if in 1D pix[1] = pt[1]/pt[2]
                pix[ ::l, ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1))
                pix = np.dot(Image1.intrinsic,pix[0:Image1.Size[0],0:Image1.Size[1]].transpose(0,2,1)).transpose(1,2,0)
                column_index = (np.round(pix[:,:,0])).astype(int)
                line_index = (np.round(pix[:,:,1])).astype(int)
                
                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < Image2.Size[1])
                cdt_line = (line_index > -1) * (line_index < Image2.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column
                #Indexes = line_index + Image2.Size[0]*column_index
                Indexes = column_index + Image2.Size[1]*line_index
                
                diff_Vtx = Image2.Vtx[line_index[:][:], column_index[:][:]] - pt[:,:,0:3]
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=2)
                
                diff_Nmle = Image2.Nmls[line_index[:][:], column_index[:][:]] - nmle
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=2)
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
                
                mask = cdt_line*cdt_column * (pt[:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print sum(sum(mask))
                
                w = 1.0
                Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
                      w*mask[:,:]*nmle[ :, :,1], \
                      w*mask[:,:]*nmle[ :, :,2], \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,1] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,2]), \
                      w*mask[:,:]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,0] - Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,2]), \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,0] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,1]) ))
                
                Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(nmle[:,:,0]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0] - pt[:,:,0]) + nmle[:,:,1]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1] - pt[:,:,1]) + nmle[:,:,2]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2] - pt[:,:,2])) ).transpose()
                        
                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                print res
                
    """
    Function that estimate the relative rigid transformation between an input mesh and an RGB-D image
    """
    def RegisterMesh2RGBD_GPU(self, MC, Image2, pose):
        
        """ Initialise buffers """
        if (MC.nb_vertices[0] > self.buff_size):
            self.buff_size = MC.nb_vertices[0] + 10000
            self.Buffer_d_1 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_2 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_3 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_4 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_5 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_6 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d = [self.Buffer_d_1, self.Buffer_d_2, self.Buffer_d_3, self.Buffer_d_4, self.Buffer_d_5, self.Buffer_d_6]
            self.Buffer_B_d = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
        else:
            self.Buffer_d_1.fill(0.0) 
            self.Buffer_d_2.fill(0.0) 
            self.Buffer_d_3.fill(0.0) 
            self.Buffer_d_4.fill(0.0) 
            self.Buffer_d_5.fill(0.0) 
            self.Buffer_d_6.fill(0.0) 
            self.Buffer_B_d.fill(0.0) 

        res = pose
        b = np.zeros(6, np.float32)
        A = np.zeros((6,6), np.float32)
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                self.Pose_d.set(res.astype(np.float32))
                fact = 4**(self.lvl-l)
                
                self.GICP_M2Depth(MC.VerticesGPU, MC.NormalesGPU, Image2.vmap_d, Image2.nmap_d,
                          self.Buffer_d_1, self.Buffer_d_2, self.Buffer_d_3, self.Buffer_d_4, self.Buffer_d_5,
                          self.Buffer_d_6, self.Buffer_B_d, self.Pose_d, self.intrinsic_d,
                          self.thresh_dist, self.thresh_norm, self.Size[0], self.Size[1], fact, MC.nb_vertices[0])
                
                self.GPUManager.queue.finish()
            
                for i in range(6):
                    for j in range(i,7):
                        if (j == 6):
                            value = self.my_dot(self.Buffer_d[i],self.Buffer_B_d).get()
                            value = value.reshape((1,1))
                            b[i] = value[0,0]
                        else:
                            value = self.my_dot(self.Buffer_d[i],self.Buffer_d[j]).get()
                            value = value.reshape((1,1))
                            A[i,j] = A[j,i] = value[0,0]
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                #print res
        return res
                
    
    
    """
    Function that estimate the relative rigid transformation between augmented mesh and RGB-D images
    """
    def RegisterBumpToRGBD_GPU(self, BS, Image2, pose):
        
        """ Initialise buffers """
        if (self.buff_size == 1):
            self.buff_size = BS.Size[0]*BS.Size[1]
            self.Buffer_d_1 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_2 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_3 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_4 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_5 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d_6 = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
            self.Buffer_d = [self.Buffer_d_1, self.Buffer_d_2, self.Buffer_d_3, self.Buffer_d_4, self.Buffer_d_5, self.Buffer_d_6]
            self.Buffer_B_d = cl.array.zeros(self.GPUManager.queue, (self.buff_size, 1), dtype = np.float32)
        else:
            self.Buffer_d_1.fill(0.0) 
            self.Buffer_d_2.fill(0.0) 
            self.Buffer_d_3.fill(0.0) 
            self.Buffer_d_4.fill(0.0) 
            self.Buffer_d_5.fill(0.0) 
            self.Buffer_d_6.fill(0.0) 
            self.Buffer_B_d.fill(0.0) 
            
            
        rgb2depth = Image2.RGB2Depth.get()
        vmap = Image2.vmap_d.get()
        
        landmark = np.ones(4, dtype = np.float32)
        landmark_mesh = np.ones(4, dtype = np.float32)
        landmark_mesh_nmle = np.zeros(3, dtype = np.float32)
            
        res = pose
        b = np.zeros(6, np.float32)
        A = np.zeros((6,6), np.float32)
        row = np.zeros(7, np.float32)
        weight = 10.0
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                self.Pose_d.set(res.astype(np.float32))
                
                self.GICP(BS.VMapBump_d, BS.NMapBump_d, Image2.vmap_d, Image2.nmap_d,
                          self.Buffer_d_1, self.Buffer_d_2, self.Buffer_d_3, self.Buffer_d_4, self.Buffer_d_5,
                          self.Buffer_d_6, self.Buffer_B_d, self.Pose_d, self.intrinsic_d,
                          self.thresh_dist, self.thresh_norm, self.Size[0], self.Size[1])
                
                for i in range(6):
                    for j in range(i,7):
                        if (j == 6):
                            value = self.my_dot(self.Buffer_d[i],self.Buffer_B_d).get()
                            value = value.reshape((1,1))
                            b[i] = value[0,0]
                        else:
                            value = self.my_dot(self.Buffer_d[i],self.Buffer_d[j]).get()
                            value = value.reshape((1,1))
                            A[i,j] = A[j,i] = value[0,0]
                            
                
                # Add correspondences from facial features
                for k in range(51):
                    idx = rgb2depth[Image2.shape[k][1], Image2.shape[k][0]]
                    if (idx == 0):
                        #print "No landmark"
                        continue
                    
                    landmark[0:3] = vmap[idx/Image2.Size[1], idx%Image2.Size[1],:]
                    
                    landmark_mesh[0:3] = BS.BlendShapes[0].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3]
                    landmark_mesh_nmle[0:3] = BS.BlendShapes[0].Normales[BSManager.FACIAL_LANDMARKS[k],0:3]
                    
                    for bs_idx in range(1,28):
                        landmark_mesh[0:3] = landmark_mesh[0:3] + (BS.BlendShapes[bs_idx].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3]-BS.BlendShapes[0].Vertices[BSManager.FACIAL_LANDMARKS[k],0:3])*BS.BlendshapeCoeff[bs_idx]
                        landmark_mesh_nmle[0:3] = landmark_mesh_nmle[0:3] + (BS.BlendShapes[bs_idx].Normales[BSManager.FACIAL_LANDMARKS[k],0:3]-BS.BlendShapes[0].Normales[BSManager.FACIAL_LANDMARKS[k],0:3])*BS.BlendshapeCoeff[bs_idx]
                    
                    landmark_mesh = np.dot(res, landmark_mesh)
                    landmark_mesh_nmle = np.dot(res[0:3,0:3], landmark_mesh_nmle)
                    
                    if (landmark_mesh_nmle[2] > 0.0):
                        #print ("non visible landmark", k)
                        continue
                    
                    dist = LA.norm(landmark[0:3]-landmark_mesh[0:3])
                    if (dist > 0.1):
                        #print "too far"
                        continue
                    
                    row[0] = weight*landmark_mesh_nmle[0]
                    row[1] = weight*landmark_mesh_nmle[1]
                    row[2] = weight*landmark_mesh_nmle[2]
                    row[3] = weight*(-landmark[2]*landmark_mesh_nmle[1] + landmark[1]*landmark_mesh_nmle[2])
                    row[4] = weight*(landmark[2]*landmark_mesh_nmle[0] - landmark[0]*landmark_mesh_nmle[2])
                    row[5] = weight*(-landmark[1]*landmark_mesh_nmle[0] + landmark[0]*landmark_mesh_nmle[1])
                       
                    row[6] = weight*(landmark_mesh_nmle[0]*(landmark[0] - landmark_mesh[0]) + 
                               landmark_mesh_nmle[1]*(landmark[1] - landmark_mesh[1]) + 
                               landmark_mesh_nmle[2]*(landmark[2] - landmark_mesh[2]))
        
                    for i in range(6):
                        for j in range(i,7):
                            if (j == 6):
                                b[i] = b[i] + row[i]*row[j]
                            else:
                                A[i,j] = A[i,j] + row[i]*row[j]
                                A[j,i] = A[j,i] + row[i]*row[j]
                                
                
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
        return res