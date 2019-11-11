#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:06:51 2017

@author: diegothomas
"""
import imp
from os import path
import numpy as np
from numpy import linalg as LA
from math import sqrt, acos, cos, sin
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

APP_ROOT = path.dirname( path.abspath( __file__ ) )
KernelsOpenCL = imp.load_source('BSKernels', APP_ROOT + '/BSKernels.py')


FACIAL_LANDMARKS = [3749, 3745, 3731, 3725, 3704, 1572, 1592, 1598, 1612, 1617, 3662, 
                    2207, 3093, 966, 2650, 2774, 2693, 662, 558, 2345, 2336, 2369, 
                    4266, 2237, 2298, 2096, 1981, 1980, 1978, 237, 181, 2838, 4307, 
                    4304, 4301, 4298, 4295, 732, 913, 921, 1034, 3024, 3016, 2865, 
                    2728, 2188,  629, 733, 1011, 3114, 3110]

BackIndices = [174, 175, 1063, 1064, 1065, 1062, 1066, 1067, 1068, 1069, 1070, 1075, 1076, 1074, 1077, 1078, 
               1079, 1106, 1107, 1104, 1103, 1108, 1110, 1145, 1146, 1147, 1148, 1134, 1149, 1133, 1150, 1151, 
               1152, 1153, 1154, 1156, 1157, 1158, 1155, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 
               1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1185, 1183, 1184, 1186, 1187, 1188, 1206, 
               1207, 1208, 1205, 1209, 1210, 1212, 1213, 1214, 1211, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 
               1222, 1224, 1225, 1226, 1223, 1227, 1228, 2198, 3344, 3345, 1229, 1230, 1231, 2199, 1232, 1233, 
               1234, 1235, 2200, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1247, 1248, 1249, 
               1246, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 
               1265, 1266, 1267, 2201, 3384, 1268, 1269, 3387, 3388, 1270, 1271, 1272, 1273, 3393, 2202, 1274, 
               1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1699, 
               1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1698, 1713, 1726, 
               1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 3875, 3876, 1696, 1736, 1737, 1738, 3880, 
               3881, 177, 1739, 1740, 3884, 1697, 1741, 3894, 1749, 1750, 1751, 176, 1752, 1753, 2223, 2224, 2217, 
               2214, 3183, 3182, 3185, 3184, 3186, 3187, 3189, 3188, 3190, 3195, 3196, 3194, 3197, 3199, 3198, 3225, 
               3226, 3228, 3229, 3230, 3232, 3231, 3265, 3254, 3266, 3253, 3267, 3269, 3268, 3271, 3270, 3273, 3272, 
               3275, 3274, 3277, 3276, 3278, 3280, 3279, 3281, 3283, 3282, 3284, 3285, 3286, 3289, 3288, 3287, 3291, 
               3290, 3293, 3292, 3299, 3298, 3301, 3300, 3302, 3304, 3303, 3305, 3321, 3320, 3323, 3322, 3324, 3325, 
               3327, 3326, 3329, 3328, 3330, 3331, 3332, 3335, 3334, 3333, 3336, 3337, 3339, 3338, 3341, 3340, 3342, 
               3343, 3347, 3346, 3349, 3348, 3209, 3350, 3351, 3208, 3352, 3353, 3354, 3355, 3356, 3359, 3358, 3357, 
               3361, 3360, 3363, 3362, 3365, 3364, 3366, 3368, 3367, 3370, 3369, 3371, 3372, 3373, 3374, 3375, 3376, 
               3377, 3378, 3379, 3380, 3381, 3382, 3383, 3385, 3386, 3389, 3392, 3391, 3390, 3394, 3395, 3396, 3399, 
               3398, 3397, 3402, 3401, 3400, 3403, 3405, 3404, 3407, 3406, 3834, 3837, 3836, 3835, 3839, 3838, 3840, 
               3841, 3842, 3843, 3844, 3845, 3846, 3847, 3833, 3848, 3866, 3865, 3868, 3867, 3870, 3869, 3872, 3871, 
               3874, 3873, 3831, 3879, 3878, 3877, 3882, 3883, 3832, 3885, 3893, 3895, 3896, 3897]

def quaternion2matrix(q):
     m = np.identity(3)
     q00 = q[0] * q[0]
     q11 = q[1] * q[1]
     q22 = q[2] * q[2]
     q33 = q[3] * q[3]
     q03 = q[0] * q[3]
     q13 = q[1] * q[3]
     q23 = q[2] * q[3]
     q02 = q[0] * q[2]
     q12 = q[1] * q[2]
     q01 = q[0] * q[1]
     m[0][0] = q00 + q11 - q22 - q33
     m[1][1] = q00 - q11 + q22 - q33
     m[2][2] = q00 - q11 - q22 + q33
     m[0][1] = 2.0*(q12 - q03)
     m[1][0] = 2.0*(q12 + q03)
     m[0][2] = 2.0*(q13 + q02)
     m[2][0] = 2.0*(q13 - q02)
     m[1][2] = 2.0*(q23 - q01)
     m[2][1] = 2.0*(q23 + q01)
     
     return m


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


'''
    The class to manipulate 3D .obj mesh
'''
class MyMesh():
    
    def __init__(self, filename):
        self.filename = filename
        
        
    def LoadMesh(self):
        print "Loading ", self.filename
        
        f = open(self.filename, 'rb') 
        
        idx_v = 0
        idx_n = 0
        idx_uv = 0
        idx_f = 0
        for line in f:
            words = line.split()
            if (len(words) < 2):
                continue
            
            if (words[0] == "#"):
                if (words[1] == "Vertices:"):
                    self.Vertices = np.zeros((int(words[2]), 6), dtype = np.float32)
                    self.Normales = np.zeros((int(words[2]), 3), dtype = np.float32)
                    self.Rotations = np.zeros((int(words[2]), 3, 3), dtype = np.float32)
                elif (words[1] == "Faces:"):
                    self.Faces = np.zeros((int(words[2]), 6), dtype = np.int32)
                elif (words[1] == "Uvs:"):
                    self.UV = np.zeros((int(words[2]), 2), dtype = np.float32)
                continue
            
            if (words[0] == "v"):
                self.Vertices[idx_v] = [float(words[1]), float(words[2]), float(words[3]), idx_v, 0, 0]
                idx_v+=1
            
            if (words[0] == "vn"):
                self.Normales[idx_n] = [float(words[1]), float(words[2]), float(words[3])]
                self.Normales[idx_n] = self.Normales[idx_n]/LA.norm(self.Normales[idx_n])
                idx_n+=1
            
            if (words[0] == "vt"):
                self.UV[idx_uv] = [words[1], words[2]]
                idx_uv+=1
                
            if (words[0] == "f"):
                self.Faces[idx_f] = np.array([words[1].split("/"), words[2].split("/"), words[3].split("/")]).reshape(6)
                self.Faces[idx_f] = self.Faces[idx_f] - 1
                
                self.Vertices[self.Faces[idx_f,0],4:6] = self.UV[self.Faces[idx_f,1],:]
                self.Vertices[self.Faces[idx_f,2],4:6] = self.UV[self.Faces[idx_f,3],:]
                self.Vertices[self.Faces[idx_f,4],4:6] = self.UV[self.Faces[idx_f,5],:]
                
                idx_f+=1

        f.close()
        
        # set back point values to true
        self.IsBack = np.zeros(self.Vertices.shape[0], dtype = np.bool)
        self.IsBack[BackIndices] = True
    
        
    '''
        Function to draw the mesh using tkinter
    '''
    def DrawMesh(self, Pose, intrinsic, Size, canvas):
        
        #Draw all faces
        nb_faces, _ = self.Faces.shape
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        for i in range(nb_faces):
            inviewingvolume = False
            poly = []
            nmle = np.zeros(3, np.float32)
            for k in range(3):
                nmle[0] += self.Normales[self.Faces[i,2*k],0]
                nmle[1] += self.Normales[self.Faces[i,2*k],1]
                nmle[2] += self.Normales[self.Faces[i,2*k],2]
                nmle = np.dot(Pose[0:3,0:3], nmle)
                pt[0] = self.Vertices[self.Faces[i,2*k],0]
                pt[1] = self.Vertices[self.Faces[i,2*k],1]
                pt[2] = self.Vertices[self.Faces[i,2*k],2]
                pt = np.dot(Pose, pt)
                pix[0] = pt[0]/pt[2]
                pix[1] = pt[1]/pt[2]
                pix = np.dot(intrinsic, pix)
                column_index = int(round(pix[0]))
                line_index = int(round(pix[1]))
                poly.append((column_index, line_index))
                    
                if (column_index > -1 and column_index < Size[1] and line_index > -1 and line_index < Size[0] and pt[2] > 0. and nmle[2] < 0.):
                    inviewingvolume = True
            
            nmle = (nmle/LA.norm(nmle) + 1.0)*126.0
            r = str(hex(int(nmle[0])))[2::] 
            if (len(r) == 1):
                r = '0'+r
            g = str(hex(int(nmle[1])))[2::] 
            if (len(g) == 1):
                g = '0'+g
            b = str(hex(int(nmle[2])))[2::] 
            if (len(b) == 1):
                b = '0'+ b
            arg = '#' + r + g + b
            if inviewingvolume:
                canvas.create_polygon(*poly, fill=arg)
                
                
    def ComputeTgtPlane(self):
        
        for face in self.Faces:
        #for pt1, nmle1 in zip(self.Vertices, self.Normales):
            for k in range(3):
                pt_idx = face[2*k]
                if (not LA.norm(self.Rotations[pt_idx]) == 0.0):
                    k += 1
                    continue
                
                # Compute original and transformed tangent basis
                pt1 = self.Vertices[pt_idx][0:3]
                e3 = self.Normales[pt_idx]
                
                # Compute original basis, centered on pt1, with z nmle and oriented in p1->p2
                pt2 = self.Vertices[face[2*((k+1)%3)]][0:3]
                e1 = pt2-pt1
                proj = np.dot(e3, e1)
                e1 = e1 - proj*e3
                if (not LA.norm(e1) == 0.):
                    e1 = e1/LA.norm(e1)
            
                e2 = np.cross(e3, e1)
                if (not LA.norm(e2) == 0.):
                    e2 = e2/LA.norm(e2)
            
                # Compute transformed basis, centered on pt1', with z nmle and oriented in p1'->p2'
                Tpt1 = self.T_Vertices[pt_idx][0:3]
                Tpt2 = self.T_Vertices[face[2*((k+1)%3)]][0:3]
                Te3 = self.T_Normales[pt_idx]
                Te1 = Tpt2 - Tpt1
                proj = np.dot(Te1, Te3)
                Te1 = Te1 - proj*Te3
                if (not LA.norm(Te1) == 0.):
                    Te1 = Te1/LA.norm(Te1)
                
                Te2 = np.cross(Te3, Te1)
                if (not LA.norm(Te2) == 0.):
                    Te2 = Te2/LA.norm(Te2)
                
                k += 1
                
                # Compute Rotation matrix
                B1 = np.array([e1.T, e2.T, e3.T])
                TB1 = np.array([Te1.T, Te2.T, Te3.T])
                
                Rot = np.dot(TB1, LA.inv(B1))
                
                self.Rotations[pt_idx] = Rot
                
    def GetWeightedNormal(self, face):
        v1 = self.Vertices[face[2]][0:3] - self.Vertices[face[0]][0:3]
        v2 = self.Vertices[face[4]][0:3] - self.Vertices[face[0]][0:3]
        
        b = v1 / LA.norm(v1)
        
        proj = np.dot(b,v2)
        
        h = v2 - proj*b
        hauteur = sqrt(np.dot(h,h))
        area = LA.norm(v1)*hauteur/2.
        
        tmp = np.cross(v1,v2)
        
        res = (tmp/LA.norm(tmp))*area
        
        return res
        

'''
    The main class to manipulate blendshapes
'''
class BSMng():
    
    ''' Constructor '''
    def __init__(self, GPUManager, path, D2RGB):
        self.path = path
        self.nbBS = 28
        self.BlendShapes = []
        self.Size = (240,240,3)
        self.SizeDraw = (480, 640, 3)
        self.GPUManager = GPUManager
        
        img0 = cv2.imread("../data/Weights-240.png", cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread("../data/Labels-240.png", cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread("../data/FrontFace.png", cv2.IMREAD_UNCHANGED)
        img3 = cv2.imread("../data/Labelsb-240.png", cv2.IMREAD_UNCHANGED)
        
        self.Bump = np.zeros(self.Size, dtype = np.float32)
        self.Bump[:,:,2] = np.ascontiguousarray(img1[:,:,0].astype(np.float32) -1.0)
        self.Bump[img3[:,:,2] > 100,1] = -1.0
        self.labels = np.zeros((self.Size[0], self.Size[1]), dtype = np.int8)
        self.labels = np.ascontiguousarray(img2[:,:,2].astype(np.float32)>100).astype(np.int8)
        self.RGB = np.zeros(self.Size, dtype = np.float32)
        self.WeightMap = np.ascontiguousarray(img0.astype(np.float32) /65535.0)
        self.Vertices = np.zeros((28*4325,3), dtype = np.float32)
        self.Normales = np.zeros((28*4325,3), dtype = np.float32)
        self.BlendshapeCoeff = np.zeros(28, dtype = np.float32)
        
        #self.BlendshapeCoeff[21] = 1.0
        
        
        self.Bump_Mapping = ElementwiseKernel(self.GPUManager.context, 
                                               """float *Bump, float *RGBMapBump, float *BumpSwap, float *RGBMapBumpSwap, 
                                               float *VMapBump, float *NMapBump, float *VerticesBS, float *BlendshapeCoeff, 
                                               float *VMap, float *NMap, unsigned char *RGBMap, float *Pose, 
                                               float *Pose_D2RGB, float *calib_depth, float *calib_rgb, int n_bump, 
                                               int m_bump, int n_rgbd, int m_rgbd""",
                                               KernelsOpenCL.Kernel_Bump,
                                               "Bump_Mapping")
        
        #self.MedianFilter = ElementwiseKernel(self.GPUManager.context, 
        #                                       "float *depth, float *depth_out, int d, float sigma_r, float sigma_d, int nbLines, int nbColumns",
        #                                       KernelsOpenCL.Kernel_MedianFilter,
        #                                       "MedianFilter")
        
        self.vmap_to_nmap = ElementwiseKernel(self.GPUManager.context, 
                                               "float *vmap, float *nmap, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_NMap,
                                               "vmap_to_nmap")
        
        self.DataProc = ElementwiseKernel(self.GPUManager.context, 
                                               "float *Bump, float *WeightMap, float *Vertices, float *Normales, int *Triangles, float *VerticesBS, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_DataProc,
                                               "DataProc")
        
        self.draw_vmap = ElementwiseKernel(self.GPUManager.context, 
                                               "float *vmap, float *nmap, float *color, unsigned int *res, float *Pose, float *Intrinsic, int color_flag, int nbLines, int nbColumns",
                                               KernelsOpenCL.Kernel_Draw,
                                               "draw_vmap")
        
        # Allocate on CPU 
        self.draw_result = np.zeros(self.SizeDraw, dtype = np.uint32)
        
        self.BumpImage_d = cl.array.to_device(self.GPUManager.queue, self.Bump)
        self.WeightMap_d = cl.array.to_device(self.GPUManager.queue, self.WeightMap)
        self.labels_d = cl.array.to_device(self.GPUManager.queue, self.labels)
        
        self.VMapBump_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.NMapBump_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.RGBBump_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.RGBSwap_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.BumpSwap_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.float32)
        self.VerticesBS_d = cl.array.zeros(self.GPUManager.queue, (28*self.Size[0], self.Size[1], 6), np.float32)
        
        self.intrinsic_d = cl.array.zeros(self.GPUManager.queue, 4, np.float32)
        self.intrinsic_RGB_d = cl.array.zeros(self.GPUManager.queue, 4, np.float32)
        self.Pose_d = cl.array.zeros(self.GPUManager.queue, (4,4), np.float32)
        self.Pose_D2RGB_d = cl.array.zeros(self.GPUManager.queue, (4,4), np.float32)
        self.draw_d = cl.array.zeros(self.GPUManager.queue, self.SizeDraw, dtype = np.uint32)
        self.BlendshapeCoeff_d = cl.array.zeros(self.GPUManager.queue, 28, np.float32)
               
        self.Pose_D2RGB_d.set(D2RGB.astype(np.float32))
        
        
    def LoadBS(self):
        
        currMesh = MyMesh(self.path + "Neutralm.obj")
        currMesh.LoadMesh()
        self.BlendShapes.append(currMesh)
        
        for i in range(48):
            if ((i > 1 and i < 14) or i == 17 or i == 18 or i == 19 or i == 22 or i == 26 or i == 27 or i == 34 or i == 35 or i == 42):
                continue
            if (i == 0 or i == 1):
                currMesh = MyMesh(self.path + "0" + str(i) + "m.obj")
            else:
                currMesh = MyMesh(self.path + str(i) + "m.obj")
            currMesh.LoadMesh()
            self.BlendShapes.append(currMesh)
            
    ''' Rescale all blendshapes to match user's landmarks '''
    def Rescale(self, rgbd):
        rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
        
        ''' Compute average factor in X length from outer corner of eyes ''' 
        idx = rgb2depth[rgbd.shape[19][1], rgbd.shape[19][0]]
        if (idx == 0):
            print "No landmark for the left eye"
            return False
        
        left_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        
        idx = rgb2depth[rgbd.shape[28][1], rgbd.shape[28][0]]
        if (idx == 0):
            print "No landmark for the right eye"
            return False
        
        right_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        eye_dist = LA.norm(np.array(left_eye - right_eye))
        eye_dist_mesh = LA.norm(np.array(self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[19],0:3] - 
                                         self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[28],0:3]))     
        fact = eye_dist/eye_dist_mesh
        
        ''' Compute average factor in X length from inner corner of eyes '''
        idx = rgb2depth[rgbd.shape[22][1], rgbd.shape[22][0]]
        if (idx == 0):
            print "No landmark for the inner left eye"
            return False
        
        left_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        
        idx = rgb2depth[rgbd.shape[25][1], rgbd.shape[25][0]]
        if (idx == 0):
            print "No landmark for the inner right eye"
            return False
        
        right_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        eye_dist = LA.norm(np.array(left_eye - right_eye))
        eye_dist_mesh = LA.norm(np.array(self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[22],0:3] - 
                                         self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[25],0:3]))     
        fact = fact + eye_dist/eye_dist_mesh
       
        ''' Compute average factor in X length from mouth '''
        idx = rgb2depth[rgbd.shape[31][1], rgbd.shape[31][0]]
        if (idx == 0):
            print "No landmark for the lrft mouth"
            return False
        
        left_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        
        idx = rgb2depth[rgbd.shape[37][1], rgbd.shape[37][0]]
        if (idx == 0):
            print "No landmark for the right mouth"
            return False
        
        right_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        eye_dist = LA.norm(np.array(left_eye - right_eye))
        eye_dist_mesh = LA.norm(np.array(self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[31],0:3] - 
                                         self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[37],0:3]))     
        fact = fact + eye_dist/eye_dist_mesh
       
        ''' Compute average factor in Y length from nose '''
        idx = rgb2depth[rgbd.shape[10][1], rgbd.shape[10][0]]
        if (idx == 0):
            print "No landmark for the up nose"
            return False
        
        left_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        
        idx = rgb2depth[rgbd.shape[16][1], rgbd.shape[16][0]]
        if (idx == 0):
            print "No landmark for the down nose"
            return False
        
        right_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        
        eye_dist = LA.norm(np.array(left_eye - right_eye))
        eye_dist_mesh = LA.norm(np.array(self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[10],0:3] - 
                                         self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[16],0:3]))     
        fact = fact + eye_dist/eye_dist_mesh
        
        fact = fact/4.
        print "Scale factor: ", fact
        
        for mesh in self.BlendShapes:
            mesh.Vertices[:,0:3] = mesh.Vertices[:,0:3]*fact
            
        return True
    
    def RefineAlignment(self, rgbd):
        rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
        res = np.identity(4)
        delta_transfo = np.identity(4)
        
        Jac = np.zeros((3*51,6), dtype = np.float32)
        Jac_B = np.zeros((3*51,1), dtype = np.float32)
        
        landmark_mesh = np.ones(4, dtype = np.float32)
#        landmark_mesh_nmle = np.ones(3, dtype = np.float32)
        
        max_iter = 10
        for i in range(max_iter):
            b = np.zeros(6, np.float32)
            A = np.zeros((6,6), np.float32)
            
            # Build the jacobian matrix
            for k in range(51):
                idx = rgb2depth[rgbd.shape[k][1], rgbd.shape[k][0]]
                if (idx == 0):
                    print "No landmark"
                    return False
                landmark = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
                landmark_mesh[0:3] = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[k],0:3]
                landmark_mesh = np.dot(res, landmark_mesh)
                
                Jac[3*k][:] = [1.0, 0.0, 0.0, 0.0, 2.0*landmark_mesh[2], -2.0*landmark_mesh[1]]
                Jac[3*k+1][:] = [0.0, 1.0, 0.0, -2.0*landmark_mesh[2], 0.0, 2.0*landmark_mesh[0]]
                Jac[3*k+2][:] = [0.0, 0.0, 1.0, 2.0*landmark_mesh[1], -2.0*landmark_mesh[0], 0.0]
                
                Jac_B[3*k][0] = landmark[0] - landmark_mesh[0]
                Jac_B[3*k+1][0] = landmark[1] - landmark_mesh[1]
                Jac_B[3*k+2][0] = landmark[2] - landmark_mesh[2]
                
#                landmark_mesh_nmle[0:3] = self.BlendShapes[0].Normales[FACIAL_LANDMARKS[k],0:3]
#                landmark_mesh_nmle = np.dot(res[0:3,0:3], landmark_mesh_nmle)
                
#                Jac[k][:] = [landmark_mesh_nmle[0], landmark_mesh_nmle[1], landmark_mesh_nmle[2],
#                    -landmark[2]*landmark_mesh_nmle[1] + landmark[1]*landmark_mesh_nmle[2],
#                    landmark[2]*landmark_mesh_nmle[0] - landmark[0]*landmark_mesh_nmle[2],
#                    -landmark[1]*landmark_mesh_nmle[0] + landmark[0]*landmark_mesh_nmle[1]]
#                
#                Jac_B[k][0] = landmark_mesh_nmle[0]*(landmark[0] - landmark_mesh[0]) + landmark_mesh_nmle[1]*(landmark[1] - landmark_mesh[1]) + landmark_mesh_nmle[2]*(landmark[2] - landmark_mesh[2])
#                
            
            A = np.dot(Jac.transpose(), Jac)
            b = np.dot(Jac.transpose(), Jac_B).reshape(6)
            
            det = LA.det(A)
            if (det < 1.0e-10):
                print "determinant null"
                return False
       
            #delta_qsi = -LA.tensorsolve(A, b)
            #delta_transfo = LA.inv(Exponential(delta_qsi))
            
            delta_qsi = LA.tensorsolve(A, b)
            norm = (delta_qsi[3]*delta_qsi[3] + delta_qsi[4]*delta_qsi[4] + delta_qsi[5]*delta_qsi[5])
            if (norm > 1.0) :
				delta_qsi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
			
            quat = [sqrt(1.0 - norm), delta_qsi[3], delta_qsi[4], delta_qsi[5]]

            delta_transfo[0:3,0:3] = quaternion2matrix(quat);
            delta_transfo[0:3,3] = delta_qsi[0:3]
            
            res = np.dot(delta_transfo, res)
        
        for mesh in self.BlendShapes:
            mesh.Normales[:,0:3] = np.dot(mesh.Normales[:,0:3], res[0:3,0:3].T)
            mesh.Vertices[:,0:3] = np.dot(mesh.Vertices[:,0:3], res[0:3,0:3].T)
            mesh.Vertices[:,0:3] = mesh.Vertices[:,0:3] + res[0:3,3]
            
        print res
        return True
    
    def AlignToFace(self, rgbd):
        rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
        
        idx = rgb2depth[rgbd.shape[13][1], rgbd.shape[13][0]]
        if (idx == 0):
            print "No landmark for the nose"
            return False
        
        nose = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        nose_mesh = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[13],0:3]
        
        idx = rgb2depth[rgbd.shape[19][1], rgbd.shape[19][0]]
        if (idx == 0):
            print "No landmark for the left eye"
            return False
        
        left_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        left_eye_mesh = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[19],0:3]
        
        
        idx = rgb2depth[rgbd.shape[28][1], rgbd.shape[28][0]]
        if (idx == 0):
            print "No landmark for the right eye"
            return False
        
        right_eye = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        right_eye_mesh = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[28],0:3]
        
        v1 = np.cross(left_eye - nose, right_eye - nose)
        v1 = v1/LA.norm(v1)
        
        v2 = np.cross(left_eye_mesh - nose_mesh, right_eye_mesh - nose_mesh)
        v2 = v2/LA.norm(v2)
        
        rotation_axis = np.cross(v1,v2)
        rotation_axis = rotation_axis/LA.norm(rotation_axis)
        rotation_angle = -acos(np.dot(v1,v2))
        q = [cos(rotation_angle/2.), rotation_axis[0]*sin(rotation_angle/2.), rotation_axis[1]*sin(rotation_angle/2.), rotation_axis[2]*sin(rotation_angle/2.)]
        
        s = 1./(LA.norm(q)**2)
        
        Rotation = np.array([[1.-2.*s*(q[2]**2 + q[3]**2), 2.*s*(q[1]*q[2] - q[3]*q[0]), 2.*s*(q[1]*q[3] + q[2]*q[0])],
                     [2.*s*(q[1]*q[2] + q[3]*q[0]), 1.-2.*s*(q[1]**2 + q[3]**2), 2.*s*(q[2]*q[3] - q[1]*q[0])],
                     [2.*s*(q[1]*q[3]-q[2]*q[0]), 2.*s*(q[2]*q[3] + q[1]*q[0]), 1.-2.*s*(q[1]**2 + q[2]**2)]])
    
        
        for mesh in self.BlendShapes:
            mesh.Normales[:,0:3] = np.dot(mesh.Normales[:,0:3], Rotation.T)
            mesh.Vertices[:,0:3] = np.dot(mesh.Vertices[:,0:3], Rotation.T)
            
        
        idx = rgb2depth[rgbd.shape[13][1], rgbd.shape[13][0]]
        nose = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1]]
        nose_mesh = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[13],0:3]
        
        translation = nose-nose_mesh
        
        for mesh in self.BlendShapes:
            mesh.Vertices[:,0] = mesh.Vertices[:,0] + translation[0]
            mesh.Vertices[:,1] = mesh.Vertices[:,1] + translation[1]
            mesh.Vertices[:,2] = mesh.Vertices[:,2] + translation[2]
            
        return self.RefineAlignment(rgbd)
        
    
    def DrawLandmarks(self, rgbd, Pose, intrinsic, canvas, idx_L):
        
        rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
        
        landmark = np.ones(4, dtype = np.float32)
        landmark_mesh = np.ones(4, dtype = np.float32)
        pix = np.array([0., 0., 1.])
        
        for k in range(51):        #range(idx_L,idx_L+1):
            idx = rgb2depth[rgbd.shape[k][1], rgbd.shape[k][0]]
            if (idx == 0):
                #print "No landmark"
                continue
            landmark[0:3] = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1],:]
            #landmark = np.dot(Pose, landmark)  
            landmark_mesh[0:3] = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[k],0:3]
            
            for bs_idx in range(1,28):
                landmark_mesh[0:3] = landmark_mesh[0:3] + (self.BlendShapes[bs_idx].Vertices[FACIAL_LANDMARKS[k],0:3]-self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[k],0:3])*self.BlendshapeCoeff[bs_idx]
            
            landmark_mesh = np.dot(Pose, landmark_mesh)
            
            pix[0] = landmark[0]/landmark[2]
            pix[1] = landmark[1]/landmark[2]
            pix = np.dot(intrinsic, pix)
            column_index = int(round(pix[0]))
            line_index = int(round(pix[1]))
            
            poly = [(column_index-2, line_index-2), (column_index-2, line_index+2), (column_index+2, line_index+2), (column_index+2, line_index+2)]
            canvas.create_polygon(*poly, fill='red')
                
            #print k, landmark_mesh
            pix[0] = landmark_mesh[0]/landmark_mesh[2]
            pix[1] = landmark_mesh[1]/landmark_mesh[2]
            pix = np.dot(intrinsic, pix)
            column_index = int(round(pix[0]))
            line_index = int(round(pix[1]))
            
            poly = [(column_index-2, line_index-2), (column_index-2, line_index+2), (column_index+2, line_index+2), (column_index+2, line_index+2)]
            canvas.create_polygon(*poly, fill='green')
        
    def ChangeLandmark(self, idx, inc):
        FACIAL_LANDMARKS[idx] = FACIAL_LANDMARKS[idx] + inc
        print FACIAL_LANDMARKS[idx]
        
    def LoadAffineTransfo(self):
        data = np.load("AffineTransfo.npz")
        self.MatList1 = data["MatList1"]
        self.MatList2 = data["MatList2"]
        
        
    def ComputeAffineTransfo(self):
        '''Inititialisation'''
        RefMesh = self.BlendShapes[0]
        
        self.TransfoExpression = np.zeros((RefMesh.Faces.shape[0], len(self.BlendShapes)-1, 3, 3))
            
        print RefMesh.Faces.shape[0]
        
        So = np.zeros((3,3))
        Si = np.zeros((3,3))
        idxFace = 0
        for face in RefMesh.Faces:
            # Compute normal and weight of the face
            nmle = RefMesh.GetWeightedNormal(face)
            
            # Compute tetrahedron for b0
            summit4 = (RefMesh.Vertices[face[0]] + RefMesh.Vertices[face[2]]+ RefMesh.Vertices[face[4]])[0:3]/3. + nmle
            
            So[0:3,0] = RefMesh.Vertices[face[2]][0:3] - RefMesh.Vertices[face[0]][0:3]
            So[0:3,1] = RefMesh.Vertices[face[4]][0:3] - RefMesh.Vertices[face[0]][0:3]
            So[0:3,2] = summit4 - RefMesh.Vertices[face[0]][0:3]
            
            # Go through all other blendshapes
            idxMesh = 0
            for mesh in self.BlendShapes[1::]:
                # Compute normal and weight of the face
                nmlei = mesh.GetWeightedNormal(face)
            
                # Compute tetrahedron for b0
                summit4 = (mesh.Vertices[face[0]] + mesh.Vertices[face[2]]+ mesh.Vertices[face[4]])[0:3]/3. + nmlei
                
                Si[0:3,0] = mesh.Vertices[face[2]][0:3] - mesh.Vertices[face[0]][0:3]
                Si[0:3,1] = mesh.Vertices[face[4]][0:3] - mesh.Vertices[face[0]][0:3]
                Si[0:3,2] = summit4 - mesh.Vertices[face[0]][0:3]
                
                self.TransfoExpression[idxFace,idxMesh] = np.dot(Si, LA.inv(So))
                
                idxMesh += 1
                
            idxFace += 1
        
        print "TransfoExpression computed"
            
        # Compute Matrix F that fix points on the backface.
        F = sp.lil_matrix((3*RefMesh.Vertices.shape[0], 3*RefMesh.Vertices.shape[0]))
        for ind in range(RefMesh.Vertices.shape[0]):
            if (RefMesh.IsBack[ind]):
                F[3*ind:3*ind + 3, 3*ind:3*ind + 3] = np.identity(3)
         
        # Compute Matrix G that transform vertices to edges
        G = sp.lil_matrix((6*RefMesh.Faces.shape[0], 3*RefMesh.Vertices.shape[0]))
        indx = 0
        for face in RefMesh.Faces:
            G[6 * indx:6 * indx+3, 3*face[0]:3*face[0]+3] = -np.identity(3)
            G[6 * indx+3:6 * indx+6, 3*face[0]:3*face[0]+3] = -np.identity(3)
            
            G[6 * indx:6 * indx+3, 3*face[2]:3*face[2]+3] = np.identity(3)
            G[6 * indx+3:6 * indx+6, 3*face[4]:3*face[4]+3] = np.identity(3)
            
            indx += 1
            
        print "Matrix G computed"
        
        nu = 100.0
        self.MatList1 = []
        self.MatList2 = []
        for indx_mesh in range(len(self.BlendShapes)-1):
            print indx_mesh
            # Compute the affine transfo matrix
            H = sp.lil_matrix((6*RefMesh.Faces.shape[0], 6*RefMesh.Faces.shape[0]))
            indx = 0
            for face in RefMesh.Faces:
                H[6 * indx:6 * indx+3, 6 * indx:6 * indx+3] = self.TransfoExpression[indx, indx_mesh]
                H[6 * indx+3:6 * indx+6, 6 * indx+3:6 * indx+6] = self.TransfoExpression[indx, indx_mesh]
                indx += 1
                
            print "Matrix H computed"
                
            MatTmp = G.T * G + nu*F
            MatT = (G.T * H) * G + nu*F
            self.MatList1.append(MatTmp)
            self.MatList2.append(MatT)
            np.savez_compressed("../data/AffineTransfo", MatList1 = self.MatList1, MatList2 = self.MatList2)
        
        print "MatList1/2 computed"
        
        
    def ElasticRegistration(self, rgbd):
        self.BlendShapes[0].T_Vertices = self.BlendShapes[0].Vertices.copy()
        self.BlendShapes[0].T_Normales = self.BlendShapes[0].Normales.copy()
        
        rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
                                      
        nb_col = 3*self.BlendShapes[0].Vertices.shape[0]
        nb_lines = 3*51 + 3*6*self.BlendShapes[0].Faces.shape[0]
            
        A = sp.lil_matrix((nb_lines, nb_col), dtype = np.float32)
        b1 = np.zeros(nb_lines, dtype = np.float32)
        
        '''/****Compute local tangent plane transforms*****/'''
        self.BlendShapes[0].ComputeTgtPlane()
        
        for iter in range(3):
            '''/****Point correspondences***/
    
        		/****Solve linear system*****/
                Build Matrix A
                Nx -Ny -Nz 0 ......... 0]
                [0 0 0 -Nx -Ny -Nz 0 ... 0]
                ...
                [0............ -Nx -Ny -Nz]
                [0..0...1....-1 .......]
                [0..0...0 1...0 -1 ....]
                [0..0...0 0 1.0 0 -1...]*/'''
            
            '''/*****************************Add landmarks******************************************/'''
            for i in range(51):
                A[3*i:3*i+3, 3*FACIAL_LANDMARKS[i]:3*FACIAL_LANDMARKS[i]+3] = 100.0*np.identity(3)
                
                idx = rgb2depth[rgbd.shape[i][1], rgbd.shape[i][0]]
                b1[3*i:3*i+3] = 100.0*vmap[idx/rgbd.Size[1], idx%rgbd.Size[1],:]
        
            
            '''/***************Populate matrix from neighboors of the vertices***************************/'''
            idx = 51
            for face in self.BlendShapes[0].Faces:
                k = 0
                for k in range(3):
                    v_idx = face[2*k]
                    A[3*idx:3*idx+3, 3*v_idx:3*v_idx+3] = -np.identity(3)
                    A[3*idx:3*idx+3, 3*face[2*((k+1)%3)]:3*face[2*((k+1)%3)]+3] = np.identity(3)
                    pt = self.BlendShapes[0].T_Vertices[face[2*((k+1)%3)]][0:3] - self.BlendShapes[0].T_Vertices[v_idx][0:3]
                    #Rotate with the quaternion
                    Rpt = np.dot(self.BlendShapes[0].Rotations[v_idx], pt)
                    b1[3*idx] = Rpt[0]
                    b1[3*idx+1] = Rpt[1]
                    b1[3*idx+2] = Rpt[2]
                    idx += 1
                    
                    A[3*idx:3*idx+3, 3*v_idx:3*v_idx+3] = -np.identity(3)
                    A[3*idx:3*idx+3, 3*face[2*((k+2)%3)]:3*face[2*((k+2)%3)]+3] = np.identity(3)
                    pt = self.BlendShapes[0].T_Vertices[face[2*((k+2)%3)]][0:3] - self.BlendShapes[0].T_Vertices[v_idx][0:3]
                    #Rotate with the quaternion
                    Rpt = np.dot(self.BlendShapes[0].Rotations[v_idx], pt)
                    b1[3*idx] = Rpt[0]
                    b1[3*idx+1] = Rpt[1]
                    b1[3*idx+2] = Rpt[2]
                    idx += 1
                    
                    k += 1
            
            #MatA = np.dot(A.T,A)
            #b = np.dot(A.T, b1)
            #xres = LA.tensorsolve(MatA, b)
            xres = spsolve(A.T*A, A.T*b1)
        
            print xres[0:3]
            print self.BlendShapes[0].Vertices[0]
            
            ''' Affect result to mesh '''
            idx = 0
            for (vtx,nmle) in zip(self.BlendShapes[0].T_Vertices, self.BlendShapes[0].T_Normales):
                vtx[0] = xres[3*idx]
                vtx[1] = xres[3*idx+1]
                vtx[2] = xres[3*idx+2]
                nmle[0] = 0.
                nmle[1] = 0.
                nmle[2] = 0.
                idx += 1
                
            '''/***************Compute normals*******************/'''
            
            for face in self.BlendShapes[0].Faces:
                # Compute normal and weight of the face
                v1 = self.BlendShapes[0].T_Vertices[face[2]] - self.BlendShapes[0].T_Vertices[face[0]]
                v2 = self.BlendShapes[0].T_Vertices[face[4]] - self.BlendShapes[0].T_Vertices[face[0]]
                nmle = np.cross(v1[0:3],v2[0:3])
                nmle = nmle / LA.norm(nmle)
                
                self.BlendShapes[0].T_Normales[face[0], 0] = self.BlendShapes[0].T_Normales[face[0], 0] + nmle[0]
                self.BlendShapes[0].T_Normales[face[0], 1] = self.BlendShapes[0].T_Normales[face[0], 1] + nmle[1]
                self.BlendShapes[0].T_Normales[face[0], 2] = self.BlendShapes[0].T_Normales[face[0], 2] + nmle[2]
                
                self.BlendShapes[0].T_Normales[face[2], 0] = self.BlendShapes[0].T_Normales[face[2], 0] + nmle[0]
                self.BlendShapes[0].T_Normales[face[2], 1] = self.BlendShapes[0].T_Normales[face[2], 1] + nmle[1]
                self.BlendShapes[0].T_Normales[face[2], 2] = self.BlendShapes[0].T_Normales[face[2], 2] + nmle[2]
                
                self.BlendShapes[0].T_Normales[face[4], 0] = self.BlendShapes[0].T_Normales[face[4], 0] + nmle[0]
                self.BlendShapes[0].T_Normales[face[4], 1] = self.BlendShapes[0].T_Normales[face[4], 1] + nmle[1]
                self.BlendShapes[0].T_Normales[face[4], 2] = self.BlendShapes[0].T_Normales[face[4], 2] + nmle[2]
                
            for nmle in self.BlendShapes[0].T_Normales:  
                if (LA.norm(nmle) > 0.):
                    nmle = nmle / LA.norm(nmle)
            
        self.BlendShapes[0].Vertices = self.BlendShapes[0].T_Vertices.copy()
        self.BlendShapes[0].Normales = self.BlendShapes[0].T_Normales.copy()
        
        '''/***********************Transfer expression deformation******************************/'''
        
        #boV = self.BlendShapes[0].Vertices.copy()
        boV = np.zeros(nb_col, dtype = np.float32)
        idx = 0
        for vtx in self.BlendShapes[0].Vertices:
            boV[idx] = vtx[0]
            boV[idx+1] = vtx[1]
            boV[idx+2] = vtx[2]
            idx += 3
        
        indx_mesh = 0
        for mesh in self.BlendShapes[1::]:
            
            #xres = LA.tensorsolve(self.MatList1[indx_mesh], np.dot(self.MatList2[indx_mesh], boV))
            xres = spsolve(self.MatList1[indx_mesh], self.MatList2[indx_mesh] * boV)
            indx_mesh += 1
            
            ''' Affect result to mesh '''
            idx = 0
            for (vtx,nmle) in zip(mesh.Vertices, mesh.Normales):
                vtx[0] = xres[3*idx]
                vtx[1] = xres[3*idx+1]
                vtx[2] = xres[3*idx+2]
                nmle[0] = 0.
                nmle[1] = 0.
                nmle[2] = 0.
                idx += 1
                
            '''/***************Compute normals*******************/'''
            for face in mesh.Faces:
                # Compute normal and weight of the face
                v1 = mesh.Vertices[face[2]] - mesh.Vertices[face[0]]
                v2 = mesh.Vertices[face[4]] - mesh.Vertices[face[0]]
                nmle = np.cross(v1[0:3],v2[0:3])
                nmle = nmle / LA.norm(nmle)
                
                mesh.Normales[face[0], 0] = mesh.Normales[face[0], 0] + nmle[0]
                mesh.Normales[face[0], 1] = mesh.Normales[face[0], 1] + nmle[1]
                mesh.Normales[face[0], 2] = mesh.Normales[face[0], 2] + nmle[2]
                
                mesh.Normales[face[2], 0] = mesh.Normales[face[2], 0] + nmle[0]
                mesh.Normales[face[2], 1] = mesh.Normales[face[2], 1] + nmle[1]
                mesh.Normales[face[2], 2] = mesh.Normales[face[2], 2] + nmle[2]
                
                mesh.Normales[face[4], 0] = mesh.Normales[face[4], 0] + nmle[0]
                mesh.Normales[face[4], 1] = mesh.Normales[face[4], 1] + nmle[1]
                mesh.Normales[face[4], 2] = mesh.Normales[face[4], 2] + nmle[2]
                
            for nmle in mesh.Normales:  
                if (LA.norm(nmle) > 0.):
                    nmle = nmle / LA.norm(nmle)
        

    ''' Generate / Update bump image '''
    def BumpImage(self, rgbd, Pose):
        
        self.Pose_d.set(Pose.astype(np.float32))
        intrinsic_depth = np.array([rgbd.intrinsic[0,0], rgbd.intrinsic[1,1], rgbd.intrinsic[0,2], rgbd.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_depth)
        intrinsic_rgb = np.array([rgbd.intrinsicRGB[0,0], rgbd.intrinsicRGB[1,1], rgbd.intrinsicRGB[0,2], rgbd.intrinsicRGB[1,2]])
        self.intrinsic_RGB_d = cl.array.to_device(self.GPUManager.queue, intrinsic_rgb)
        
        self.Bump_Mapping(self.BumpImage_d, self.RGBBump_d, self.BumpSwap_d, self.RGBSwap_d, self.VMapBump_d, 
                          self.NMapBump_d, self.VerticesBS_d, self.BlendshapeCoeff_d, 
                          rgbd.vmap_d, rgbd.nmap_d, rgbd.color_d, self.Pose_d, self.Pose_D2RGB_d, 
                          self.intrinsic_d, self.intrinsic_RGB_d, self.Size[0], self.Size[1], self.SizeDraw[0], self.SizeDraw[1])
        
        #self.MedianFilter(self.Bump_buffer, self.BumpImage)
        self.BumpImage_d = self.BumpSwap_d.copy()
        self.RGBBump_d = self.RGBSwap_d.copy()
        
        self.vmap_to_nmap(self.VMapBump_d, self.NMapBump_d, self.Size[0], self.Size[1])
        
    def PreProcessing(self):
        for i in range(28):
            self.Vertices[i*4325:(i+1)*4325, ::] = self.BlendShapes[i].Vertices[::,0:3]
            self.Normales[i*4325:(i+1)*4325, ::] = self.BlendShapes[i].Normales[::]
        
        self.Vertices_d = cl.array.to_device(self.GPUManager.queue, self.Vertices)
        self.Normales_d = cl.array.to_device(self.GPUManager.queue, self.Normales)
        self.Triangles_d = cl.array.to_device(self.GPUManager.queue, self.BlendShapes[0].Faces)
        
        self.DataProc(self.BumpImage_d, self.WeightMap_d, self.Vertices_d, self.Normales_d, self.Triangles_d, self.VerticesBS_d, self.Size[0], self.Size[1])
        
        
    def DrawBump(self, Pose, intrinsic, color = 0):
        print "Draw bump image"
        self.Pose_d.set(Pose.astype(np.float32))
        intrinsic_curr = np.array([intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        self.draw_d.set(self.draw_result)
        self.draw_vmap(self.VMapBump_d, self.NMapBump_d, self.RGBBump_d, self.draw_d, self.Pose_d, 
                       self.intrinsic_d, color, self.SizeDraw[0], self.SizeDraw[1])
        return self.draw_d.get()
    
    def SetLandmarks(self, rgbd):
        self.LandmarksBump = np.zeros((51,2), dtype = np.int32)
        for i in range(51):
            best_i = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[i], 4]*self.Size[0]
            best_j = self.BlendShapes[0].Vertices[FACIAL_LANDMARKS[i], 5]*self.Size[1]
            
            if (i == 22):
                best_i = 110
                best_j = 169
            if (i == 25):
                best_i = 130
                best_j = 169
                
            self.LandmarksBump[i] = (best_i, best_j)
            
        '''rgb2depth = rgbd.RGB2Depth.get()
        vmap = rgbd.vmap_d.get()
        # search for vertex index closest to the 44 landmark
        idx = rgb2depth[rgbd.shape[43][1], rgbd.shape[43][0]]
        if (idx == 0):
            print "No landmark"
            
        landmark = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1],:]
        min_dist = 10000.0
        best_idx = 0
        curr_i = 0
        for v in self.BlendShapes[0].Vertices:
            dist = LA.norm(v[0:3]-landmark)
            if (min_dist > dist):
                min_dist = dist
                best_idx = curr_i
            curr_i += 1
        print "index for landmark 43: ", best_idx 
        
        # search for vertex index closest to the 48 landmark
        idx = rgb2depth[rgbd.shape[47][1], rgbd.shape[47][0]]
        if (idx == 0):
            print "No landmark"
            
        landmark = vmap[idx/rgbd.Size[1], idx%rgbd.Size[1],:]
        min_dist = 10000.0
        best_idx = 0
        curr_i = 0
        for v in self.BlendShapes[0].Vertices:
            dist = LA.norm(v[0:3]-landmark)
            if (min_dist > dist):
                min_dist = dist
                best_idx = curr_i
            curr_i += 1
        print "index for landmark 47: ", best_idx'''
            
    def Save(self, dest):
        print "recording result at: ", dest
        
        vmap = self.VMapBump_d.get()
        nmap = self.NMapBump_d.get()
        rgb = self.RGBBump_d.get()
        bump = self.BumpImage_d.get()
        
        cv2.imwrite(dest+'/Bump_RGB.png',rgb.astype(np.uint8))
        cv2.imwrite(dest+'/Bump.png',50.0*(bump[:,:,0]+10.0)*(bump[:,:,1]>0))              
        
        # Write vertices
        Index = np.zeros((self.Size[0], self.Size[1]), dtype = np.int32)
        curr_ind = 0
        Vertices = []
        Normales = []
        Colors = []
        for i in range(self.Size[0]):
            for j in range(self.Size[1]):
                if (bump[i,j,1] != 0):
                    Index[i,j] = curr_ind
                    Vertices.append([vmap[i,j,0], vmap[i,j,1], vmap[i,j,2]])
                    Normales.append([nmap[i,j,0], nmap[i,j,1], nmap[i,j,2]])
                    Colors.append([int(rgb[i,j,0]), int(rgb[i,j,1]), int(rgb[i,j,2])])
                    curr_ind += 1
                
        # Write the faces
        faces = []
        for i in range(self.Size[0]-1):
            for j in range(self.Size[1]-1):
                # triangle 1
                if (bump[i,j,1] != 0.0 and bump[i+1,j,1] != 0.0 and bump[i,j+1,1] != 0.0):
                    faces.append([Index[i,j], Index[i+1,j], Index[i,j+1]])
                #triangle 2
                if (bump[i,j+1,1] != 0.0 and bump[i+1,j,1] != 0.0 and bump[i+1,j+1,1] != 0.0):
                    faces.append([Index[i,j+1], Index[i+1,j], Index[i+1,j+1]])
                   
        
        ''' Write result 3D mesh into a .ply file'''
        f = open(dest+"/Mesh.ply", 'wb')
        
        # Write headers
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ply file created by Diego Thomas\n")
        f.write("element vertex %d \n" %(len(Vertices)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face %d \n" %(len(faces)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        
        for (v,n,r) in zip(Vertices, Normales, Colors):
            f.write("%f %f %f %f %f %f %d %d %d\n" %(v[0], v[1], v[2], 
                                                n[0], n[1], n[2],
                                                r[0], r[1], r[2]))
            
        for face_curr in faces:
            f.write("3 %d %d %d \n" %(face_curr[0], face_curr[1], face_curr[2])) 
        
        f.close()
            