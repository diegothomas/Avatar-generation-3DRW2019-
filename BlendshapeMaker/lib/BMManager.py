#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import imp
import sys
import time
import os
from os import path
import Tkinter as tk
import numpy as np
from numpy import linalg as LA
from math import sqrt, acos, cos, sin
from scipy import optimize
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import concurrent.futures
from Queue import Queue
import copy


APP_ROOT = path.dirname( path.abspath( __file__ ) )


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
                    self.Normals = np.zeros((int(words[2]), 3), dtype = np.float32)
                    self.Rotations = np.zeros((int(words[2]), 3, 3), dtype = np.float32)
                elif (words[1] == "Faces:"):
                    self.Faces = np.zeros((int(words[2]), 7), dtype = np.int32)
                    #self.V4 = np.zeros((int(words[2]), 3), dtype = np.float32)
                elif (words[1] == "Uvs:"):
                    self.UV = np.zeros((int(words[2]), 2), dtype = np.float32)
                continue
            
            if (words[0] == "v"):
                self.Vertices[idx_v] = [float(words[1]), float(words[2]), float(words[3]), idx_v, 0, 0]
                idx_v+=1
            
            if (words[0] == "vn"):
                self.Normals[idx_n] = [float(words[1]), float(words[2]), float(words[3])]
                self.Normals[idx_n] = self.Normals[idx_n]/LA.norm(self.Normals[idx_n])
                idx_n+=1
            
            if (words[0] == "vt"):
                self.UV[idx_uv] = [words[1], words[2]]
                idx_uv+=1
                
            if (words[0] == "f"):
                self.Faces[idx_f] = np.hstack((np.array([words[1].split("/"), words[2].split("/"), words[3].split("/")]).reshape(6), 0))
                self.Faces[idx_f] = self.Faces[idx_f] - 1

                self.Vertices[self.Faces[idx_f,0],4:6] = self.UV[self.Faces[idx_f,1],:]
                self.Vertices[self.Faces[idx_f,2],4:6] = self.UV[self.Faces[idx_f,3],:]
                self.Vertices[self.Faces[idx_f,4],4:6] = self.UV[self.Faces[idx_f,5],:]
                
                idx_f+=1

        f.close()

        #compute v4 from v1,v2,v3
        #for k in range(idx_f):
        #    cross = np.cross(self.Vertices[self.Faces[k, 2],0:3] - self.Vertices[self.Faces[k, 0],0:3], self.Vertices[self.Faces[k, 4],0:3]-self.Vertices[self.Faces[k, 0],0:3])
        #    self.V4[k] = self.Vertices[self.Faces[k, 0],0:3] + cross / sqrt(np.linalg.norm(cross))
        

    

    def LoadAvatarMesh(self):
        print "Loading ", self.filename

        f = open(self.filename, 'rb') 
        idx_v = 0
        idx_f = 0
        num_v = 0
        num_f = 0

        self.TexName = []
        self.Flags = [False]

        for line in f:
            words = line.split()
            #read header
            if(words[0]=="format"):
                if(words[1]=="ascii"):
                    continue
                else:
                    print "cannot read binary format. please convert it to a non-binary format."
                    sys.exit()

            if(words[0]=="comment"):
                if(words[1]=="TextureFile"):
                    self.TexName += [words[2]]
                continue

            if(words[0]=="element"):
                if(words[1]=="vertex"):
                    num_v = int(words[2])
                    self.Vertices = np.zeros((int(words[2]), 4), dtype = np.float32)
                elif(words[1]=="face"):
                    num_f = int(words[2])
                    self.Faces = np.zeros((int(words[2]), 7), dtype = np.int32)
                    self.Textures = np.zeros((int(words[2]), 6), dtype = np.float32)
                continue

            if(words[0]=="property"):
                if(words[1]=="list" and words[2]=="uchar" and words[3]=="float" and words[4]=="texcoord"):
                    self.Flags[0] = True
                continue
            

            if(words[0]=="end_header"):
                break
            
        
        #read vertices
        for line in f:
            words = line.split()

            self.Vertices[idx_v] = [float(words[0]), float(words[1]), float(words[2]), idx_v]
            

            #test
            #print self.Vertices[idx_v]

            idx_v+=1

            if (idx_v >= num_v):
                break

        #read faces
        for line in f:
            words = line.split()
            if(self.Flags[0]):
                self.Faces[idx_f]=np.array([int(words[1]), 0, int(words[2]), 0,  int(words[3]), 0, int(words[11])])
                self.Textures[idx_f, :] = words[5:11]
            else:
                self.Faces[idx_f,0:6]=np.array([int(words[1]), 0, int(words[2]), 0,  int(words[3]), 0])
            #test
            #print "Face %d" %(idx_f), self.Faces[idx_f]

            idx_f+=1

            if(idx_f >= num_f):
                break

        f.close()
        
        #test
        #for i in range (900):
        #    print self.Vertices[i]

    
class Mesh_VFN():
    def __init__(self, Vertices, Faces):
        self.Vertices = Vertices
        self.Faces = Faces
        self.Normals = np.zeros((len(Faces), 3), dtype=np.float32)

        for i in range(0, len(Faces)):
            cross = np.cross(Vertices[Faces[i, 2],0:3] - Vertices[Faces[i, 0],0:3], Vertices[Faces[i, 4],0:3]-Vertices[Faces[i, 0],0:3])
            self.Normals[i] = cross / LA.norm(cross)


class dtf():
    def __init__(self, corrList, SOURCE_LANDMARKS, AVATAR_LANDMARKS, SourceExpression, SourceNeutral, AvatarNeutral, PartsSize, PartsRot, Ed_A, Ed_ATA, Es_ATA, Es_ATc, Ei_ATA, Ei_ATc, backlandmark_list):
        self.corrList = corrList
        self.SOURCE_LANDMARKS = SOURCE_LANDMARKS
        self.AVATAR_LANDMARKS = AVATAR_LANDMARKS
        self.SE_Vertices = SourceExpression.Vertices
        self.SN_Vertices = SourceNeutral.Vertices
        self.AN_Vertices = AvatarNeutral.Vertices
        self.Vertices = AvatarNeutral.Vertices
        self.SE_Faces = SourceExpression.Faces
        self.SN_Faces = SourceNeutral.Faces
        self.AN_Faces = AvatarNeutral.Faces
        self.Faces = AvatarNeutral.Faces
        self.PartsSize = PartsSize
        self.PartsRot = PartsRot
        self.Ed_A = Ed_A
        self.Ed_ATA = Ed_ATA
        self.Es_ATA = Es_ATA
        self.Es_ATc = Es_ATc
        self.Ei_ATA = Ei_ATA
        self.Ei_ATc = Ei_ATc
        self.backlandmark_list = backlandmark_list


    def deformation_transfer(self):
        start_dtf = time.time()
        SourceRotation = np.zeros((len(self.SE_Faces), 3, 3), dtype = np.float32)

        #compute Source triangle rotation
        start_time = time.time(); print "Compute Source rotation...",

        
        for i in range(0, len(self.SN_Faces)):
            a = np.array([self.SN_Vertices[self.SN_Faces[i,2],0:3] - self.SN_Vertices[self.SN_Faces[i,0],0:3]]) #v2-v1
            b = np.array([self.SN_Vertices[self.SN_Faces[i,4],0:3] - self.SN_Vertices[self.SN_Faces[i,0],0:3]]) #v3-v1
            cross = np.cross(a, b)
            c = cross / sqrt(LA.norm(cross)) #v4-v1

            Vinv = LA.solve(np.hstack([a.T, b.T, c.T]), np.identity(3)) #Source Neutral Vinv

            a = np.array([self.SE_Vertices[self.SE_Faces[i,2],0:3] - self.SE_Vertices[self.SE_Faces[i,0],0:3]]) #v2-v1
            b = np.array([self.SE_Vertices[self.SE_Faces[i,4],0:3] - self.SE_Vertices[self.SE_Faces[i,0],0:3]]) #v3-v1
            cross = np.cross(a, b)
            c = cross / sqrt(LA.norm(cross)) #v4-v1
            
            SourceRotation[i] = np.dot(np.hstack([a.T, b.T, c.T]), Vinv) # S = np.dot(SE_Vtil, SN_Vinv)

        elapsed_time = time.time() - start_time; print "done (%f sec)" % (elapsed_time)
        
        #Deformation Transfer term
        def MakeEd_ATc():
            Ed_cVector = sp.lil_matrix((1, len(self.corrList)*9), dtype=np.float32)            
            for i in range(len(self.corrList)):
                Ed_cVector[0, i*9:i*9+9] = (SourceRotation[self.corrList[i][0]].T).flatten()
            Ed_cVector = Ed_cVector.T

            return np.dot(self.Ed_A.T, Ed_cVector)


        #Landmark imitation term
        def MakeEl_ATA_ATc():
            El_cVector = sp.lil_matrix((1, len(self.SOURCE_LANDMARKS)*3), dtype=np.float32)
            El_A = sp.lil_matrix((len(self.SOURCE_LANDMARKS)*3, len(self.AN_Vertices)*3 + len(self.AN_Faces)*3), dtype=np.float32)

            for i in range(len(self.SOURCE_LANDMARKS)):
                idx = self.AVATAR_LANDMARKS[i]
                El_A[i*3, idx*3] = 1
                El_A[i*3+1, idx*3+1] = 1
                El_A[i*3+2, idx*3+2] = 1

            for i in range(len(self.SOURCE_LANDMARKS)):
                vector_s = self.SE_Vertices[self.SOURCE_LANDMARKS[i],0:3] - self.SN_Vertices[self.SOURCE_LANDMARKS[i],0:3]
                if(0 <= i <= 4):
                    El_cVector[0, i*3:i*3+3] = np.dot(self.PartsRot[0], vector_s)*self.PartsSize[0] + self.AN_Vertices[self.AVATAR_LANDMARKS[i],0:3]
                elif(5 <= i <= 9):
                    El_cVector[0, i*3:i*3+3] = np.dot(self.PartsRot[1], vector_s)*self.PartsSize[1] + self.AN_Vertices[self.AVATAR_LANDMARKS[i],0:3]
                elif(10 <= i <= 18):
                    El_cVector[0, i*3:i*3+3] = np.dot(self.PartsRot[2], vector_s)*self.PartsSize[2] + self.AN_Vertices[self.AVATAR_LANDMARKS[i],0:3]
                elif(19 <= i <= 24):
                    El_cVector[0, i*3:i*3+3] = np.dot(self.PartsRot[3], vector_s)*self.PartsSize[3] + self.AN_Vertices[self.AVATAR_LANDMARKS[i],0:3]
                elif(25 <= i <= 30):
                    El_cVector[0, i*3:i*3+3] = np.dot(self.PartsRot[4], vector_s)*self.PartsSize[4] + self.AN_Vertices[self.AVATAR_LANDMARKS[i],0:3]
                elif(31 <= i <= 50):
                    El_cVector[0, i*3:i*3+3] = np.dot(self.PartsRot[5], vector_s)*self.PartsSize[5] + self.AN_Vertices[self.AVATAR_LANDMARKS[i],0:3]
            El_cVector = El_cVector.T
            return (np.dot(El_A.T, El_A), np.dot(El_A.T, El_cVector))


        #Constraints term
        def MakeEcons_ATA_ATc():
            Econs_A = sp.lil_matrix((0, len(self.AN_Vertices)*3 + len(self.AN_Faces)*3), dtype=np.float32)
            Econs_cVector = sp.csc_matrix((0, 1), dtype=np.float32)

            def AddConstraints(idx, coordinate):
                Aadd = sp.lil_matrix((3, len(self.AN_Vertices)*3 + len(self.AN_Faces)*3), dtype=np.float32)
                Aadd[0, idx*3]   = 1
                Aadd[1, idx*3+1] = 1
                Aadd[2, idx*3+2] = 1
                cadd = sp.csc_matrix(np.array(coordinate).reshape(3, 1), dtype=np.float32)
                return (sp.vstack((Econs_A, Aadd)), sp.vstack((Econs_cVector, cadd)))

            #move unused points to (0, 0, 0)
            unusedpoints = []
            Used = np.full(len(self.AN_Vertices), False, dtype=bool)
            for i in range(len(self.AN_Faces)):
                Used[self.AN_Faces[i,0]] = True
                Used[self.AN_Faces[i,2]] = True
                Used[self.AN_Faces[i,4]] = True
            for i in range(len(self.AN_Vertices)):
                if(Used[i] ==False):
                    unusedpoints += [i]
            for i in range(len(unusedpoints)):
                (Econs_A, Econs_cVector) = AddConstraints(unusedpoints[i], [0, 0, 0])            

            #bind non-facial area
            for ls in self.backlandmark_list:
                idx = ls[1]
                (Econs_A, Econs_cVector) = AddConstraints(idx, self.AN_Vertices[idx,0:3])
                
            return (np.dot(Econs_A.T, Econs_A), np.dot(Econs_A.T, Econs_cVector))


        start_time = time.time(); print "Make Ed...   ",
        Ed_ATc = MakeEd_ATc()
        elapsed_time = time.time() - start_time; print "done (%f sec)" % (elapsed_time)

        start_time = time.time(); print "Make El...   ",
        (El_ATA, El_ATc) = MakeEl_ATA_ATc()
        elapsed_time = time.time() - start_time; print "done (%f sec)" % (elapsed_time)

        start_time = time.time(); print "Make Econs...",
        (Econs_ATA, Econs_ATc) = MakeEcons_ATA_ATc()
        elapsed_time = time.time() - start_time; print "done (%f sec)" % (elapsed_time)

        start_time = time.time(); print "Solving Matrix system...",
        wd=1; wl=100; ws=10; wi=1; wcons=1000
        ATA_sum = wd*self.Ed_ATA + wl*El_ATA + ws*self.Es_ATA + wi*self.Ei_ATA + wcons*Econs_ATA
        ATc_sum = wd*Ed_ATc + wl*El_ATc + ws*self.Es_ATc + wi*self.Ei_ATc + wcons*Econs_ATc
        #ATA_sum = wd*self.Ed_ATA + 0.001*self.Ei_ATA
        #ATc_sum = wd*Ed_ATc + 0.001*self.Ei_ATc

        x = spsolve(ATA_sum, ATc_sum)
        elapsed_time = time.time() - start_time; print "done (%f sec)" % (elapsed_time)
        elapsed_time = time.time() - start_dtf; print "\nAll calculation was finished  (Elapsed time: %f)" % (elapsed_time)

        self.Vertices = x[0:len(self.AN_Vertices)*3].reshape(len(self.AN_Vertices), 3)



class BMMng():
    def __init__(self, blendshapespath, avatarpath):
        self.path = blendshapespath
        self.avatarpath = avatarpath
        self.BlendShapes = []
        self.Avatar = []
        self.Correspondences = []
        self.AVATAR_LANDMARKS = []
        self.AVATAR_BACKWARD_LANDMARKS = []
        self.SOURCE_LANDMARKS = self.LoadLandmarks("data/landmarks/SOURCE/FACIAL_LANDMARKS.txt")
        self.SOURCE_BACKWARD_LANDMARKS = self.LoadLandmarks("data/landmarks/SOURCE/BACKWARD_LANDMARKS.txt")
        self.config = self.LoadConfig("config.txt")
        self.BS_forPartsSize = []
        d = path.split(self.avatarpath) #split path in (head, tail) 
        d = d[1].split(".")
        self.avatarname = d[0]

    def LoadConfig(self, filepath):
        config = [False]
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    words = line.split()
                    if words[0] == "SELECT_BACKWARD_LANDMARKS_MANUALLY":
                        if words[1] == "True":
                            config[0] = True
        return config
        



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
            if (i==14 or i==15 or i==16 or i==21 or i==28 or i==29 or i==36 or i==38 or i==40 or i==41):
                self.BS_forPartsSize.append(currMesh)


    def LoadAvatar(self):
        currAvatar = MyMesh(self.avatarpath)
        currAvatar.LoadAvatarMesh()
        self.Avatar += [currAvatar]



    def LoadLandmarks(self, filepath):
        if path.exists(filepath):
            print "\nLoading ", filepath

            with open(filepath, 'r') as f:
                line = f.readline()
                words = line.split()
                landmarks = [int(i) for i in words]
                print landmarks
            return landmarks
        else:
            print("Landmark file does not exists: {}".format(filepath))
            return []


    def SaveLandmarks(self, filepath, landmarks):
        dirpath = path.dirname(filepath)
        if not path.exists(dirpath):
            os.makedirs(dirpath)
            
        with open(filepath, 'w') as f:
            for i in landmarks:
                f.write("%d " %(i))


    def SaveMesh(self, dest, filename, Vertices, Faces, Textures, TexName, Flags):

        ''' Write result 3D mesh into a .ply file'''
        print "Saving ", filename

        f = open(dest+filename, 'wb')

        # Write headers
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        for name in TexName:
            f.write("comment TextureFile %s\n" %(name))
        f.write("element vertex %d\n" %(len(Vertices)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face %d\n" %(len(Faces)))
        f.write("property list uchar int vertex_indices\n")
        if(Flags[0]):
            f.write("property list uchar float texcoord\n")
        if(len(TexName)>1):
            f.write("property int texnumber\n")
        f.write("end_header\n")


        for i in range(len(Vertices)):
            if(self.CaredVertices[i]==True):
                f.write("%f %f %f\n" %(Vertices[i,0], Vertices[i,1], Vertices[i,2]))
            else:
                f.write("%f %f %f\n" %(self.Avatar[0].Vertices[i,0], self.Avatar[0].Vertices[i,1], self.Avatar[0].Vertices[i,2]))

        for i in range(len(Faces)):
            if(Flags[0]):
                if(len(TexName)>1):
                    f.write("3 %d %d %d 6 %f %f %f %f %f %f %d\n" %(Faces[i, 0], Faces[i, 2], Faces[i, 4], 
                                                                Textures[i, 0], Textures[i, 1],
                                                                Textures[i, 2], Textures[i, 3], 
                                                                Textures[i, 4], Textures[i, 5], Faces[i, 6]))
                else:
                    f.write("3 %d %d %d 6 %f %f %f %f %f %f\n" %(Faces[i, 0], Faces[i, 2], Faces[i, 4], 
                                                                Textures[i, 0], Textures[i, 1],
                                                                Textures[i, 2], Textures[i, 3], 
                                                                Textures[i, 4], Textures[i, 5]))
            else:
                f.write("3 %d %d %d\n" %(Faces[i, 0], Faces[i, 2], Faces[i, 4]))

        f.close()

    def RescaleAvatar(self):
        '''
        #アバターを回転させたいときに使うやつ
        angle_x=-20*np.pi/180
        angle_y=0*np.pi/180
        RotX = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]], dtype=np.float32)
        RotY = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=np.float32)
        for i in range(len(self.Avatar[0].Vertices)):
            self.Avatar[0].Vertices[i,0:3] = np.dot(np.dot(RotX, RotY), self.Avatar[0].Vertices[i,0:3])
        self.CaredVertices = np.full(len(self.Avatar[0].Vertices), True, dtype=bool)
        self.SaveMesh("Results/", "rot.ply", self.Avatar[0].Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)
        '''


        Vertices = self.Avatar[0].Vertices

        xs=Vertices[0,0]; xl=xs
        ys=Vertices[0,1]; yl=ys
        zs=Vertices[0,2]; zl=zs

        #AvatarSize measurement
        for v in Vertices:
            if(v[0]<xs):
                xs=v[0]
            elif(xl<v[0]):
                xl=v[0]
            if(v[1]<ys):
                ys=v[1]
            elif(yl<v[1]):
                yl=v[1]
            if(v[2]<zs):
                zs=v[2]
            elif(zl<v[2]):
                zl=v[2]

        print "AvatarSize: x[%f, %f], y[%f, %f], z[%f, %f]" %(xs, xl, ys, yl, zs, zl)

        #Move Avatar to center
        xc=(xs+xl)/2
        yc=(ys+yl)/2
        zc=(zs+zl)/2
        k=0
        for v in Vertices:
            v[0] = v[0]-xc
            v[1] = v[1]-yc
            v[2] = v[2]-zc
            self.Avatar[0].Vertices[k] = v
            k+=1

        #Rescale
        avg=(xl-xs+yl-ys+zl-zs)/3
        k=0
        for v in Vertices:
            v = v*214.8093/avg
            self.Avatar[0].Vertices[k][0:3] = v[0:3]
            k+=1

        '''''''''select landmarks in tkinter'''''''''

        V = imp.load_source('show3d', APP_ROOT+'/show3d.py')
        def select_landmarks(reference_landmarks, previous_landmarks):
            root = tk.Tk()
            root.title(u"Select Landmarks")
            root.resizable(0, 0)
            Avatar = Mesh_VFN(self.Avatar[0].Vertices, self.Avatar[0].Faces)
            Source = Mesh_VFN(self.BlendShapes[0].Vertices, self.BlendShapes[0].Faces)
            Visualize = V.SelectLandmarks(root, Avatar, Source, reference_landmarks, previous_landmarks)
            Visualize.mainloop()
            if (min(Visualize.AVATAR_LANDMARKS) == None):
                print "\nplease select more landmarks"
                select_landmarks(reference_landmarks, Visualize.AVATAR_LANDMARKS)

            return Visualize.AVATAR_LANDMARKS
        
        filepath = "data/landmarks/landmarks_%s.txt" %(self.avatarname)
        previous_landmarks = self.LoadLandmarks(filepath)

        if self.config[0]: # If you select backward landmarks manually
            print("Use facial + backward landmarks")
            reference_landmarks = self.SOURCE_LANDMARKS + self.SOURCE_BACKWARD_LANDMARKS
        else:
            print("Use only facial landmarks")
            reference_landmarks = self.SOURCE_LANDMARKS
        self.AVATAR_LANDMARKS = select_landmarks(reference_landmarks, previous_landmarks)
        self.SaveLandmarks(filepath, self.AVATAR_LANDMARKS)

        if self.config[0]:
            self.AVATAR_BACKWARD_LANDMARKS = list(np.array(self.AVATAR_LANDMARKS)[len(self.SOURCE_LANDMARKS):])
            self.AVATAR_LANDMARKS = list(np.array(self.AVATAR_LANDMARKS)[:len(self.SOURCE_LANDMARKS)])


        ''''''''''''''''''''''''''''''''''''''''''''''''
        #bind nosetop
        xc=Vertices[self.AVATAR_LANDMARKS[13], 0] - self.BlendShapes[0].Vertices[self.SOURCE_LANDMARKS[13]][0]
        yc=Vertices[self.AVATAR_LANDMARKS[13], 1] - self.BlendShapes[0].Vertices[self.SOURCE_LANDMARKS[13]][1]
        zc=Vertices[self.AVATAR_LANDMARKS[13], 2] - self.BlendShapes[0].Vertices[self.SOURCE_LANDMARKS[13]][2]
        k=0
        for v in Vertices:
            v[0] = v[0]-xc
            v[1] = v[1]-yc
            v[2] = v[2]-zc
            self.Avatar[0].Vertices[k] = v
            k+=1


        self.CaredVertices = np.full(len(self.Avatar[0].Vertices), True, dtype=bool)
        self.SaveMesh("Results/", "Rescaled.ply", self.Avatar[0].Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)

    def TriangleCorrespondence(self, level):
        AvatarFaces = self.Avatar[0].Faces
        AvatarVertices = self.Avatar[0].Vertices
        AvatarV4 = np.zeros((len(AvatarFaces), 3), dtype = np.float32)
        SourceFaces = self.BlendShapes[0].Faces
        SourceVertices = self.BlendShapes[0].Vertices
        SourceV4 = np.zeros((len(SourceFaces), 3), dtype = np.float32)
        unusedpoints = []

        for i in self.SOURCE_BACKWARD_LANDMARKS:
            norm = LA.norm(SourceVertices[i,0:3] - AvatarVertices[:,0:3], axis=1)
            self.AVATAR_BACKWARD_LANDMARKS += [np.argmin(norm)]

        Vinv = np.zeros((len(AvatarFaces), 3, 3), dtype = np.float32)  # V inverse of the Avatar triangle
        self.Adjacent = [[] for row in range(len(AvatarFaces))]         #indices of self.adjacent triangles
        self.Result_iteration = np.zeros((4, len(AvatarVertices), 3), dtype = np.float32)


        #find adjacent triangles
        start_time = time.time()
        print "Finding adjacent triangles...",
        Flist = AvatarFaces[:,[0, 2, 4]].tolist()
        for i in xrange(len(Flist)):
            set1 = set(Flist[i])
            for j in xrange(i+1, len(Flist)):
                set2 = set(Flist[j])
                if(len(set1 & set2)==2):
                    self.Adjacent[i] += [j]
                    self.Adjacent[j] += [i]
                if(len(self.Adjacent[i])>=3): break
        
        print "done (%f sec)" %(time.time() - start_time)

        #find parts that contains facial landmarks
        start_time = time.time()
        print "Finding parts that contains facial landmarks...",
        self.CaredFaces = np.full(len(AvatarFaces), False, dtype=bool)
        self.CaredVertices = np.full(len(AvatarVertices), False, dtype=bool)
        StartFaces = []
        for i in range(len(AvatarFaces)):
            for j in self.AVATAR_LANDMARKS:
                if(AvatarFaces[i][0]==j or AvatarFaces[i][2]==j or AvatarFaces[i][4]==j):
                    self.CaredFaces[i] = True
                    StartFaces += [i]
                    break

        #breadth‐first search
        def Search_all_adjacent(start_idx):
            Q = Queue()
            Q.put(start_idx)
            while(not Q.empty()):
                idx = Q.get()
                for i in self.Adjacent[idx]:
                    if(self.CaredFaces[i]==False):
                        self.CaredFaces[i] = True
                        Q.put(i)
        for i in StartFaces:
            Search_all_adjacent(i)



        for i in range(len(AvatarFaces)):
            if(self.CaredFaces[i]==True):
                self.CaredVertices[AvatarFaces[i,0]] = True
                self.CaredVertices[AvatarFaces[i,2]] = True
                self.CaredVertices[AvatarFaces[i,4]] = True

        print "done (%f sec)" %(time.time() - start_time)

        #compute avatar v4 from v1,v2,v3
        start_time = time.time()
        print "Computing V4 and Normals...",
        for k in xrange(len(AvatarFaces)):
            cross = np.cross(AvatarVertices[AvatarFaces[k, 2],0:3] - AvatarVertices[AvatarFaces[k, 0],0:3], AvatarVertices[AvatarFaces[k, 4],0:3]-AvatarVertices[AvatarFaces[k, 0],0:3])
            AvatarV4[k] = AvatarVertices[AvatarFaces[k, 0],0:3] + cross / sqrt(LA.norm(cross))
        
        #compute source v4 and vertex normals from v1,v2,v3
        SourceTrianglesList = [[] for row in range(len(SourceVertices))] #Triangle index list that shares the vertex
        SourceTriangleNormals = np.zeros((len(SourceFaces), 3), dtype = np.float32)
        SourceVertexNormals = np.zeros((len(SourceFaces), 3), dtype = np.float32)
        
        for k in xrange(len(SourceFaces)):
            cross = np.cross(SourceVertices[SourceFaces[k, 2],0:3] - SourceVertices[SourceFaces[k, 0],0:3], SourceVertices[SourceFaces[k, 4],0:3]-SourceVertices[SourceFaces[k, 0],0:3])
            SourceV4[k] = SourceVertices[SourceFaces[k, 0],0:3] + cross / sqrt(LA.norm(cross))
            SourceTriangleNormals[k] = cross / LA.norm(cross)

        for i in xrange(len(SourceFaces)):
            SourceTrianglesList[SourceFaces[i,0]] += [i]
            SourceTrianglesList[SourceFaces[i,2]] += [i]
            SourceTrianglesList[SourceFaces[i,4]] += [i]

        for i in xrange(len(SourceVertices)):
                SourceN = SourceTriangleNormals[SourceTrianglesList[i][0]]
                for j in range(1, len(SourceTrianglesList[i])):
                    SourceN = SourceN + SourceTriangleNormals[SourceTrianglesList[i][j]]
                SourceVertexNormals[i] = SourceN / LA.norm(SourceN)

        print "done (%f sec)" %(time.time() - start_time)


        start_time = time.time()
        print "Computing V inverce...",
        #compute avatar V inverse
        for i in xrange(len(AvatarFaces)):
            a = np.array([AvatarVertices[AvatarFaces[i,2],0:3] - AvatarVertices[AvatarFaces[i,0],0:3]])
            b = np.array([AvatarVertices[AvatarFaces[i,4],0:3] - AvatarVertices[AvatarFaces[i,0],0:3]])
            c = np.array([AvatarV4[i] - AvatarVertices[AvatarFaces[i,0],0:3]])

            Vinv[i] = np.linalg.solve(np.hstack([a.T, b.T, c.T]), np.identity(3))

        print "done (%f sec)" %(time.time() - start_time)


        #make large matrices and vectors
        def MakeEs_ATA_ATc():
            adj_num = 0
            for i in xrange(len(self.Adjacent)):
                for j in xrange(len(self.Adjacent[i])):
                    adj_num+=1

            Es_cVector = sp.csr_matrix((1, adj_num*9), dtype=np.float32)
            Es_cVector = Es_cVector.T
            Es_A = sp.lil_matrix((adj_num*9, len(AvatarVertices)*3 + len(AvatarFaces)*3), dtype=np.float32)

            row = 0
            Vinv_list = Vinv.tolist()
            Faces = AvatarFaces.tolist()
            len_AvatarVertices = len(AvatarVertices)

            for i in xrange(len(self.Adjacent)):
                for j in self.Adjacent[i]:

                    e1 = np.sum(Vinv[i,0:3,0])
                    e2 = np.sum(Vinv[i,0:3,1])
                    e3 = np.sum(Vinv[i,0:3,2])

                    idx_v1 = Faces[i][0]
                    Es_A[row*9, idx_v1*3]     = -e1
                    Es_A[row*9+1, idx_v1*3+1] = -e1
                    Es_A[row*9+2, idx_v1*3+2] = -e1
                    Es_A[row*9+3, idx_v1*3]   = -e2
                    Es_A[row*9+4, idx_v1*3+1] = -e2
                    Es_A[row*9+5, idx_v1*3+2] = -e2
                    Es_A[row*9+6, idx_v1*3]   = -e3
                    Es_A[row*9+7, idx_v1*3+1] = -e3
                    Es_A[row*9+8, idx_v1*3+2] = -e3

                    idx_v2 = Faces[i][2]
                    Es_A[row*9, idx_v2*3]     = Vinv_list[i][0][0]
                    Es_A[row*9+1, idx_v2*3+1] = Vinv_list[i][0][0]
                    Es_A[row*9+2, idx_v2*3+2] = Vinv_list[i][0][0]
                    Es_A[row*9+3, idx_v2*3]   = Vinv_list[i][0][1]
                    Es_A[row*9+4, idx_v2*3+1] = Vinv_list[i][0][1]
                    Es_A[row*9+5, idx_v2*3+2] = Vinv_list[i][0][1]
                    Es_A[row*9+6, idx_v2*3]   = Vinv_list[i][0][2]
                    Es_A[row*9+7, idx_v2*3+1] = Vinv_list[i][0][2]
                    Es_A[row*9+8, idx_v2*3+2] = Vinv_list[i][0][2]

                    idx_v3 = Faces[i][4]
                    Es_A[row*9, idx_v3*3]     = Vinv_list[i][1][0]
                    Es_A[row*9+1, idx_v3*3+1] = Vinv_list[i][1][0]
                    Es_A[row*9+2, idx_v3*3+2] = Vinv_list[i][1][0]
                    Es_A[row*9+3, idx_v3*3]   = Vinv_list[i][1][1]
                    Es_A[row*9+4, idx_v3*3+1] = Vinv_list[i][1][1]
                    Es_A[row*9+5, idx_v3*3+2] = Vinv_list[i][1][1]
                    Es_A[row*9+6, idx_v3*3]   = Vinv_list[i][1][2]
                    Es_A[row*9+7, idx_v3*3+1] = Vinv_list[i][1][2]
                    Es_A[row*9+8, idx_v3*3+2] = Vinv_list[i][1][2]

                    Es_A[row*9, i*3+len_AvatarVertices*3]     = Vinv_list[i][2][0]
                    Es_A[row*9+1, i*3+len_AvatarVertices*3+1] = Vinv_list[i][2][0]
                    Es_A[row*9+2, i*3+len_AvatarVertices*3+2] = Vinv_list[i][2][0]
                    Es_A[row*9+3, i*3+len_AvatarVertices*3]   = Vinv_list[i][2][1]
                    Es_A[row*9+4, i*3+len_AvatarVertices*3+1] = Vinv_list[i][2][1]
                    Es_A[row*9+5, i*3+len_AvatarVertices*3+2] = Vinv_list[i][2][1]
                    Es_A[row*9+6, i*3+len_AvatarVertices*3]   = Vinv_list[i][2][2]
                    Es_A[row*9+7, i*3+len_AvatarVertices*3+1] = Vinv_list[i][2][2]
                    Es_A[row*9+8, i*3+len_AvatarVertices*3+2] = Vinv_list[i][2][2]

                    e1 = np.sum(Vinv[j,0:3,0])
                    e2 = np.sum(Vinv[j,0:3,1])
                    e3 = np.sum(Vinv[j,0:3,2])

                    idx_v1 = Faces[j][0]
                    Es_A[row*9, idx_v1*3]     += e1
                    Es_A[row*9+1, idx_v1*3+1] += e1
                    Es_A[row*9+2, idx_v1*3+2] += e1
                    Es_A[row*9+3, idx_v1*3]   += e2
                    Es_A[row*9+4, idx_v1*3+1] += e2
                    Es_A[row*9+5, idx_v1*3+2] += e2
                    Es_A[row*9+6, idx_v1*3]   += e3
                    Es_A[row*9+7, idx_v1*3+1] += e3
                    Es_A[row*9+8, idx_v1*3+2] += e3

                    idx_v2 = Faces[j][2]
                    Es_A[row*9, idx_v2*3]     += -Vinv_list[j][0][0]
                    Es_A[row*9+1, idx_v2*3+1] += -Vinv_list[j][0][0]
                    Es_A[row*9+2, idx_v2*3+2] += -Vinv_list[j][0][0]
                    Es_A[row*9+3, idx_v2*3]   += -Vinv_list[j][0][1]
                    Es_A[row*9+4, idx_v2*3+1] += -Vinv_list[j][0][1]
                    Es_A[row*9+5, idx_v2*3+2] += -Vinv_list[j][0][1]
                    Es_A[row*9+6, idx_v2*3]   += -Vinv_list[j][0][2]
                    Es_A[row*9+7, idx_v2*3+1] += -Vinv_list[j][0][2]
                    Es_A[row*9+8, idx_v2*3+2] += -Vinv_list[j][0][2]

                    idx_v3 = Faces[j][4]
                    Es_A[row*9, idx_v3*3]     += -Vinv_list[j][1][0]
                    Es_A[row*9+1, idx_v3*3+1] += -Vinv_list[j][1][0]
                    Es_A[row*9+2, idx_v3*3+2] += -Vinv_list[j][1][0]
                    Es_A[row*9+3, idx_v3*3]   += -Vinv_list[j][1][1]
                    Es_A[row*9+4, idx_v3*3+1] += -Vinv_list[j][1][1]
                    Es_A[row*9+5, idx_v3*3+2] += -Vinv_list[j][1][1]
                    Es_A[row*9+6, idx_v3*3]   += -Vinv_list[j][1][2]
                    Es_A[row*9+7, idx_v3*3+1] += -Vinv_list[j][1][2]
                    Es_A[row*9+8, idx_v3*3+2] += -Vinv_list[j][1][2]

                    Es_A[row*9, j*3+len_AvatarVertices*3]     += -Vinv_list[j][2][0]
                    Es_A[row*9+1, j*3+len_AvatarVertices*3+1] += -Vinv_list[j][2][0]
                    Es_A[row*9+2, j*3+len_AvatarVertices*3+2] += -Vinv_list[j][2][0]
                    Es_A[row*9+3, j*3+len_AvatarVertices*3]   += -Vinv_list[j][2][1]
                    Es_A[row*9+4, j*3+len_AvatarVertices*3+1] += -Vinv_list[j][2][1]
                    Es_A[row*9+5, j*3+len_AvatarVertices*3+2] += -Vinv_list[j][2][1]
                    Es_A[row*9+6, j*3+len_AvatarVertices*3]   += -Vinv_list[j][2][2]
                    Es_A[row*9+7, j*3+len_AvatarVertices*3+1] += -Vinv_list[j][2][2]
                    Es_A[row*9+8, j*3+len_AvatarVertices*3+2] += -Vinv_list[j][2][2]

                    row += 1

            return (np.dot(Es_A.T, Es_A), np.dot(Es_A.T, Es_cVector))

        def MakeEi_ATA_ATc():
            Flat_identity = sp.csc_matrix((0, 1), dtype=np.float32)
            for i in xrange(len(AvatarFaces)):
                Flat_identity = sp.vstack((Flat_identity, sp.identity(3, format="lil").reshape((9, 1))), format="csc", dtype=np.float32)

            Ei_cVector = Flat_identity
            Ei_A = sp.lil_matrix((len(AvatarFaces)*9, len(AvatarVertices)*3 + len(AvatarFaces)*3), dtype=np.float32)

            Vinv_list = Vinv.tolist()
            Faces = AvatarFaces.tolist()
            len_AvatarVertices = len(AvatarVertices)
            for i in xrange(len(AvatarFaces)):
                idx_f = i
                e1 = np.sum(Vinv[idx_f,0:3,0])
                e2 = np.sum(Vinv[idx_f,0:3,1])
                e3 = np.sum(Vinv[idx_f,0:3,2])
                
                idx_v1 = Faces[idx_f][0]
                Ei_A[i*9, idx_v1*3]     = -e1
                Ei_A[i*9+1, idx_v1*3+1] = -e1
                Ei_A[i*9+2, idx_v1*3+2] = -e1
                Ei_A[i*9+3, idx_v1*3]   = -e2
                Ei_A[i*9+4, idx_v1*3+1] = -e2
                Ei_A[i*9+5, idx_v1*3+2] = -e2
                Ei_A[i*9+6, idx_v1*3]   = -e3
                Ei_A[i*9+7, idx_v1*3+1] = -e3
                Ei_A[i*9+8, idx_v1*3+2] = -e3

                idx_v2 = Faces[idx_f][2]
                Ei_A[i*9, idx_v2*3]     = Vinv_list[idx_f][0][0]
                Ei_A[i*9+1, idx_v2*3+1] = Vinv_list[idx_f][0][0]
                Ei_A[i*9+2, idx_v2*3+2] = Vinv_list[idx_f][0][0]
                Ei_A[i*9+3, idx_v2*3]   = Vinv_list[idx_f][0][1]
                Ei_A[i*9+4, idx_v2*3+1] = Vinv_list[idx_f][0][1]
                Ei_A[i*9+5, idx_v2*3+2] = Vinv_list[idx_f][0][1]
                Ei_A[i*9+6, idx_v2*3]   = Vinv_list[idx_f][0][2]
                Ei_A[i*9+7, idx_v2*3+1] = Vinv_list[idx_f][0][2]
                Ei_A[i*9+8, idx_v2*3+2] = Vinv_list[idx_f][0][2]
                
                idx_v3 = Faces[idx_f][4]
                Ei_A[i*9, idx_v3*3]     = Vinv_list[idx_f][1][0]
                Ei_A[i*9+1, idx_v3*3+1] = Vinv_list[idx_f][1][0]
                Ei_A[i*9+2, idx_v3*3+2] = Vinv_list[idx_f][1][0]
                Ei_A[i*9+3, idx_v3*3]   = Vinv_list[idx_f][1][1]
                Ei_A[i*9+4, idx_v3*3+1] = Vinv_list[idx_f][1][1]
                Ei_A[i*9+5, idx_v3*3+2] = Vinv_list[idx_f][1][1]
                Ei_A[i*9+6, idx_v3*3]   = Vinv_list[idx_f][1][2]
                Ei_A[i*9+7, idx_v3*3+1] = Vinv_list[idx_f][1][2]
                Ei_A[i*9+8, idx_v3*3+2] = Vinv_list[idx_f][1][2]

                Ei_A[i*9, idx_f*3+len_AvatarVertices*3]    = Vinv_list[idx_f][2][0]
                Ei_A[i*9+1, idx_f*3+len_AvatarVertices*3+1] = Vinv_list[idx_f][2][0]
                Ei_A[i*9+2, idx_f*3+len_AvatarVertices*3+2] = Vinv_list[idx_f][2][0]
                Ei_A[i*9+3, idx_f*3+len_AvatarVertices*3]   = Vinv_list[idx_f][2][1]
                Ei_A[i*9+4, idx_f*3+len_AvatarVertices*3+1] = Vinv_list[idx_f][2][1]
                Ei_A[i*9+5, idx_f*3+len_AvatarVertices*3+2] = Vinv_list[idx_f][2][1]
                Ei_A[i*9+6, idx_f*3+len_AvatarVertices*3]   = Vinv_list[idx_f][2][2]
                Ei_A[i*9+7, idx_f*3+len_AvatarVertices*3+1] = Vinv_list[idx_f][2][2]
                Ei_A[i*9+8, idx_f*3+len_AvatarVertices*3+2] = Vinv_list[idx_f][2][2]

            return (np.dot(Ei_A.T, Ei_A), np.dot(Ei_A.T, Ei_cVector))

        def MakeEcons_ATA_ATc():
            Econs_A = sp.lil_matrix((0, len(AvatarVertices)*3 + len(AvatarFaces)*3), dtype=np.float32)
            Econs_cVector = sp.csc_matrix((0, 1), dtype=np.float32)

            def AddConstraints(idx, coordinate):
                Aadd = sp.lil_matrix((3, len(AvatarVertices)*3 + len(AvatarFaces)*3), dtype=np.float32)
                Aadd[0, idx*3]   = 1
                Aadd[1, idx*3+1] = 1
                Aadd[2, idx*3+2] = 1
                cadd = sp.csc_matrix(np.array(coordinate).reshape(3, 1), dtype=np.float32)
                return (sp.vstack((Econs_A, Aadd)), sp.vstack((Econs_cVector, cadd)))

            #move unused points to (0, 0, 0)
            unusedpoints = []
            Used = np.full(len(AvatarVertices), False, dtype=bool)
            for i in xrange(len(AvatarFaces)):
                Used[AvatarFaces[i,0]] = True
                Used[AvatarFaces[i,2]] = True
                Used[AvatarFaces[i,4]] = True
            for i in xrange(len(AvatarVertices)):
                if(Used[i] ==False):
                    unusedpoints += [i]
            for i in xrange(len(unusedpoints)):
                (Econs_A, Econs_cVector) = AddConstraints(unusedpoints[i], [0, 0, 0])
            #Facial landmark constraints
            for i in xrange(len(self.AVATAR_LANDMARKS)):
                (Econs_A, Econs_cVector) = AddConstraints(self.AVATAR_LANDMARKS[i], SourceVertices[self.SOURCE_LANDMARKS[i],0:3])
            #Backward landmark constraints
            if self.config[0]:
                for i in xrange(len(self.AVATAR_BACKWARD_LANDMARKS)):
                    (Econs_A, Econs_cVector) = AddConstraints(self.AVATAR_BACKWARD_LANDMARKS[i], SourceVertices[self.SOURCE_LANDMARKS[i],0:3])
            else:
                for i in self.SOURCE_BACKWARD_LANDMARKS:
                    norm = LA.norm(SourceVertices[i,0:3] - AvatarVertices[:,0:3], axis=1)
                    (Econs_A, Econs_cVector) = AddConstraints(np.argmin(norm), SourceVertices[i,0:3])

            #prevent collapse
            #(Ei_A, Ei_cVector) = Ei_AddConstraints(499, AvatarVertices[499, 0:3])

            return (np.dot(Econs_A.T, Econs_A), np.dot(Econs_A.T, Econs_cVector))



        def ClosestValidPoint(vertices, threshold):
            AvatarTrianglesList = [[] for row in range(len(AvatarVertices))] #Triangle index list that shares the vertex
            AvatarTriangleNormals = np.zeros((len(AvatarFaces), 3), dtype = np.float32)
            AvatarVertexNormals = np.zeros((len(AvatarVertices), 3), dtype = np.float32)

            Validpoint = [[] for row in range(len(AvatarVertices))]      #index of closest valid point on the source mesh to target vertex

            Ec_cVector = sp.lil_matrix((1, len(AvatarVertices)*3), dtype = np.float32)
            Ec_A = sp.lil_matrix((len(AvatarVertices)*3, len(AvatarVertices)*3 + len(AvatarFaces)*3), dtype=np.float32)

            for i in xrange(0, len(AvatarFaces)):
                AvatarTrianglesList[AvatarFaces[i,0]] += [i]
                AvatarTrianglesList[AvatarFaces[i,2]] += [i]
                AvatarTrianglesList[AvatarFaces[i,4]] += [i]

            #compute polygon normals of the arguments
            for k in xrange(0, len(AvatarFaces)):
                cross = np.cross(vertices[AvatarFaces[k, 2]*3:AvatarFaces[k, 2]*3+3] - vertices[AvatarFaces[k, 0]*3:AvatarFaces[k, 0]*3+3], vertices[AvatarFaces[k, 4]*3:AvatarFaces[k, 4]*3+3]-vertices[AvatarFaces[k, 0]*3:AvatarFaces[k, 0]*3+3])
                AvatarTriangleNormals[k] = cross / LA.norm(cross)

            #compute vertex normals
            for i in xrange(0, len(AvatarVertices)):
                if(len(AvatarTrianglesList[i]) == 0):
                    continue
                AvatarN = AvatarTriangleNormals[AvatarTrianglesList[i][0]]
                for j in xrange(1, len(AvatarTrianglesList[i])):
                    AvatarN = AvatarN + AvatarTriangleNormals[AvatarTrianglesList[i][j]]
                AvatarVertexNormals[i] = AvatarN / LA.norm(AvatarN)

            #compute closest valid point
            for i in xrange(len(AvatarVertices)):
                minimum_cost = 1000
                validindex = -1
                norm = LA.norm(vertices[i*3:i*3+3] - SourceVertices[:,0:3], axis=1)

                while(np.min(norm) < threshold):
                    idx = np.argmin(norm)
                    if(np.dot(AvatarVertexNormals[i], SourceVertexNormals[idx])>0):
                        validindex = idx
                        break
                    else:
                        norm[idx] = threshold

                if(validindex >= 0):
                    Validpoint[i] += [validindex]


            #make Ec_A matrix and Ec_cVector
            for i in xrange(len(AvatarVertices)):
                if(len(Validpoint[i]) > 0):
                    Ec_cVector[0,i*3:i*3+3] = SourceVertices[Validpoint[i],0:3].flatten()
            Ec_cVector = Ec_cVector.T

            for i in xrange(len(AvatarVertices)):
                for j in xrange(3):
                    if(len(Validpoint[i]) > 0):
                        Ec_A[i*3+j, i*3+j] = 1

            return (np.dot(Ec_A.T, Ec_A), np.dot(Ec_A.T, Ec_cVector))

        

        #Iterate optimization 4 times
        print "Generating large matrices..."
        start_time = time.time()
        (self.Es_ATA, self.Es_ATc) = MakeEs_ATA_ATc()
        print "Es :",time.time() - start_time

        start_time = time.time()
        (self.Ei_ATA, self.Ei_ATc) = MakeEi_ATA_ATc()
        print "Ei :",time.time() - start_time

        '''
        #skip iteration and correspondence process when corr_(avatarname).txt exists
        if(path.exists("data/correspondences/corr_%s.txt" %(self.avatarname))):
            f = open("data/correspondences/corr_%s.txt" %(self.avatarname), "rb")
            corr = []
            for line in f:
                word = line.split()
                corr += [[int(word[0]), int(word[1])]]
            self.Correspondences = corr
            f.close()
            return
        '''

        start_time = time.time()
        (Econs_ATA, Econs_ATc) = MakeEcons_ATA_ATc()
        print "Econs :",time.time() - start_time


        #1st
        ws = 1; wi = 0.1; wcons = 1
        A_sum = ws*self.Es_ATA + wi*self.Ei_ATA + wcons*Econs_ATA
        c_sum = ws*self.Es_ATc + wi*self.Ei_ATc + wcons*Econs_ATc
        start_time = time.time()
        FirstIteration = spsolve(A_sum, c_sum)
        elapsed_time = time.time() - start_time
        print "\nFirst iteration was finished  (%f sec)" % (elapsed_time)
        self.Result_iteration[0] = FirstIteration[0:len(AvatarVertices)*3].reshape(len(AvatarVertices), 3)
        self.SaveMesh("Results/", "iteration_1.ply", self.Result_iteration[0], self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)

        #2nd
        print "\n\nComputing closest valid points...",
        start_time = time.time()
        (Ec_ATA, Ec_ATc) = ClosestValidPoint(FirstIteration[0:len(AvatarVertices)*3], 100)
        elapsed_time = time.time() - start_time
        print "done (%f sec)" %(elapsed_time)

        ws = 1; wi = 0.001; wc = 0.1; wcons = 1
        A_sum = ws*self.Es_ATA + wi*self.Ei_ATA + wc*Ec_ATA + Econs_ATA
        c_sum = ws*self.Es_ATc + wi*self.Ei_ATc + wc*Ec_ATc + Econs_ATc
        start_time = time.time()
        SecondIteration = spsolve(A_sum, c_sum)
        elapsed_time = time.time() - start_time
        print "Second iteration was finished (%f sec)" % (elapsed_time)
        self.Result_iteration[1] = SecondIteration[0:len(AvatarVertices)*3].reshape(len(AvatarVertices), 3)
        self.SaveMesh("Results/", "iteration_2.ply", self.Result_iteration[1], self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)

        #3rd
        print "\n\nComputing closest valid points...",
        start_time = time.time()
        (Ec_ATA, Ec_ATc) = ClosestValidPoint(SecondIteration[0:len(AvatarVertices)*3], 10)
        elapsed_time = time.time() - start_time
        print "done (%f sec)" %(elapsed_time)

        ws = 1; wi = 0.001; wc = 1; wcons = 1
        A_sum = ws*self.Es_ATA + wi*self.Ei_ATA + wc*Ec_ATA + Econs_ATA
        c_sum = ws*self.Es_ATc + wi*self.Ei_ATc + wc*Ec_ATc + Econs_ATc
        start_time = time.time()
        ThirdIteration = spsolve(A_sum, c_sum)
        elapsed_time = time.time() - start_time
        print "Third iteration was finished  (%f sec)" % (elapsed_time)
        self.Result_iteration[2] = ThirdIteration[0:len(AvatarVertices)*3].reshape(len(AvatarVertices), 3)
        self.SaveMesh("Results/", "iteration_3.ply", self.Result_iteration[2], self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)

        #4th
        print "\n\nComputing closest valid points...",
        start_time = time.time()
        (Ec_ATA, Ec_ATc) = ClosestValidPoint(ThirdIteration[0:len(AvatarVertices)*3], 5)
        elapsed_time = time.time() - start_time
        print "done (%f sec)" %(elapsed_time)
        
        ws = 1; wi = 0.001; wc = 1; wcons = 1
        A_sum = ws*self.Es_ATA + wi*self.Ei_ATA + wc*Ec_ATA + Econs_ATA
        c_sum = ws*self.Es_ATc + wi*self.Ei_ATc + wc*Ec_ATc + Econs_ATc
        start_time = time.time()
        FourthIteration = spsolve(A_sum, c_sum)
        elapsed_time = time.time() - start_time
        print "Fourth iteration was finished (%f sec)" % (elapsed_time)
        self.Result_iteration[3] = FourthIteration[0:len(AvatarVertices)*3].reshape(len(AvatarVertices), 3)
        self.SaveMesh("Results/", "iteration_4.ply", self.Result_iteration[3], self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)

        ResultVertices = FourthIteration[0:len(AvatarVertices)*3].reshape(len(AvatarVertices), 3)




        ########### get correspondence ##########

        start_time = time.time()

        print "\n\nComputing triangle correspondences...",
        ##compute centroid
        ResultTriangleCentroid = np.zeros((len(AvatarFaces), 3), dtype=np.float32)
        SourceTriangleCentroid = np.zeros((len(SourceFaces), 3), dtype=np.float32)
        ResultTriangleNormals = np.zeros((len(AvatarFaces), 3), dtype=np.float32)
        for i in xrange(len(AvatarFaces)):
            ResultTriangleCentroid[i] = (ResultVertices[AvatarFaces[i, 0]] + ResultVertices[AvatarFaces[i, 2]] + ResultVertices[AvatarFaces[i, 4]])/3
        for i in xrange(len(SourceFaces)):
            SourceTriangleCentroid[i] = (SourceVertices[SourceFaces[i, 0],0:3] + SourceVertices[SourceFaces[i, 2],0:3] + SourceVertices[SourceFaces[i, 4],0:3])/3

        ##compute Result triangle normals
        for i in xrange(len(AvatarFaces)):
            cross = np.cross(ResultVertices[AvatarFaces[i, 2]] - ResultVertices[AvatarFaces[i, 0]], ResultVertices[AvatarFaces[i, 4]]-ResultVertices[AvatarFaces[i, 0]])
            ResultTriangleNormals[i] = cross / LA.norm(cross)

        corr = []
        num_corrS_triangles = np.zeros(len(AvatarFaces), dtype="int") #Hold the number of corresponding source triangles
        ##for all Avatar Faces
        for i in xrange(len(AvatarFaces)):
            threshold = 10
            validindex = -1
            norm = LA.norm(ResultTriangleCentroid[i] - SourceTriangleCentroid, axis=1)

            while(np.min(norm) < threshold):
                idx = np.argmin(norm)
                if(np.dot(ResultTriangleNormals[i], SourceTriangleNormals[idx])>0):
                    validindex = idx
                    break
                else:
                    norm[idx] = threshold
            if(validindex >= 0):
                corr += [[validindex, i]]
                num_corrS_triangles[i] += 1

        ##for all Source Faces
        for i in xrange(len(SourceFaces)):
            threshold = 10
            validindex = -1
            norm = LA.norm(SourceTriangleCentroid[i] - ResultTriangleCentroid, axis=1)

            while(np.min(norm) < threshold):
                idx = np.argmin(norm)
                if(np.dot(SourceTriangleNormals[i], ResultTriangleNormals[idx])>0):
                    validindex = idx
                    break
                else:
                    norm[idx] = threshold
            if(validindex >= 0):
                corr += [[i, validindex]]
                num_corrS_triangles[validindex] += 1

        #Bind non-facial area
        self.backlandmark_list = []
        if self.config[0]:
            for s, a in zip(self.SOURCE_BACKWARD_LANDMARKS, self.AVATAR_BACKWARD_LANDMARKS):
                self.backlandmark_list += [[s, a]]
        else:
            for i in self.SOURCE_BACKWARD_LANDMARKS:
                norm = LA.norm(SourceVertices[i,0:3] - AvatarVertices[:,0:3], axis=1)
                self.backlandmark_list += [[i, np.argmin(norm)]]

        print "done (%f sec)" %(time.time() - start_time)




        #Split Avatar Triangle that have many corresponding Source triangles
        def SplitTriangles(Vertices, Faces, Textures, num_corrS_triangles):
            NewVertices = Vertices
            NewFaces = np.empty((0, 7), dtype=np.int32)
            NewTextures = np.empty((0, 6), dtype=np.float32)

            D = imp.load_source('delaunay', APP_ROOT+'/delaunay.py')


            def delaunay_split(V, Face, Texture, density_mode):
                AddVertices = np.empty((0, 4), dtype=np.float32)
                AddFaces = np.empty((0, 7), dtype=np.int32)
                AddTextures = np.empty((0, 6), dtype=np.float32)

                #compute angle x,y from [0, 0, 1] and Rotation matrix
                cross = np.cross(V[Face[2],0:3] - V[Face[0],0:3], V[Face[4],0:3] - V[Face[0],0:3])
                normal = cross/LA.norm(cross)
                angle_x = -np.arcsin(normal[1])
                cos_angle_y = normal[2]/np.cos(angle_x)
                if(cos_angle_y>1.): cos_angle_y = 1.0
                if(cos_angle_y<-1.): cos_angle_y = -1.0
                angle_y = np.arccos(cos_angle_y)
                if(normal[0]<0): angle_y = -angle_y
                Rot = np.dot([[cos(angle_y), 0, sin(angle_y)], [0 ,1 ,0], [-sin(angle_y), 0, cos(angle_y)]], [[1., 0., 0.], [0, np.cos(angle_x), -sin(angle_x)], [0, sin(angle_x), cos(angle_x)]])
                z = np.dot(Rot.T, V[Face[0],0:3])[2]
                v_coords = np.hstack([np.dot(Rot.T, V[Face[0],0:3])[0:2], np.dot(Rot.T, V[Face[2],0:3])[0:2], np.dot(Rot.T, V[Face[4],0:3])[0:2]])

                (points, tex, simplices) = D.alignedplot_tri(v_coords, Texture[0:6], density_mode)

                new_idx_v = len(V)
                idxlist = []
                for i in xrange(3):
                    idxlist += [Face[2*i]]
                for i in xrange(3, len(points)):
                    AddVertices = np.vstack((AddVertices, np.hstack([np.dot(Rot, np.hstack((points[i], z))), new_idx_v+i-3])))
                    idxlist += [new_idx_v+i-3]
                for i in xrange(len(simplices)):
                    AddFaces = np.vstack((AddFaces, [idxlist[simplices[i,0]], 0, idxlist[simplices[i,1]], 0, idxlist[simplices[i,2]], 0, Face[6]]))
                    AddTextures = np.vstack((AddTextures, [tex[simplices[i,0],0], tex[simplices[i,0],1], tex[simplices[i,1],0], tex[simplices[i,1],1], tex[simplices[i,2],0], tex[simplices[i,2],1]]))

                return (AddVertices, AddFaces, AddTextures)


            for i in xrange(len(Faces)):
                if(num_corrS_triangles[i]<4):
                    NewFaces = np.vstack((NewFaces, Faces[i]))
                    NewTextures = np.vstack((NewTextures, Textures[i]))
                elif(4<=num_corrS_triangles[i]<7):
                    (V, F, T) = delaunay_split(NewVertices, Faces[i], Textures[i], 0)
                    NewVertices = np.vstack((NewVertices, V))
                    NewFaces = np.vstack((NewFaces, F))
                    NewTextures = np.vstack((NewTextures, T))
                elif(7<=num_corrS_triangles[i]<15):
                    (V, F, T) = delaunay_split(NewVertices, Faces[i], Textures[i], 1)
                    NewVertices = np.vstack((NewVertices, V))
                    NewFaces = np.vstack((NewFaces, F))
                    NewTextures = np.vstack((NewTextures, T))
                elif(15<=num_corrS_triangles[i]):
                    (V, F, T) = delaunay_split(NewVertices, Faces[i], Textures[i], 2)
                    NewVertices = np.vstack((NewVertices, V))
                    NewFaces = np.vstack((NewFaces, F))
                    NewTextures = np.vstack((NewTextures, T))

            return (NewVertices, NewFaces, NewTextures)
        
        def SplitTriangles_2(Vertices, Faces, Textures, num_corrS_triangles):
            for i in xrange(len(Faces)):
                if(4<=num_corrS_triangles[i]):
                    #find longest edge
                    v1 = Vertices[Faces[i,0]]; v2 = Vertices[Faces[i,2]]; v3 = Vertices[Faces[i,4]]
                    a = LA.norm(v1-v2); b = LA.norm(v1-v3); c = LA.norm(v2-v3)
                    def find_longest_edge(a, b, c):
                        if(a>b and a>c):   idx1 = Faces[i,2]; idx2 = Faces[i,0]; idx3_i = Faces[i,4]; tex1=Textures[i,2:4]; tex2=Textures[i,0:2]; tex3_i=Textures[i,4:6]; edge=0
                        elif(b>a and b>c): idx1 = Faces[i,0]; idx2 = Faces[i,4]; idx3_i = Faces[i,2]; tex1=Textures[i,0:2]; tex2=Textures[i,4:6]; tex3_i=Textures[i,2:4]; edge=1
                        elif(c>a and c>b): idx1 = Faces[i,4]; idx2 = Faces[i,2]; idx3_i = Faces[i,0]; tex1=Textures[i,4:6]; tex2=Textures[i,2:4]; tex3_i=Textures[i,0:2]; edge=2
                        return (idx1, idx2, idx3_i, tex1, tex2, tex3_i, edge)
                    def adjacent_face(adj_list, idxa, idxb):
                        #find self.adjacent face to the edge
                        adj_face=None
                        for j in adj_list:
                            if((Faces[j,0]==idxa)or(Faces[j,2]==idxa)or(Faces[j,4]==idxa)):
                                if((Faces[j,0]==idxb)or(Faces[j,2]==idxb)or(Faces[j,4]==idxb)):
                                    adj_face = j
                        return adj_face

                    while(True):
                        (idx1, idx2, idx3_i, tex1, tex2, tex3_i, edge) = find_longest_edge(a, b, c)
                        adj = adjacent_face(self.Adjacent[i], idx1, idx2)
                        if(adj==None):
                            if(edge==0): a=0
                            elif(edge==1): b=0
                            elif(edge==2): c=0
                        else: break

                    if(Faces[adj,0]!=idx1 and Faces[adj,0]!=idx2): idx3_adj = Faces[adj,0]; tex3_adj=Textures[adj,0:2]
                    elif(Faces[adj,2]!=idx1 and Faces[adj,2]!=idx2): idx3_adj = Faces[adj,2]; tex3_adj=Textures[adj,2:4]
                    elif(Faces[adj,4]!=idx1 and Faces[adj,4]!=idx2): idx3_adj = Faces[adj,4]; tex3_adj=Textures[adj,4:6]
                    i_adj = adjacent_face(self.Adjacent[i], idx1, idx3_i)
                    adj_adj = adjacent_face(self.Adjacent[adj], idx2, idx3_adj)
                    new_idx_v = len(Vertices)
                    new_idx_f1 = len(Faces)
                    new_idx_f2 = len(Faces)+1

                    #update self.adjacent info
                    self.Adjacent[i] = [new_idx_f1 if x==i_adj else x for x in self.Adjacent[i]]
                    self.Adjacent[i] = [new_idx_f2 if x==adj else x for x in self.Adjacent[i]]
                    self.Adjacent[adj] = [new_idx_f1 if x==i else x for x in self.Adjacent[adj]]
                    self.Adjacent[adj] = [new_idx_f2 if x==adj_adj else x for x in self.Adjacent[adj]]
                    if(i_adj is not None):
                        self.Adjacent += [[i, i_adj, adj]]
                        self.Adjacent[i_adj] = [new_idx_f1 if x==i else x for x in self.Adjacent[i_adj]]
                    else: self.Adjacent += [[i, adj]] 
                    if(adj_adj is not None):
                        self.Adjacent += [[i, adj, adj_adj]]
                        self.Adjacent[adj_adj] = [new_idx_f2 if x==adj else x for x in self.Adjacent[adj_adj]]
                    else: self.Adjacent += [[i, adj]]

                    #add a new vertice
                    middlepoint_v = (Vertices[idx1,0:3]+Vertices[idx2,0:3])/2
                    AddVertices = np.hstack((middlepoint_v, new_idx_v))
                    Vertices = np.vstack((Vertices, AddVertices))
                    #add new faces
                    middlepoint_t = (tex1+tex2)/2
                    AddFaces = np.array([idx1, 0, idx3_i, 0, new_idx_v, 0, Faces[i,6]], dtype=np.int32)
                    AddFaces = np.vstack((AddFaces, np.array([idx3_adj, 0, new_idx_v, 0, idx2, 0, Faces[adj,6]], dtype=np.int32)))
                    AddTextures = np.hstack((tex1, tex3_i, middlepoint_t))
                    AddTextures = np.vstack((AddTextures, np.hstack((tex3_adj, middlepoint_t, tex2))))
                    Faces = np.vstack((Faces, AddFaces))
                    Textures = np.vstack((Textures, AddTextures))
                    #update face info
                    Faces[i,0:6] = np.array([new_idx_v, 0, idx3_i, 0, idx2, 0])
                    Faces[adj,0:6] = np.array([idx3_adj, 0, idx1, 0, new_idx_v, 0])
                    Textures[i] = np.hstack((middlepoint_t, tex3_i, tex2))
                    Textures[adj] = np.hstack((tex3_adj, tex1, middlepoint_t))

            return (Vertices, Faces, Textures)

        print "\nMax number of correspondences per triangle :", max(num_corrS_triangles)
        if(max(num_corrS_triangles)>=10 and level<0 and len(SourceFaces)/2>len(self.Avatar[0].Faces)):
            start_time = time.time()
            
            print "\nSplitting some triangles...",
            (self.Avatar[0].Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures) = SplitTriangles_2(self.Avatar[0].Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures, num_corrS_triangles)
            print "done (%f sec)" %(time.time() - start_time)
            print "Vertices", len(self.Avatar[0].Vertices)
            print "Faces", len(self.Avatar[0].Faces)

            print "Recompute Correspondences (%d time(s))" %(level + 1)

            corr = self.TriangleCorrespondence(level + 1)

        
        if(level>0): return corr
        self.Correspondences = corr
        
  
        elapsed_time = time.time() - start_time
        print "%d correspondences were found (%f sec)" %(len(corr), elapsed_time)

        '''
        #find no-corresopndence triangles
        unusedfaces = []
        Used = np.full(len(self.Avatar[0].Faces), False, dtype=bool)
        for i in xrange(len(self.Correspondences)):
            Used[self.Correspondences[i][1]] = True
        for i in xrange(len(self.Avatar[0].Faces)):
            if(Used[i]==False): 
                unusedfaces += [i]
        for i in unusedfaces:
            print "Avatar face %d has no corresponding triangle" %(i)
        print "total unused faces :", len(unusedfaces) 
        '''

        '''''''''show correspondences in tkinter'''''''''

        V = imp.load_source('show3d', APP_ROOT+'/show3d.py')
        
        root = tk.Tk()
        root.title(u"Triangle Correspondences")
        root.resizable(0, 0)

        #Avatar = Mesh_VFN(ResultVertices, AvatarFaces)
        Avatar = Mesh_VFN(self.Avatar[0].Vertices, self.Avatar[0].Faces)
        Source = Mesh_VFN(SourceVertices, SourceFaces)
        Visualize = V.Visualize(root, Avatar, Source, self.Correspondences)
        Visualize.mainloop()

        ''''''''''''''''''''''''''''''''''''''''''''''''
        #save correspondences as corr_(avatarname).txt
        if not path.exists("data/correspondences/"):
            os.makedirs("data/correspondences/")
        with open('data/correspondences/corr_%s.txt' %(self.avatarname), 'w') as f:
            for c in self.Correspondences:
                f.write("%d %d\n" %(c[0], c[1]))

        self.SaveMesh("Results/", "Splitted.ply", self.Avatar[0].Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)


        
    def DeformationTransfer(self):
        corrList = self.Correspondences
        PartsSize = np.empty([0, 3], dtype=np.float32)
        PartsRot = np.zeros([6, 3, 3], dtype = np.float32)
        AN_Faces = self.Avatar[0].Faces
        SN_Vertices = self.BlendShapes[0].Vertices
        AN_Vertices = self.Avatar[0].Vertices

        AN_Vinv = np.zeros((len(AN_Faces), 3, 3), dtype = np.float32)

        #Compute Rotation of facial parts from Source to Avatar
        def SN_Energy(args):
            RotX = np.array([[1, 0, 0], [0, np.cos(args[0]), -np.sin(args[0])], [0, np.sin(args[0]), np.cos(args[0])]], dtype=np.float32)
            RotY = np.array([[np.cos(args[1]), 0, np.sin(args[1])], [0, 1, 0], [-np.sin(args[1]), 0, np.cos(args[1])]], dtype=np.float32)
            normal = np.dot(np.dot(RotY, RotX), np.array([0, 0, 1]))
            normal = normal/LA.norm(normal)
            E = []
            for i in xrange(len(cared_landmarks)):
                for j in xrange(i,len(cared_landmarks)):
                    E += [np.dot(normal, SN_Vertices[self.SOURCE_LANDMARKS[cared_landmarks[j]],0:3]-SN_Vertices[self.SOURCE_LANDMARKS[cared_landmarks[i]],0:3])]
            return E
        def AN_Energy(args):
            RotX = np.array([[1, 0, 0], [0, np.cos(args[0]), -np.sin(args[0])], [0, np.sin(args[0]), np.cos(args[0])]], dtype=np.float32)
            RotY = np.array([[np.cos(args[1]), 0, np.sin(args[1])], [0, 1, 0], [-np.sin(args[1]), 0, np.cos(args[1])]], dtype=np.float32)
            normal = np.dot(np.dot(RotY, RotX), np.array([0, 0, 1]))
            normal = normal/LA.norm(normal)
            E = []
            for i in xrange(len(cared_landmarks)):
                for j in xrange(i,len(cared_landmarks)):
                    E += [np.dot(normal, AN_Vertices[self.AVATAR_LANDMARKS[cared_landmarks[j]],0:3]-AN_Vertices[self.AVATAR_LANDMARKS[cared_landmarks[i]],0:3])]
            return E

        #this function needs improvement
        def SN_partsnormals():
            vlist=np.zeros((len(cared_landmarks),3))
            nlist=np.zeros((len(cared_landmarks),3))
            for i in xrange(len(cared_landmarks)):
                vlist[i] = AN_Vertices[cared_landmarks[i],0:3]
            vc=np.mean(vlist,axis=0)
            vlist = vlist-vc
            for i in xrange(len(cared_landmarks)-1):
                nlist[i]=np.cross(vlist[i+1], vlist[i])
            nlist[len(cared_landmarks)-1]=np.cross(vlist[0], vlist[len(cared_landmarks)-1])
            normal = np.mean(nlist,axis=0)
            normal=normal/LA.norm(normal)
            '''
            vlist=np.zeros((len(cared_landmarks),3))
            for i in xrange(len(cared_landmarks)):
                vlist[i] = SN_Vertices[cared_landmarks[i],0:3]
            vc=np.mean(vlist,axis=0)
            A = sp.lil_matrix((len(cared_landmarks), 2), dtype=np.float32)
            b_vector = sp.lil_matrix((len(cared_landmarks), 1), dtype=np.float32)
            for i in xrange(len(cared_landmarks)):
                A[i,0:2] = vc[0:2] - SN_Vertices[cared_landmarks[i],0:2]
                b_vector[i] = SN_Vertices[cared_landmarks[i],2] - vc[2]
            (p, q) = spsolve(np.dot(A.T,A), np.dot(A.T,b_vector))
            c = 1/np.sqrt(p**2+q**2+1); a = p*c; b = q*c
            normal=np.array([a, b, c])
            '''
            angle_x = -np.arcsin(normal[1])
            cos_angle_y = normal[2]/np.cos(angle_x)
            if(cos_angle_y>1.): cos_angle_y = 1.0
            if(cos_angle_y<-1.): cos_angle_y = -1.0
            angle_y = np.arccos(cos_angle_y)
            if(normal[0]<0): angle_y = -angle_y
            return (angle_x, angle_y)

        def AN_partsnormals():
            vlist=np.zeros((len(cared_landmarks),3))
            nlist=np.zeros((len(cared_landmarks),3))
            for i in xrange(len(cared_landmarks)):
                vlist[i] = AN_Vertices[cared_landmarks[i],0:3]
            vc=np.mean(vlist,axis=0)
            vlist = vlist-vc
            for i in xrange(len(cared_landmarks)-1):
                nlist[i]=np.cross(vlist[i+1], vlist[i])
            nlist[len(cared_landmarks)-1]=np.cross(vlist[0], vlist[len(cared_landmarks)-1])
            normal = np.mean(nlist,axis=0)
            normal=normal/LA.norm(normal)

            '''
            vlist=np.zeros((len(cared_landmarks),3))
            for i in xrange(len(cared_landmarks)):
                vlist[i] = AN_Vertices[cared_landmarks[i],0:3]
            vc=np.mean(vlist,axis=0)
            A = sp.lil_matrix((len(cared_landmarks), 2), dtype=np.float32)
            b_vector = sp.lil_matrix((len(cared_landmarks), 1), dtype=np.float32)
            for i in xrange(len(cared_landmarks)):
                A[i,0:2] = vc[0:2] - AN_Vertices[cared_landmarks[i],0:2]
                b_vector[i] = AN_Vertices[cared_landmarks[i],2] - vc[2]
            (p, q) = spsolve(np.dot(A.T,A), np.dot(A.T,b_vector))
            c = 1/np.sqrt(p**2+q**2+1); a = p*c; b = q*c
            normal=np.array([a, b, c])
            '''

            print "anormal:", normal
            angle_x = -np.arcsin(normal[1])
            cos_angle_y = normal[2]/np.cos(angle_x)
            if(cos_angle_y>1.): cos_angle_y = 1.0
            if(cos_angle_y<-1.): cos_angle_y = -1.0
            angle_y = np.arccos(cos_angle_y)
            if(normal[0]<0): angle_y = -angle_y
            return (angle_x, angle_y)
            

        initial = np.array([0., 0.], dtype=np.float32)
        PartsAngles_S = np.empty([0, 2], dtype=np.float32)
        PartsAngles_A = np.empty([0, 2], dtype=np.float32)
        PartsRot_S = np.zeros([6, 3, 3], dtype = np.float32)
        PartsRot_A = np.zeros([6, 3, 3], dtype = np.float32)
        cared_landmarks_list = [[0, 1, 2, 3, 4],    #right eyebrow
                                [5, 6, 7, 8, 9],    #left eyebrow
                                [10, 14, 15, 16, 17, 18],   #nose
                                [19, 20, 21, 22, 23, 24],   #right eye
                                [25, 26, 27, 28, 29, 30],   #left eye
                                [33, 34, 35, 39, 40, 41, 44, 45, 46, 48, 49, 50]]   #mouth
        

        # this part should be modified. the acculacy is not so good.
        for ls in cared_landmarks_list:
            cared_landmarks = ls
            angles = optimize.least_squares(SN_Energy, initial, bounds=([-np.pi/2, -np.pi/2],[np.pi/2, np.pi/2])).x
            PartsAngles_S = np.vstack((PartsAngles_S, [[angles[0], angles[1]]]))
            angles = optimize.least_squares(AN_Energy, initial, bounds=([-np.pi/2, -np.pi/2],[np.pi/2, np.pi/2])).x
            PartsAngles_A = np.vstack((PartsAngles_A, [[angles[0], angles[1]]]))
        '''
        for ls in cared_landmarks_list:
            cared_landmarks = ls
            angles = SN_partsnormals()
            PartsAngles_S = np.vstack((PartsAngles_S, [[angles[0], angles[1]]]))
            angles = AN_partsnormals()
            PartsAngles_A = np.vstack((PartsAngles_A, [[angles[0], angles[1]]]))
        '''
        print "Angles x, y"
        print PartsAngles_A*180/np.pi
        

        for i in xrange(6):
            angle_x = PartsAngles_S[i,0]; angle_y = PartsAngles_S[i,1]
            RotX = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]], dtype=np.float32)
            RotY = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=np.float32)
            Rot_S = np.dot(RotY, RotX)
            angle_x = PartsAngles_A[i,0]; angle_y = PartsAngles_A[i,1]
            RotX = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]], dtype=np.float32)
            RotY = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=np.float32)
            Rot_A = np.dot(RotY, RotX)
            PartsRot[i] = np.dot(Rot_A, Rot_S.T)
            PartsRot_S[i] = Rot_S
            PartsRot_A[i] = Rot_A

        #measure facial parts size
        def Measure_Source_Parts_Size(lower, upper, partsnum):
            xs=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[lower],0:3])[0]; xl=xs
            ys=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[lower],0:3])[1]; yl=ys
            zs=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[lower],0:3])[2]; zl=zs

            for i in xrange(lower, upper+1):
                if(np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[0]<xs):
                    xs=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[0]
                elif(xl<np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[0]):
                    xl=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[0]
                if(np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[1]<ys):
                    ys=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[1]
                elif(yl<np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[1]):
                    yl=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[1]
                if(np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[2]<zs):
                    zs=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[2]
                elif(zl<np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[2]):
                    zl=np.dot(PartsRot_S[partsnum], SN_Vertices[self.SOURCE_LANDMARKS[i],0:3])[2]
            size_x = xl-xs
            size_y = yl-ys
            size_z = zl-zs
            return (size_x, size_y, size_z)

        (x, y, z) = Measure_Source_Parts_Size(0, 4, 0)
        S_PartsSize = np.array([[x, y, z]]) ;print "Right eyebrow: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Source_Parts_Size(5, 9, 1)
        S_PartsSize = np.vstack((S_PartsSize, [[x, y, z]])) ;print "Left eyebrow: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Source_Parts_Size(10, 18, 2)
        S_PartsSize = np.vstack((S_PartsSize, [[x, y, z]])) ;print "Nose: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Source_Parts_Size(19, 24, 3)
        S_PartsSize = np.vstack((S_PartsSize, [[x, y, z]])) ;print "Right eye: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Source_Parts_Size(25, 30, 4)
        S_PartsSize = np.vstack((S_PartsSize, [[x, y, z]])) ;print "Left eye: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Source_Parts_Size(31, 50, 5)
        y1 = LA.norm(SN_Vertices[self.SOURCE_LANDMARKS[15],0:3]-SN_Vertices[self.SOURCE_LANDMARKS[44],0:3])
        y2 = LA.norm(SN_Vertices[self.SOURCE_LANDMARKS[16],0:3]-SN_Vertices[self.SOURCE_LANDMARKS[45],0:3])
        y3 = LA.norm(SN_Vertices[self.SOURCE_LANDMARKS[17],0:3]-SN_Vertices[self.SOURCE_LANDMARKS[46],0:3])
        y = (y1+y2+y3)/3 
        S_PartsSize = np.vstack((S_PartsSize, [[x, y, z]])) ;print "Mouth: x: %f, y: %f, z: %f" %(x, y, z)



        def Measure_Avatar_Parts_Size(lower, upper, partsnum):
            xs=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[lower],0:3])[0]; xl=xs
            ys=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[lower],0:3])[1]; yl=ys
            zs=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[lower],0:3])[2]; zl=zs

            for i in xrange(lower, upper+1):
                if(np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[0]<xs):
                    xs=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[0]
                elif(xl<np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[0]):
                    xl=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[0]
                if(np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[1]<ys):
                    ys=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[1]
                elif(yl<np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[1]):
                    yl=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[1]
                if(np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[2]<zs):
                    zs=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[2]
                elif(zl<np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[2]):
                    zl=np.dot(PartsRot_A[partsnum], AN_Vertices[self.AVATAR_LANDMARKS[i],0:3])[2]
            size_x = xl-xs
            size_y = yl-ys
            size_z = zl-zs
            return (size_x, size_y, size_z)


        (x, y, z) = Measure_Avatar_Parts_Size(0, 4, 0)
        A_PartsSize = np.array([[x, y, z]]) ;print "Right eyebrow: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Avatar_Parts_Size(5, 9, 1)
        A_PartsSize = np.vstack((A_PartsSize, [[x, y, z]])) ;print "Left eyebrow: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Avatar_Parts_Size(10, 18, 2)
        A_PartsSize = np.vstack((A_PartsSize, [[x, y, z]])) ;print "Nose: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Avatar_Parts_Size(19, 24, 3)
        A_PartsSize = np.vstack((A_PartsSize, [[x, y, z]])) ;print "Right eye: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Avatar_Parts_Size(25, 30, 4)
        A_PartsSize = np.vstack((A_PartsSize, [[x, y, z]])) ;print "Left eye: x: %f, y: %f, z: %f" %(x, y, z)
        (x, y, z) = Measure_Avatar_Parts_Size(31, 50, 5)
        y1 = LA.norm(AN_Vertices[self.AVATAR_LANDMARKS[15],0:3]-AN_Vertices[self.AVATAR_LANDMARKS[44],0:3])
        y2 = LA.norm(AN_Vertices[self.AVATAR_LANDMARKS[16],0:3]-AN_Vertices[self.AVATAR_LANDMARKS[45],0:3])
        y3 = LA.norm(AN_Vertices[self.AVATAR_LANDMARKS[17],0:3]-AN_Vertices[self.AVATAR_LANDMARKS[46],0:3])
        y = (y1+y2+y3)/3
        z = S_PartsSize[5][2]
        A_PartsSize = np.vstack((A_PartsSize, [[x, y, z]])) ;print "Mouth: x: %f, y: %f, z: %f" %(x, y, z)


        for i in xrange(len(A_PartsSize)):
            size = A_PartsSize[i]/S_PartsSize[i]
            PartsSize = np.vstack((PartsSize, size))
       

        print PartsSize

        #compute Avatar Neutral Vinv
        for i in xrange(len(AN_Faces)):
            a = np.array([AN_Vertices[AN_Faces[i,2],0:3] - AN_Vertices[AN_Faces[i,0],0:3]]) #v2-v1
            b = np.array([AN_Vertices[AN_Faces[i,4],0:3] - AN_Vertices[AN_Faces[i,0],0:3]]) #v3-v1
            cross = np.cross(a, b)
            c = cross / sqrt(LA.norm(cross)) #v4-v1

            AN_Vinv[i] = LA.solve(np.hstack([a.T, b.T, c.T]), np.identity(3)) #Avatar Neutral Vinv

        #Make Deformation Transfer matrix
        def MakeEd_A():
            Ed_A = sp.lil_matrix((len(corrList)*9, len(AN_Vertices)*3 + len(AN_Faces)*3), dtype=np.float32)

            for i in xrange(len(corrList)):
                idx_f = corrList[i][1]
                e1 = np.sum(AN_Vinv[idx_f,0:3,0])
                e2 = np.sum(AN_Vinv[idx_f,0:3,1])
                e3 = np.sum(AN_Vinv[idx_f,0:3,2])

                Ed_A[i*9, AN_Faces[idx_f,0]*3]     = -e1
                Ed_A[i*9+1, AN_Faces[idx_f,0]*3+1] = -e1
                Ed_A[i*9+2, AN_Faces[idx_f,0]*3+2] = -e1
                Ed_A[i*9+3, AN_Faces[idx_f,0]*3]   = -e2
                Ed_A[i*9+4, AN_Faces[idx_f,0]*3+1] = -e2
                Ed_A[i*9+5, AN_Faces[idx_f,0]*3+2] = -e2
                Ed_A[i*9+6, AN_Faces[idx_f,0]*3]   = -e3
                Ed_A[i*9+7, AN_Faces[idx_f,0]*3+1] = -e3
                Ed_A[i*9+8, AN_Faces[idx_f,0]*3+2] = -e3

                Ed_A[i*9, AN_Faces[idx_f,2]*3]     = AN_Vinv[idx_f,0,0]
                Ed_A[i*9+1, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,0,0]
                Ed_A[i*9+2, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,0,0]
                Ed_A[i*9+3, AN_Faces[idx_f,2]*3]   = AN_Vinv[idx_f,0,1]
                Ed_A[i*9+4, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,0,1]
                Ed_A[i*9+5, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,0,1]
                Ed_A[i*9+6, AN_Faces[idx_f,2]*3]   = AN_Vinv[idx_f,0,2]
                Ed_A[i*9+7, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,0,2]
                Ed_A[i*9+8, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,0,2]

                Ed_A[i*9, AN_Faces[idx_f,4]*3]     = AN_Vinv[idx_f,1,0]
                Ed_A[i*9+1, AN_Faces[idx_f,4]*3+1] = AN_Vinv[idx_f,1,0]
                Ed_A[i*9+2, AN_Faces[idx_f,4]*3+2] = AN_Vinv[idx_f,1,0]
                Ed_A[i*9+3, AN_Faces[idx_f,4]*3]   = AN_Vinv[idx_f,1,1]
                Ed_A[i*9+4, AN_Faces[idx_f,4]*3+1] = AN_Vinv[idx_f,1,1]
                Ed_A[i*9+5, AN_Faces[idx_f,4]*3+2] = AN_Vinv[idx_f,1,1]
                Ed_A[i*9+6, AN_Faces[idx_f,4]*3]   = AN_Vinv[idx_f,1,2]
                Ed_A[i*9+7, AN_Faces[idx_f,4]*3+1] = AN_Vinv[idx_f,1,2]
                Ed_A[i*9+8, AN_Faces[idx_f,4]*3+2] = AN_Vinv[idx_f,1,2]

                Ed_A[i*9, idx_f*3+len(AN_Vertices)*3]     = AN_Vinv[idx_f,2,0]
                Ed_A[i*9+1, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,0]
                Ed_A[i*9+2, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,0]
                Ed_A[i*9+3, idx_f*3+len(AN_Vertices)*3]   = AN_Vinv[idx_f,2,1]
                Ed_A[i*9+4, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,1]
                Ed_A[i*9+5, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,1]
                Ed_A[i*9+6, idx_f*3+len(AN_Vertices)*3]   = AN_Vinv[idx_f,2,2]
                Ed_A[i*9+7, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,2]
                Ed_A[i*9+8, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,2]
            
            return Ed_A
            

        def MakeEi_ATA_ATc():
            Flat_identity = sp.csc_matrix((0, 1), dtype=np.float32)
            for i in xrange(len(AN_Faces)):
                Flat_identity = sp.vstack((Flat_identity, sp.identity(3, format="lil").reshape((9, 1))), format="csc", dtype=np.float32)
            Ei_cVector = Flat_identity
            Ei_A = sp.lil_matrix((len(AN_Faces)*9, len(AN_Vertices)*3 + len(AN_Faces)*3), dtype=np.float32)

            for i in xrange(len(AN_Faces)):
                idx_f = i
                e1 = np.sum(AN_Vinv[idx_f,0:3,0])
                e2 = np.sum(AN_Vinv[idx_f,0:3,1])
                e3 = np.sum(AN_Vinv[idx_f,0:3,2])

                Ei_A[i*9, AN_Faces[idx_f,0]*3]     = -e1
                Ei_A[i*9+1, AN_Faces[idx_f,0]*3+1] = -e1
                Ei_A[i*9+2, AN_Faces[idx_f,0]*3+2] = -e1
                Ei_A[i*9+3, AN_Faces[idx_f,0]*3]   = -e2
                Ei_A[i*9+4, AN_Faces[idx_f,0]*3+1] = -e2
                Ei_A[i*9+5, AN_Faces[idx_f,0]*3+2] = -e2
                Ei_A[i*9+6, AN_Faces[idx_f,0]*3]   = -e3
                Ei_A[i*9+7, AN_Faces[idx_f,0]*3+1] = -e3
                Ei_A[i*9+8, AN_Faces[idx_f,0]*3+2] = -e3

                Ei_A[i*9, AN_Faces[idx_f,2]*3]     = AN_Vinv[idx_f,0,0]
                Ei_A[i*9+1, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,0,0]
                Ei_A[i*9+2, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,0,0]
                Ei_A[i*9+3, AN_Faces[idx_f,2]*3]   = AN_Vinv[idx_f,0,1]
                Ei_A[i*9+4, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,0,1]
                Ei_A[i*9+5, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,0,1]
                Ei_A[i*9+6, AN_Faces[idx_f,2]*3]   = AN_Vinv[idx_f,0,2]
                Ei_A[i*9+7, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,0,2]
                Ei_A[i*9+8, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,0,2]

                Ei_A[i*9, AN_Faces[idx_f,4]*3]     = AN_Vinv[idx_f,1,0]
                Ei_A[i*9+1, AN_Faces[idx_f,4]*3+1] = AN_Vinv[idx_f,1,0]
                Ei_A[i*9+2, AN_Faces[idx_f,4]*3+2] = AN_Vinv[idx_f,1,0]
                Ei_A[i*9+3, AN_Faces[idx_f,4]*3]   = AN_Vinv[idx_f,1,1]
                Ei_A[i*9+4, AN_Faces[idx_f,4]*3+1] = AN_Vinv[idx_f,1,1]
                Ei_A[i*9+5, AN_Faces[idx_f,4]*3+2] = AN_Vinv[idx_f,1,1]
                Ei_A[i*9+6, AN_Faces[idx_f,4]*3]   = AN_Vinv[idx_f,1,2]
                Ei_A[i*9+7, AN_Faces[idx_f,4]*3+1] = AN_Vinv[idx_f,1,2]
                Ei_A[i*9+8, AN_Faces[idx_f,4]*3+2] = AN_Vinv[idx_f,1,2]

                Ei_A[i*9, idx_f*3+len(AN_Vertices)*3]     = AN_Vinv[idx_f,2,0]
                Ei_A[i*9+1, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,0]
                Ei_A[i*9+2, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,0]
                Ei_A[i*9+3, idx_f*3+len(AN_Vertices)*3]   = AN_Vinv[idx_f,2,1]
                Ei_A[i*9+4, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,1]
                Ei_A[i*9+5, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,1]
                Ei_A[i*9+6, idx_f*3+len(AN_Vertices)*3]   = AN_Vinv[idx_f,2,2]
                Ei_A[i*9+7, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,2]
                Ei_A[i*9+8, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,2]

            return (np.dot(Ei_A.T, Ei_A), np.dot(Ei_A.T, Ei_cVector))

        print "Generating deformation transfer matrices..."
        start_time = time.time()
        Ed_A = MakeEd_A()
        Ed_ATA = np.dot(Ed_A.T, Ed_A)
        elapsed_time = time.time() - start_time; print "Ed_A, Ed_ATA :", elapsed_time


        #find unused points
        unusedpoints = []
        Used = np.full(len(AN_Vertices), False, dtype=bool)
        for i in xrange(len(AN_Faces)):
            Used[AN_Faces[i,0]] = True
            Used[AN_Faces[i,2]] = True
            Used[AN_Faces[i,4]] = True
        for i in xrange(len(AN_Vertices)):
            if(Used[i] ==False):
                unusedpoints += [i]
        print "Unused points :", unusedpoints

        i = 0
        currMesh = dtf(self.Correspondences, self.SOURCE_LANDMARKS, self.AVATAR_LANDMARKS, self.BlendShapes[0], self.BlendShapes[0], self.Avatar[0], PartsSize, PartsRot, Ed_A, Ed_ATA, self.Es_ATA, self.Es_ATc, self.Ei_ATA, self.Ei_ATc, self.backlandmark_list)
        print "\n\nGenerating Neutral AvatarMesh"
        currMesh.deformation_transfer()
        self.SaveMesh("Results/", "AvatarMesh_Neutral.ply", currMesh.Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)

        for meshnum in xrange(48):
            if ((meshnum > 1 and meshnum < 14) or meshnum == 17 or meshnum == 18 or meshnum == 19 or meshnum == 22 or meshnum == 26 
            or meshnum == 27 or meshnum == 34 or meshnum == 35 or meshnum == 42):
                continue
            else:
                currMesh = dtf(self.Correspondences, self.SOURCE_LANDMARKS, self.AVATAR_LANDMARKS, self.BlendShapes[i+1], self.BlendShapes[0], self.Avatar[0], PartsSize, PartsRot, Ed_A, Ed_ATA, self.Es_ATA, self.Es_ATc, self.Ei_ATA, self.Ei_ATc, self.backlandmark_list)
            print "\n\nGenerating Avatar blendshape ", meshnum
            currMesh.deformation_transfer()
            self.Avatar.append(currMesh)
            name = "AvatarMesh_%d.ply" %(meshnum)
            self.SaveMesh("Results/", name, self.Avatar[i+1].Vertices, self.Avatar[0].Faces, self.Avatar[0].Textures, self.Avatar[0].TexName, self.Avatar[0].Flags)
            i += 1

        print "\n%d blendshapes are generated\n" %(len(self.Avatar))
