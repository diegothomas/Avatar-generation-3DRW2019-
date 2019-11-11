#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:24:42 2017

@author: diegothomas
"""

#Input:
#   depth=input depth image
#   depth_out= output depth image (after filtering)
#   sigma_r= sigma for the std distance in depth    
#   sigma_d= sigma for the std distance in pixels
#   d=diameter of each pixel neighborhood
#   nbLines=number of lines in the depth image
#   nbColumns=number of columns in the depth image
#Output:
#   Perform bilateral filter of the input depth image
Kernel_BilateralFilter = """
    // i is the global index in the depth image
    int idx_i = i/nbColumns;    // line index
    int idx_j = i%nbColumns;    // column index
    
    int max_i = min(idx_i+d+1, nbLines);
    int min_i = max(idx_i-d, 0);
    int max_j = min(idx_j+d+1, nbColumns);
    int min_j = max(idx_j-d, 0);
    
    float res_val = 0.0f;
    float depth_ref = depth[i];
    float weight = 0.0f;
    if (depth_ref > 0.0f) {
        for (int k = min_i; k < max_i; k++) {
            for (int l = min_j; l < max_j; l++) {
                if (depth[l + k*nbColumns] > 0.0f && fabs(depth_ref - depth[l + k*nbColumns]) < 0.1f) {
                    res_val = res_val + depth[l + k*nbColumns]*exp(-(0.5f*pown(fabs(depth_ref - depth[l + k*nbColumns])/sigma_r, 2) +
                                        0.5*pown(sqrt((float)((k-idx_i)*(k-idx_i) + (l-idx_j)*(l-idx_j)))/sigma_d,2)));
                    
                    weight = weight + exp(-(0.5f*pown(fabs(depth_ref - depth[l + k*nbColumns])/sigma_r,2) +
                                        0.5*pown(sqrt((float)((k-idx_i)*(k-idx_i) + (l-idx_j)*(l-idx_j)))/sigma_d,2)));
                }
            }
        }
    }
        
    depth_out[i] = weight > 0.0f ? res_val/weight : depth_ref;
"""
                 
# Input
#   depth= input depth image
#   vmap= output vertex image (x,y,z) coordinates in each pixel
#   intrinsic= intrinsic matrix of the depth camera
#   nbColumns = number of columns in the depth image
# Output:
#   Compute the vertex image from the depth image
Kernel_VMap = """
    // i is the global index in the depth image
    vmap[3*i] = depth[i] > 0.0f ? depth[i]*((float)(i%nbColumns) - intrinsic[2])/intrinsic[0] : 0.0f; 
    vmap[3*i+1] = depth[i] > 0.0f ? depth[i]*((float)(i/nbColumns) - intrinsic[3])/intrinsic[1]: 0.0f;
    vmap[3*i+2] = depth[i] > 0.0f ? depth[i] : 0.0f
"""
                 
# Input:
#   vmap = input vertex image
#   nmap = output normal image
#   nbLines=number of lines in the depth image
#   nbColumns=number of columns in the depth image
# Output:
#   Compute the normal vector of each vertex in the vertex image
Kernel_NMap = """
// i is the global index in the vertex image (of size(n*m*3))
if (i%3 == 0) {
    int idx_i = (i/3)/nbColumns;    // line index
    int idx_j = (i/3)%nbColumns;    // column index
    
    if (idx_i == 0 || idx_i == nbLines-1 || idx_j == 0 || idx_j == nbColumns-1 || vmap[i + 2] == 0.0f) {
        nmap[i] = 0.0f;
        nmap[i+1] = 0.0f;
        nmap[i+2] = 0.0f;
    } else {
        float3 v0 = (float3){vmap[i], vmap[i+1], vmap[i+2]};
        
        int i1 = idx_j+1 + idx_i*nbColumns;
        int i2 = idx_j + (idx_i-1)*nbColumns;
        int i3 = idx_j-1 + idx_i*nbColumns;
        int i4 = idx_j + (idx_i+1)*nbColumns;
        
        float3 v1 = (float3){vmap[3*i1], vmap[3*i1+1], vmap[3*i1+2]};
        float3 v2 = (float3){vmap[3*i2], vmap[3*i2+1], vmap[3*i2+2]};
        float3 v3 = (float3){vmap[3*i3], vmap[3*i3+1], vmap[3*i3+2]};
        float3 v4 = (float3){vmap[3*i4], vmap[3*i4+1], vmap[3*i4+2]};
        
        float3 nmle1 = v1.z == 0.0f || v2.z == 0.0f ? 0.0f : cross(v1-v0,v2-v0);
        float3 nmle2 = v2.z == 0.0f || v3.z == 0.0f ? 0.0f : cross(v2-v0,v3-v0);
        float3 nmle3 = v3.z == 0.0f || v4.z == 0.0f ? 0.0f : cross(v3-v0,v4-v0);
        float3 nmle4 = v4.z == 0.0f || v1.z == 0.0f ? 0.0f : cross(v4-v0,v1-v0);
        
        float div_f = (float)(v1.z != 0.0f) + (float)(v2.z != 0.0f) +
                        (float)(v3.z != 0.0f) +(float)(v4.z != 0.0f) ;
        
        if (div_f > 0.0f) {
            float3 nmle = (nmle1 + nmle2 + nmle3 + nmle4)/div_f;
            nmle = normalize(nmle);
            nmap[i] = nmle.x;
            nmap[i+1] = nmle.y;
            nmap[i+2] = nmle.z;
        } else {
            nmap[i] = 0.0f;
            nmap[i+1] = 0.0f;
            nmap[i+2] = 0.0f;
        }
    }
}"""

# Input:
#   res = rendered image
#   vmap = input vertex image
#   nmap = output normal image
#   Pose = extrinsic matrix of the current camera
#   Intrinsic = intrinsic matrix of the camera
#   color_flag = color mode (normal or RGB)
#   nbLines=number of lines in the depth image
#   nbColumns=number of columns in the depth image
# Output:
#   Render the vertex image with color or normal vertices into the current camera image plane
Kernel_Draw = """
// i is the global index in the res image (of size(n*m*3))
if (i%3 == 0) {
        float3 pt = (float3) {vmap[i], vmap[i+1], vmap[i+2]};

    if (pt.z != 0.0f) {
        float3 nmle = (float3) {nmap[i], nmap[i+1], nmap[i+2]};
        
        float3 pt_T = (float3) {pt.x*Pose[0] + pt.y*Pose[1] + pt.z*Pose[2] + Pose[3],
                                pt.x*Pose[4] + pt.y*Pose[5] + pt.z*Pose[6] + Pose[7], 
                                pt.x*Pose[8] + pt.y*Pose[9] + pt.z*Pose[10] + Pose[11]};
        float3 nmle_T = (float3) {nmle.x*Pose[0] + nmle.y*Pose[1] + nmle.z*Pose[2], 
                                  nmle.x*Pose[4] + nmle.y*Pose[5] + nmle.z*Pose[6],
                                  nmle.x*Pose[8] + nmle.y*Pose[9] + nmle.z*Pose[10]};
        
        if (pt_T.z != 0.0f) {
            // pix = (line index, column index)
            int2 pix = (int2){convert_int(round(Intrinsic[1]*pt_T.y/fabs(pt_T.z) + Intrinsic[3])), 
                              convert_int(round(Intrinsic[0]*pt_T.x/fabs(pt_T.z) + Intrinsic[2]))};
            if (pix.x > -1 && pix.x < nbLines && pix.y > -1 && pix.y < nbColumns ) {
                if (color_flag == 0) {
                    atomic_xchg(&res[3*(int)(pix.y + pix.x*nbColumns)], (unsigned int)(color[i+2])); 
                    atomic_xchg(&res[3*(int)(pix.y + pix.x*nbColumns)+1], (unsigned int)(color[i+1])); 
                    atomic_xchg(&res[3*(int)(pix.y + pix.x*nbColumns)+2], (unsigned int)(color[i])); 
                } else { 
                    if (color_flag == 1) {
                        atomic_xchg(&res[3*(pix.y + pix.x*nbColumns)], (unsigned int)(pt_T.z*100.0f)); 
                        atomic_xchg(&res[3*(pix.y + pix.x*nbColumns)+1],(unsigned int)(pt_T.z*100.0f)); 
                        atomic_xchg(&res[3*(pix.y + pix.x*nbColumns)+2], (unsigned int)(pt_T.z*100.0f)); 
                    } else {
                        atomic_xchg(&res[3*(pix.y + pix.x*nbColumns)], (unsigned int)((nmle_T.x + 1.0f)*(255.0f/2.0f))); 
                        atomic_xchg(&res[3*(pix.y + pix.x*nbColumns)+1],(unsigned int)((nmle_T.y + 1.0f)*(255.0f/2.0f))); 
                        atomic_xchg(&res[3*(pix.y + pix.x*nbColumns)+2], (unsigned int)((nmle_T.z + 1.0f)*(255.0f/2.0f))); 
                    }
                }
            }       
        }
    }
}
"""

Kernel_Reproj = """
// i is the global index 
    float3 pt = (float3) {vmap[3*i], vmap[3*i+1], vmap[3*i+2]};

    if (pt.z != 0.0f) {
        
        float3 pt_T = (float3) {pt.x*Pose[0] + pt.y*Pose[1] + pt.z*Pose[2] + Pose[3],
                                pt.x*Pose[4] + pt.y*Pose[5] + pt.z*Pose[6] + Pose[7], 
                                pt.x*Pose[8] + pt.y*Pose[9] + pt.z*Pose[10] + Pose[11]};
        
        if (pt_T.z != 0.0f) {
            // pix = (line index, column index)
            int2 pix = (int2){convert_int(round(Intrinsic[1]*pt_T.y/fabs(pt_T.z) + Intrinsic[3])), 
                              convert_int(round(Intrinsic[0]*pt_T.x/fabs(pt_T.z) + Intrinsic[2]))};
            if (pix.x > -1 && pix.x < nbLines && pix.y > -1 && pix.y < nbColumns ) {
                RGB2Depth[(pix.y + pix.x*nbColumns)] = i; 
                color[3*i] = color_buff[3*(pix.y + pix.x*nbColumns)]; 
                color[3*i+1] = color_buff[3*(pix.y + pix.x*nbColumns)+1]; 
                color[3*i+2] = color_buff[3*(pix.y + pix.x*nbColumns)+2]; 
            }       
        }
    }
"""

# Input:
#   vmap = input vertex image
#   nmap = output normal image
#   Pose = extrinsic matrix of the current camera
# Output:
#   Transforms the vertex image 
Kernel_Transform = """
// i is the global index in the res image (of size(n*m*3))
if (i%3 == 0) {
    float3 pt = (float3) {vmap[i], vmap[i+1], vmap[i+2]};

    if (pt.z != 0.0f) {
        float3 nmle = (float3) {nmap[i], nmap[i+1], nmap[i+2]};
        
        float3 pt_T = (float3) {pt.x*Pose[0] + pt.y*Pose[1] + pt.z*Pose[2] + Pose[3],
                                pt.x*Pose[4] + pt.y*Pose[5] + pt.z*Pose[6] + Pose[7], 
                                pt.x*Pose[8] + pt.y*Pose[9] + pt.z*Pose[10] + Pose[11]};
        float3 nmle_T = (float3) {nmle.x*Pose[0] + nmle.y*Pose[1] + nmle.z*Pose[2], 
                                  nmle.x*Pose[4] + nmle.y*Pose[5] + nmle.z*Pose[6],
                                  nmle.x*Pose[8] + nmle.y*Pose[9] + nmle.z*Pose[10]};
        
        vmap[i] = pt_T.x;
        vmap[i+1] = pt_T.y;
        vmap[i+2] = pt_T.z;
        nmap[i] = nmle_T.x;
        nmap[i+1] = nmle_T.y;
        nmap[i+2] = nmle_T.z;
    }
}        
"""