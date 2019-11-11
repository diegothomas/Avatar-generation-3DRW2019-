#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:24:42 2017

@author: diegothomas
"""

Kernel_GICP = """
if (i%3 == 0) {
    // Transform current 3D position and normal with current transformation
    float3 pt = (float3) {vmap[i], vmap[i+1], vmap[i+2]};
    float3 nmle = (float3) {nmap[i], nmap[i+1], nmap[i+2]};
    
    if (pt.z > 0.0f /*&& (fabs(nmle.x)+fabs(nmle.y)+fabs(nmle.z) != 0.0f)*/){
        
        float3 pt_T = (float3) {pt.x*Pose[0] + pt.y*Pose[1] + pt.z*Pose[2] + Pose[3],
                                pt.x*Pose[4] + pt.y*Pose[5] + pt.z*Pose[6] + Pose[7], 
                                pt.x*Pose[8] + pt.y*Pose[9] + pt.z*Pose[10] + Pose[11]};
        float3 nmle_T = (float3) {nmle.x*Pose[0] + nmle.y*Pose[1] + nmle.z*Pose[2], 
                                  nmle.x*Pose[4] + nmle.y*Pose[5] + nmle.z*Pose[6],
                                  nmle.x*Pose[8] + nmle.y*Pose[9] + nmle.z*Pose[10]};
        
        float norme_nmle = nmle_T.x*nmle_T.x + nmle_T.y*nmle_T.y + nmle_T.z*nmle_T.z;
        if (norme_nmle > 0.0f) {
        
            // Project onto Image2
            // pix = (line index, column index)
            int2 pix = (int2){convert_int(round(Intrinsic[1]*pt_T.y/fabs(pt_T.z) + Intrinsic[3])), 
                                  convert_int(round(Intrinsic[0]*pt_T.x/fabs(pt_T.z) + Intrinsic[2]))};
            
            if (pix.x > -1 && pix.x < nbLines && pix.y > -1 && pix.y < nbColumns ) {
                int idx_match = 3*(pix.y + pix.x*nbColumns);
                // Compute distance betwn matches and btwn normals
                float3 match_vtx = (float3) {vmap2[idx_match], vmap2[idx_match+1], vmap2[idx_match+2]}; 
                float3 match_nmle = (float3) {nmap2[idx_match], nmap2[idx_match+1], nmap2[idx_match+2]}; 
                
                float distance_v = (pt_T.x - match_vtx.x)*(pt_T.x - match_vtx.x) 
                                + (pt_T.y - match_vtx.y)*(pt_T.y - match_vtx.y) 
                                + (pt_T.z - match_vtx.z)*(pt_T.z - match_vtx.z);
                
                float distance_n = (nmle_T.x - match_nmle.x)*(nmle_T.x - match_nmle.x) 
                                + (nmle_T.y - match_nmle.y)*(nmle_T.y - match_nmle.y) 
                                + (nmle_T.z - match_nmle.z)*(nmle_T.z - match_nmle.z);
                
                if (distance_v < thresh_dis && distance_n < thresh_norm) {    
                    float w = 1.0f;
                    // Complete Jacobian matrix
                    Buffer_1[i/3] = w*nmle_T.x;
                    Buffer_2[i/3] = w*nmle_T.y;
                    Buffer_3[i/3] = w*nmle_T.z;
                    Buffer_4[i/3] = w*(-match_vtx.z*nmle_T.y + match_vtx.y*nmle_T.z);
                    Buffer_5[i/3] = w*(match_vtx.z*nmle_T.x - match_vtx.x*nmle_T.z);
                    Buffer_6[i/3] = w*(-match_vtx.y*nmle_T.x + match_vtx.x*nmle_T.y);
                    Buffer_B[i/3] = w*(nmle_T.x*(match_vtx.x - pt_T.x) + 
                               nmle_T.y*(match_vtx.y - pt_T.y) + 
                               nmle_T.z*(match_vtx.z - pt_T.z));
                } else {
                    // buffer = 0
                    Buffer_1[i/3] = 0.f;
                    Buffer_2[i/3] = 0.f;
                    Buffer_3[i/3] = 0.f;
                    Buffer_4[i/3] = 0.f;
                    Buffer_5[i/3] = 0.f;
                    Buffer_6[i/3] = 0.f;
                    Buffer_B[i/3] = 0.f;
                }
            } else {
                //buffer = 0
                Buffer_1[i/3] = 0.f;
                Buffer_2[i/3] = 0.f;
                Buffer_3[i/3] = 0.f;
                Buffer_4[i/3] = 0.f;
                Buffer_5[i/3] = 0.f;
                Buffer_6[i/3] = 0.f;
                Buffer_B[i/3] = 0.f;
            }
        } else {
            // buffer =0
            Buffer_1[i/3] = 0.f;
            Buffer_2[i/3] = 0.f;
            Buffer_3[i/3] = 0.f;
            Buffer_4[i/3] = 0.f;
            Buffer_5[i/3] = 0.f;
            Buffer_6[i/3] = 0.f;
            Buffer_B[i/3] = 0.f;
        }
    } else {
        Buffer_1[i/3] = 0.f;
        Buffer_2[i/3] = 0.f;
        Buffer_3[i/3] = 0.f;
        Buffer_4[i/3] = 0.f;
        Buffer_5[i/3] = 0.f;
        Buffer_6[i/3] = 0.f;
        Buffer_B[i/3] = 0.f;
    }
}                
"""

Kernel_GICP_M2Depth = """
if (i%3 == 0 && i*fact < nbVertices) {
    // Transform current 3D position and normal with current transformation
    float3 pt = (float3) {Vertices[i*fact], Vertices[i*fact+1], Vertices[i*fact+2]};
    float3 nmle = (float3) {Normales[i*fact], Normales[i*fact+1], Normales[i*fact+2]};
    
    if (pt.z > 0.0f /*&& (fabs(nmle.x)+fabs(nmle.y)+fabs(nmle.z) != 0.0f)*/){
        
        float3 pt_T = (float3) {pt.x*Pose[0] + pt.y*Pose[1] + pt.z*Pose[2] + Pose[3],
                                pt.x*Pose[4] + pt.y*Pose[5] + pt.z*Pose[6] + Pose[7], 
                                pt.x*Pose[8] + pt.y*Pose[9] + pt.z*Pose[10] + Pose[11]};
        float3 nmle_T = (float3) {nmle.x*Pose[0] + nmle.y*Pose[1] + nmle.z*Pose[2], 
                                  nmle.x*Pose[4] + nmle.y*Pose[5] + nmle.z*Pose[6],
                                  nmle.x*Pose[8] + nmle.y*Pose[9] + nmle.z*Pose[10]};
        
        float norme_nmle = nmle_T.x*nmle_T.x + nmle_T.y*nmle_T.y + nmle_T.z*nmle_T.z;
        if (norme_nmle > 0.0f) {
        
            // Project onto Image2
            // pix = (line index, column index)
            int2 pix = (int2){convert_int(round(Intrinsic[1]*pt_T.y/fabs(pt_T.z) + Intrinsic[3])), 
                                  convert_int(round(Intrinsic[0]*pt_T.x/fabs(pt_T.z) + Intrinsic[2]))};
            
            if (pix.x > -1 && pix.x < nbLines && pix.y > -1 && pix.y < nbColumns ) {
                int idx_match = 3*(pix.y + pix.x*nbColumns);
                // Compute distance betwn matches and btwn normals
                float3 match_vtx = (float3) {vmap2[idx_match], vmap2[idx_match+1], vmap2[idx_match+2]}; 
                float3 match_nmle = (float3) {nmap2[idx_match], nmap2[idx_match+1], nmap2[idx_match+2]}; 
                
                float distance_v = (pt_T.x - match_vtx.x)*(pt_T.x - match_vtx.x) 
                                + (pt_T.y - match_vtx.y)*(pt_T.y - match_vtx.y) 
                                + (pt_T.z - match_vtx.z)*(pt_T.z - match_vtx.z);
                
                /*float distance_n = (nmle_T.x - match_nmle.x)*(nmle_T.x - match_nmle.x) 
                                + (nmle_T.y - match_nmle.y)*(nmle_T.y - match_nmle.y) 
                                + (nmle_T.z - match_nmle.z)*(nmle_T.z - match_nmle.z);*/
                
                float distance_n = nmle_T.x*match_nmle.x + nmle_T.y*match_nmle.y + nmle_T.z*match_nmle.z;
                
                if (distance_v < thresh_dis && distance_n > cos(thresh_norm)) {    
                    float w = 1.0f;
                    // Complete Jacobian matrix
                    Buffer_1[i/3] = w*nmle_T.x;
                    Buffer_2[i/3] = w*nmle_T.y;
                    Buffer_3[i/3] = w*nmle_T.z;
                    Buffer_4[i/3] = w*(-match_vtx.z*nmle_T.y + match_vtx.y*nmle_T.z);
                    Buffer_5[i/3] = w*(match_vtx.z*nmle_T.x - match_vtx.x*nmle_T.z);
                    Buffer_6[i/3] = w*(-match_vtx.y*nmle_T.x + match_vtx.x*nmle_T.y);
                    Buffer_B[i/3] = w*(nmle_T.x*(match_vtx.x - pt_T.x) + 
                               nmle_T.y*(match_vtx.y - pt_T.y) + 
                               nmle_T.z*(match_vtx.z - pt_T.z));
                } else {
                    // buffer = 0
                    Buffer_1[i/3] = 0.f;
                    Buffer_2[i/3] = 0.f;
                    Buffer_3[i/3] = 0.f;
                    Buffer_4[i/3] = 0.f;
                    Buffer_5[i/3] = 0.f;
                    Buffer_6[i/3] = 0.f;
                    Buffer_B[i/3] = 0.f;
                }
            } else {
                //buffer = 0
                Buffer_1[i/3] = 0.f;
                Buffer_2[i/3] = 0.f;
                Buffer_3[i/3] = 0.f;
                Buffer_4[i/3] = 0.f;
                Buffer_5[i/3] = 0.f;
                Buffer_6[i/3] = 0.f;
                Buffer_B[i/3] = 0.f;
            }
        } else {
            // buffer =0
            Buffer_1[i/3] = 0.f;
            Buffer_2[i/3] = 0.f;
            Buffer_3[i/3] = 0.f;
            Buffer_4[i/3] = 0.f;
            Buffer_5[i/3] = 0.f;
            Buffer_6[i/3] = 0.f;
            Buffer_B[i/3] = 0.f;
        }
    } else {
        Buffer_1[i/3] = 0.f;
        Buffer_2[i/3] = 0.f;
        Buffer_3[i/3] = 0.f;
        Buffer_4[i/3] = 0.f;
        Buffer_5[i/3] = 0.f;
        Buffer_6[i/3] = 0.f;
        Buffer_B[i/3] = 0.f;
    }
}                
"""
