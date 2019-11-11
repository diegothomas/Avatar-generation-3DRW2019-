#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:54:13 2017

@author: diegothomas
"""

Kernel_Matrix_B = """
int NB_BS = 28;
if (i%3 == 0) {
    float bump = BumpImage[i];
    float mask = BumpImage[i+1];
    //float label = BumpImage[i+2];
    int label_mask = LabelsMask[i/3]; // = is front Pixel?
    float w = 1.0f;
    float4 nBump = (float4) {NMapBump[i],NMapBump[i+1],NMapBump[i+2],0.0f};

    if (mask > 0.0f && label_mask != 0 && (nBump.x != 0.0f || nBump.y != 0.0f || nBump.z != 0.0f)) {
        float nmleBump[3];
        nmleBump[0] = nBump.x * Pose[0] + nBump.y * Pose[1] + nBump.z * Pose[2];
        nmleBump[1] = nBump.x * Pose[4] + nBump.y * Pose[5] + nBump.z * Pose[6];
        nmleBump[2] = nBump.x * Pose[8] + nBump.y * Pose[9] + nBump.z * Pose[10]; 
        
        
        if (nmleBump[2] < 0.0f) {
            float nmle[3] = {0.0f,0.0f,0.0f};
            float pt[3] = {0.0f,0.0f,0.0f};
            float pt_ref[3] = {0.0f,0.0f,0.0f};
            float pt_TB[27][3];
            float nmleTmp[3] = {0.0f,0.0f,0.0f};
            float ptTmp[3] = {0.0f,0.0f,0.0f};
            float d = bump / 1000.0f;
            
            pt[0] = VerticesBS[2*i];
            pt[1] = VerticesBS[2*i + 1];
            pt[2] = VerticesBS[2*i + 2];
            nmleTmp[0] = VerticesBS[2*i + 3];
            nmleTmp[1] = VerticesBS[2*i + 4];
            nmleTmp[2] = VerticesBS[2*i + 5];
            
            ptTmp[0] = pt[0] + d*nmleTmp[0];
            ptTmp[1] = pt[1] + d*nmleTmp[1];
            ptTmp[2] = pt[2] + d*nmleTmp[2];
    		
            nmle[0] = nmleTmp[0] * Pose[0] + nmleTmp[1] * Pose[1] + nmleTmp[2] * Pose[2];
            nmle[1] = nmleTmp[0] * Pose[4] + nmleTmp[1] * Pose[5] + nmleTmp[2] * Pose[6];
            nmle[2] = nmleTmp[0] * Pose[8] + nmleTmp[1] * Pose[9] + nmleTmp[2] * Pose[10];
            
            pt[0] = ptTmp[0] * Pose[0] + ptTmp[1] * Pose[1] + ptTmp[2] * Pose[2];
            pt[1] = ptTmp[0] * Pose[4] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[6];
            pt[2] = ptTmp[0] * Pose[8] + ptTmp[1] * Pose[9] + ptTmp[2] * Pose[10];
            pt_ref[0] = pt[0] + Pose[3];
            pt_ref[1] = pt[1] + Pose[7];
            pt_ref[2] = pt[2] + Pose[11];
            
            #pragma unroll
            for (int k = 1; k < NB_BS; k++) {
                // This blended normal is not really a normal since it may be not normalized
                nmleTmp[0] = VerticesBS[k*6*n_bump*m_bump + 2*i + 3];
                nmleTmp[1] = VerticesBS[k*6*n_bump*m_bump + 2*i + 4];
                nmleTmp[2] = VerticesBS[k*6*n_bump*m_bump + 2*i + 5];

                ptTmp[0] = VerticesBS[k*6*n_bump*m_bump + 2*i];
                ptTmp[1] = VerticesBS[k*6*n_bump*m_bump + 2*i + 1];
                ptTmp[2] = VerticesBS[k*6*n_bump*m_bump + 2*i + 2];
			
                ptTmp[0] = ptTmp[0] + d*nmleTmp[0];
                ptTmp[1] = ptTmp[1] + d*nmleTmp[1];
                ptTmp[2] = ptTmp[2] + d*nmleTmp[2];
			
                pt_TB[k-1][0] = ptTmp[0] * Pose[0] + ptTmp[1] * Pose[1] + ptTmp[2] * Pose[2];
                pt_TB[k-1][1] = ptTmp[0] * Pose[4] + ptTmp[1] * Pose[5] + ptTmp[2] * Pose[6];
                pt_TB[k-1][2] = ptTmp[0] * Pose[8] + ptTmp[1] * Pose[9] + ptTmp[2] * Pose[10];
			
                pt[0] = pt[0] + BlendshapeCoeff[k] * pt_TB[k-1][0];
                pt[1] = pt[1] + BlendshapeCoeff[k] * pt_TB[k-1][1];
                pt[2] = pt[2] + BlendshapeCoeff[k] * pt_TB[k-1][2];
            }
            
            pt[0] = pt[0] + Pose[3];
            pt[1] = pt[1] + Pose[7];
            pt[2] = pt[2] + Pose[11];
				
            int p_indx[2];	
            p_indx[0] = min(m_col-1, max(0, convert_int(round((pt[0]/fabs(pt[2]))*calib_depth[0] + calib_depth[2])))); 
            p_indx[1] = min(n_row-1, max(0, convert_int(round((pt[1]/fabs(pt[2]))*calib_depth[1] + calib_depth[3])))); 
               
            int size = 2;
            int li = max(p_indx[1] - size, 0);
            int ui = min(p_indx[1] + size+1, n_row);
            int lj = max(p_indx[0] - size, 0);
            int uj = min(p_indx[0] + size+1, m_col);
            float dist;
            float min_dist = 1000.0f;
            float4 best_n;
            float4 best_v;
            
            float4 nprev;
            float4 vprev;
            for (int ii = li; ii < ui; ii++) {
                for (int j = lj; j < uj; j++) {
			
                    nprev.x = NMap[3*(ii*m_col + j)];
                    nprev.y = NMap[3*(ii*m_col + j)+1];
                    nprev.z = NMap[3*(ii*m_col + j)+2];
				
                    if (nprev.x != 0.0 || nprev.y != 0.0 || nprev.z != 0.0) {
                        vprev.x = VMap[3*(ii*m_col + j)];
                        vprev.y = VMap[3*(ii*m_col + j)+1];
                        vprev.z = VMap[3*(ii*m_col + j)+2];
    
                        dist = sqrt((vprev.x-pt[0])*(vprev.x-pt[0]) + (vprev.y-pt[1])*(vprev.y-pt[1]) + (vprev.z-pt[2])*(vprev.z-pt[2]));
                        float dist_angle = nmleBump[0] * nprev.x + nmleBump[1] * nprev.y + nmleBump[2] * nprev.z;
    
                        if (dist < min_dist && dist_angle > angleThres) {
                            min_dist = dist;
                            best_n.x = nprev.x; best_n.y = nprev.y; best_n.z = nprev.z;
                            best_v.x = vprev.x; best_v.y = vprev.y; best_v.z = vprev.z;
                        }
                    }
                }
            }
                       
            if (min_dist < distThres) {
                for (int k = 0; k < NB_BS-1; k++) {			
                    buf[NB_BS*(i/3) + k] = double(w*(best_n.x*pt_TB[k][0] + best_n.y*pt_TB[k][1] + best_n.z*pt_TB[k][2]));
                }
                buf[NB_BS*(i/3) + NB_BS-1] = -double(w*(best_n.x * (pt[0] - best_v.x) + best_n.y * (pt[1] - best_v.y) + best_n.z * (pt[2] - best_v.z)));                
            }
        } else {
            for (int k = 0; k < NB_BS-1; k++) {			
                buf[NB_BS*(i/3) + k] = 0.0;
            }
            buf[NB_BS*(i/3) + NB_BS-1] = 0.0;                
        }			
    } else {
        for (int k = 0; k < NB_BS-1; k++) {			
            buf[NB_BS*(i/3) + k] = 0.0;
        }
        buf[NB_BS*(i/3) + NB_BS-1] = 0.0;                
    }					
}
"""