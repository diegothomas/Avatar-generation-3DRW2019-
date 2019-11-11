#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:28:58 2017

@author: diegothomas
"""

Kernel_DataProc = """
int NB_BS = 28;
// i is the global index in the vertex image (of size(n*m*3))
if (i%3 == 0) {
		
    float3 bumpIn;
    bumpIn.x = Bump[i]; bumpIn.y = Bump[i+1]; bumpIn.z = Bump[i+2];
    float3 weights;
    weights.x = WeightMap[i]; weights.y = WeightMap[i+1]; weights.z = WeightMap[i+2];
	
    if (bumpIn.z == -1.0f) {
        for (int k = 0; k < NB_BS; k++) {
            VerticesBS[k*6*nbLines*nbColumns + 2*i] = 0.0f;
            VerticesBS[k*6*nbLines*nbColumns + 2*i + 1] = 0.0f;
            VerticesBS[k*6*nbLines*nbColumns + 2*i + 2] = 0.0f;
            VerticesBS[k*6*nbLines*nbColumns + 2*i + 3] = 0.0f;
            VerticesBS[k*6*nbLines*nbColumns + 2*i + 4] = 0.0f;
            VerticesBS[k*6*nbLines*nbColumns + 2*i + 5] = 0.0f;
        }
	} else {
        int v1 = Triangles[6*convert_int(bumpIn.z)];
        int v2 = Triangles[6*convert_int(bumpIn.z) + 2];
        int v3 = Triangles[6*convert_int(bumpIn.z) + 4];
        		
        float nmle[3] = {0.0f,0.0f,0.0f};
        float pt[3] = {0.0f,0.0f,0.0f};
        	
        pt[0] = (weights.x * Vertices[3*v1] + weights.y * Vertices[3*v2] + weights.z * Vertices[3*v3]);
        pt[1] = (weights.x * Vertices[3*v1+1] + weights.y * Vertices[3*v2+1] + weights.z * Vertices[3*v3+1]);
        pt[2] = (weights.x * Vertices[3*v1+2] + weights.y * Vertices[3*v2+2] + weights.z * Vertices[3*v3+2]);
        	float ptRef[3] = {pt[0],pt[1],pt[2]};
        
        	nmle[0] = (weights.x * Normales[3*v1] + weights.y * Normales[3*v2] + weights.z * Normales[3*v3]);
        	nmle[1] = (weights.x * Normales[3*v1+1] + weights.y * Normales[3*v2+1] + weights.z * Normales[3*v3+1]);
        	nmle[2] = (weights.x * Normales[3*v1+2] + weights.y * Normales[3*v2+2] + weights.z * Normales[3*v3+2]);
        	float tmp = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
        	nmle[0] = nmle[0]/tmp;
        	nmle[1] = nmle[1]/tmp;
        	nmle[2] = nmle[2]/tmp;
        	
        	float nmleRef[3] = {nmle[0],nmle[1],nmle[2]};
        	
        	VerticesBS[2*i] = ptRef[0];
        	VerticesBS[2*i + 1] = ptRef[1];
        	VerticesBS[2*i + 2] = ptRef[2];
        	VerticesBS[2*i + 3] = nmleRef[0];
        	VerticesBS[2*i + 4] = nmleRef[1];
        	VerticesBS[2*i + 5] = nmleRef[2];
        	
        	float nTmp[3];
        	float pTmp[3];
        	for (int k = 1; k < NB_BS; k++) {
        		nTmp[0] = (weights.x * Normales[3*(k*4325 + v1)] + weights.y * Normales[3*(k*4325 + v2)] + weights.z * Normales[3*(k*4325 + v3)]);
        		nTmp[1] = (weights.x * Normales[3*(k*4325 + v1)+1] + weights.y * Normales[3*(k*4325 + v2)+1] + weights.z * Normales[3*(k*4325 + v3)+1]);
        		nTmp[2] = (weights.x * Normales[3*(k*4325 + v1)+2] + weights.y * Normales[3*(k*4325 + v2)+2] + weights.z * Normales[3*(k*4325 + v3)+2]);
        		float tmp = sqrt(nTmp[0]*nTmp[0] + nTmp[1]*nTmp[1] + nTmp[2]*nTmp[2]);
        		nTmp[0] = nTmp[0]/tmp;
        		nTmp[1] = nTmp[1]/tmp;
        		nTmp[2] = nTmp[2]/tmp;
        
        		pTmp[0] = (weights.x * Vertices[3*(k*4325 + v1)] + weights.y * Vertices[3*(k*4325 + v2)] + weights.z * Vertices[3*(k*4325 + v3)]);
        		pTmp[1] = (weights.x * Vertices[3*(k*4325 + v1)+1] + weights.y * Vertices[3*(k*4325 + v2)+1] + weights.z * Vertices[3*(k*4325 + v3)+1]);
        		pTmp[2] = (weights.x * Vertices[3*(k*4325 + v1)+2] + weights.y * Vertices[3*(k*4325 + v2)+2] + weights.z * Vertices[3*(k*4325 + v3)+2]);
        	
        		// This blended normal is not really a normal since it may be not normalized
        		VerticesBS[k*6*nbLines*nbColumns + 2*i] = (pTmp[0] - ptRef[0]);
        		VerticesBS[k*6*nbLines*nbColumns + 2*i + 1] = (pTmp[1] - ptRef[1]);
        		VerticesBS[k*6*nbLines*nbColumns + 2*i + 2] = (pTmp[2] - ptRef[2]);
        		VerticesBS[k*6*nbLines*nbColumns + 2*i + 3] = (nTmp[0] - nmleRef[0]);
        		VerticesBS[k*6*nbLines*nbColumns + 2*i + 4] = (nTmp[1] - nmleRef[1]);
        		VerticesBS[k*6*nbLines*nbColumns + 2*i + 5] = (nTmp[2] - nmleRef[2]);
        	}		
    }
}
"""




Kernel_Bump = """
int NB_BS = 28;
// i is the global index in the vertex image (of size(n*m*3))
if (i%3 == 0) {
		
    float bum_val = Bump[i];
    float maskIn = Bump[i+1];
    float label = Bump[i+2];
	
    if (label == -1.0f) {
        VMapBump[i] = 0.0f; VMapBump[i+1] = 0.0f; VMapBump[i+2] = 0.0f;
        BumpSwap[i] = 0.0f; BumpSwap[i+1] = 0.0f; BumpSwap[i+2] = 0.0f;
        RGBMapBumpSwap[i] = 0.0f; RGBMapBumpSwap[i+1] = 0.0f; RGBMapBumpSwap[i+2] = 0.0f;
    } else {
        BumpSwap[i] = bum_val; BumpSwap[i+1] = maskIn; BumpSwap[i+2] = label;
        float R = RGBMapBump[i]; float G = RGBMapBump[i+1]; float B = RGBMapBump[i+2];
        RGBMapBumpSwap[i] = R; RGBMapBumpSwap[i+1] = G; RGBMapBumpSwap[i+2] = B;
        
        float nmle[3] = {0.0f,0.0f,0.0f};
        float pt[3] = {0.0f,0.0f,0.0f};
        pt[0] = VerticesBS[2*i];
        pt[1] = VerticesBS[2*i + 1];
        pt[2] = VerticesBS[2*i + 2];
        nmle[0] = VerticesBS[2*i + 3];
        nmle[1] = VerticesBS[2*i + 4];
        nmle[2] = VerticesBS[2*i + 5];
        
        #pragma unroll
        for (int k = 1; k < NB_BS; k++) {
            // This blended normal is not really a normal since it may be not normalized
            nmle[0] = nmle[0] + VerticesBS[k*6*n_bump*m_bump + 2*i + 3] * BlendshapeCoeff[k];
            nmle[1] = nmle[1] + VerticesBS[k*6*n_bump*m_bump + 2*i + 4] * BlendshapeCoeff[k];
            nmle[2] = nmle[2] + VerticesBS[k*6*n_bump*m_bump + 2*i + 5] * BlendshapeCoeff[k];
    
            pt[0] = pt[0] + VerticesBS[k*6*n_bump*m_bump + 2*i] * BlendshapeCoeff[k];
            pt[1] = pt[1] + VerticesBS[k*6*n_bump*m_bump + 2*i + 1] * BlendshapeCoeff[k];
            pt[2] = pt[2] + VerticesBS[k*6*n_bump*m_bump + 2*i + 2] * BlendshapeCoeff[k];
        }
        
        int p_indx[2];
	
        float pt_T[3];
        pt_T[0] = pt[0] * Pose[0] + pt[1] * Pose[1] + pt[2] * Pose[2] + Pose[3];
        pt_T[1] = pt[0] * Pose[4] + pt[1] * Pose[5] + pt[2] * Pose[6] + Pose[7];
        pt_T[2] = pt[0] * Pose[8] + pt[1] * Pose[9] + pt[2] * Pose[10] + Pose[11];
    	
        float nmle_T[3];
        nmle_T[0] = nmle[0] * Pose[0] + nmle[1] * Pose[1] + nmle[2] * Pose[2];
        nmle_T[1] = nmle[0] * Pose[4] + nmle[1] * Pose[5] + nmle[2] * Pose[6];
        nmle_T[2] = nmle[0] * Pose[8] + nmle[1] * Pose[9] + nmle[2] * Pose[10];
    	
        float fact_BP = 1000.0f;
        float d = bum_val/fact_BP;
        
        float4 VMapInn = {pt[0] + d*nmle[0], pt[1] + d*nmle[1], pt[2] + d*nmle[2], 0.0f};
		
        if (maskIn == -1.0f) {
            VMapBump[i] = VMapInn.x; VMapBump[i+1] = VMapInn.y; VMapBump[i+2] = VMapInn.z;
            BumpSwap[i] = bum_val; BumpSwap[i+1] = maskIn; BumpSwap[i+2] = label;
        } else if (nmle_T[2] > 0.0f ) {
            if (maskIn > 0.0f) {
                VMapBump[i] = VMapInn.x; VMapBump[i+1] = VMapInn.y; VMapBump[i+2] = VMapInn.z;
            } else {
                VMapBump[i] = 0.0f; VMapBump[i+1] = 0.0f; VMapBump[i+2] = 0.0f;
            }
        } else {
            float3 NMapIn;
            NMapIn.x = NMapBump[i]; NMapIn.y = NMapBump[i+1]; NMapIn.z = NMapBump[i+2]; 
            float Tnmle[3];
            Tnmle[0] = NMapIn.x * Pose[0] + NMapIn.y * Pose[1] + NMapIn.z * Pose[2];
            Tnmle[1] = NMapIn.x  * Pose[4] + NMapIn.y * Pose[5] + NMapIn.z * Pose[6];
            Tnmle[2] = NMapIn.x  * Pose[8] + NMapIn.y * Pose[9] + NMapIn.z * Pose[10];
            if (Tnmle[0] == 0.0f && Tnmle[1] == 0.0f && Tnmle[2] == 0.0f) {
                Tnmle[0] = nmle_T[0];
                Tnmle[1] = nmle_T[1];
                Tnmle[2] = nmle_T[2];
            }
            
            float min_dist = 1000000000.0f;
            float best_state = -1.0f;
            float fact_curr = round(maskIn) == 0.0f ? 1.0f : min(5.0f, round(maskIn)+1.0f);
            float pos[3];
            
            //summit 1
            d = (bum_val - (50.0f / fact_curr)) / fact_BP;
            pos[0] = pt_T[0] + d*nmle_T[0];
            pos[1] = pt_T[1] + d*nmle_T[1];
            pos[2] = pt_T[2] + d*nmle_T[2];
            
            // Project the point onto the depth image
            float s1[2];
            s1[0] = convert_float(min(m_rgbd - 1, max(0, convert_int(round((pos[0] / fabs(pos[2]))*calib_depth[0] + calib_depth[2])))));
            s1[1] = convert_float(min(n_rgbd - 1, max(0, convert_int(round((pos[1] / fabs(pos[2]))*calib_depth[1] + calib_depth[3])))));
        	
            //summit 2
            d = (bum_val + (50.0f / fact_curr)) / fact_BP;
            pos[0] = pt_T[0] + d*nmle_T[0];
            pos[1] = pt_T[1] + d*nmle_T[1];
            pos[2] = pt_T[2] + d*nmle_T[2];
        	
            // Project the point onto the depth image
            float s2[2];
            s2[0] = convert_float(min(m_rgbd - 1, max(0, convert_int(round((pos[0] / fabs(pos[2]))*calib_depth[0] + calib_depth[2])))));
            s2[1] = convert_float(min(n_rgbd - 1, max(0, convert_int(round((pos[1] / fabs(pos[2]))*calib_depth[1] + calib_depth[3])))));
        	
            float length = sqrt((s1[0]-s2[0])*(s1[0]-s2[0]) + (s1[1]-s2[1])*(s1[1]-s2[1]));
        	
            float dir[2];
            dir[0] = (s2[0]-s1[0])/length;
            dir[1] = (s2[1]-s1[1])/length;
        	
            d = bum_val / fact_BP;
            pos[0] = pt_T[0] + d*nmle_T[0];
            pos[1] = pt_T[1] + d*nmle_T[1];
            pos[2] = pt_T[2] + d*nmle_T[2];
            
            float4 ptIn;
            float4 nmleIn;
        	
            float thresh_dist = round(maskIn) == 0.0f ? 0.06f : 0.01f;
            
            //#pragma unroll
            for (float lambda = 0.0f; lambda <= length; lambda += 1.0f) {
                int k = convert_int(round(s1[1]+ lambda*dir[1]));
                int l = convert_int(round(s1[0] + lambda*dir[0]));
        		
                if (k < 0 || k > n_rgbd - 1 || l < 0 || l > m_rgbd - 1)
                    continue;
                    
                int size = 1;
                int ll = max(0, (int)k-size);
                int ul = min(n_rgbd, (int)k+size+1);
                int lr = max(0, (int)l-size);
                int ur = min(m_rgbd, (int)l+size+1);
                
                for (int kk = ll; kk < ul; kk++) {
                    for (int lk = lr; lk < ur; lk++) {
                        ptIn.x = VMap[3*(kk*m_rgbd + lk)]; ptIn.y = VMap[3*(kk*m_rgbd + lk) + 1]; ptIn.z = VMap[3*(kk*m_rgbd + lk) + 2]; 
                        nmleIn.x = NMap[3*(kk*m_rgbd + lk)]; nmleIn.y = NMap[3*(kk*m_rgbd + lk) + 1]; nmleIn.z = NMap[3*(kk*m_rgbd + lk) + 2]; 
                        
                        if (nmleIn.x == 0.0f && nmleIn.y == 0.0f && nmleIn.z == 0.0f)
                            continue;
                            
                        //compute distance of point to the normal
                        float u_vect[3];
                        u_vect[0] = ptIn.x - pt_T[0];
                        u_vect[1] = ptIn.y - pt_T[1];
                        u_vect[2] = ptIn.z - pt_T[2];
        
                        float proj = u_vect[0] * nmle_T[0] + u_vect[1] * nmle_T[1] + u_vect[2] * nmle_T[2];
                        float v_vect[3];
                        v_vect[0] = u_vect[0] - proj * nmle_T[0];
                        v_vect[1] = u_vect[1] - proj * nmle_T[1];
                        v_vect[2] = u_vect[2] - proj * nmle_T[2];
                        float dist = sqrt((ptIn.x - pos[0]) * (ptIn.x - pos[0]) + (ptIn.y- pos[1]) * (ptIn.y - pos[1]) + (ptIn.z - pos[2]) * (ptIn.z - pos[2]));
                        float dist_to_nmle = sqrt(v_vect[0] * v_vect[0] + v_vect[1] * v_vect[1] + v_vect[2] * v_vect[2]);
                        float dist_angle = Tnmle[0] * nmleIn.x + Tnmle[1] * nmleIn.y + Tnmle[2] * nmleIn.z;
        
                        if (dist_to_nmle < min_dist && dist_angle > 0.7f && dist < thresh_dist) {
                            min_dist = dist_to_nmle;
                            best_state = proj * fact_BP;
                        }
                    }
                }
            }
                        
            VMapInn.x = pt[0] + d*nmle[0];
            VMapInn.y = pt[1] + d*nmle[1];
            VMapInn.z = pt[2] + d*nmle[2];
            if (best_state == -1.0f || min_dist > 0.01f) {     
                float p1[3];   
                p1[0] = pt_T[0] + d*nmle_T[0];
                p1[1] = pt_T[1] + d*nmle_T[1];
                p1[2] = pt_T[2] + d*nmle_T[2]; 
                //Test for visibility violation
                p_indx[0] = min(m_rgbd - 1, max(0, convert_int(round((p1[0] / fabs(p1[2]))*calib_depth[0] + calib_depth[2]))));
                p_indx[1] = min(n_rgbd - 1, max(0, convert_int(round((p1[1] / fabs(p1[2]))*calib_depth[1] + calib_depth[3]))));
                ptIn.x = VMap[3*(p_indx[1]*m_rgbd + p_indx[0])]; ptIn.y = VMap[3*(p_indx[1]*m_rgbd + p_indx[0]) + 1]; ptIn.z = VMap[3*(p_indx[1]*m_rgbd + p_indx[0]) + 2]; 
                
                if (ptIn.z < (p1[2] - 0.05f)) { //visibility violation
                    if (maskIn < 1.0f) {
                       VMapBump[i] = 0.0f; VMapBump[i+1] = 0.0f; VMapBump[i+2] = 0.0f;
                       BumpSwap[i] = 0.0f; BumpSwap[i+1] = 0.0f; BumpSwap[i+2] = label;
                    } else {
                        VMapBump[i] = VMapInn.x; VMapBump[i+1] = VMapInn.y; VMapBump[i+2] = VMapInn.z;
                        BumpSwap[i] = bum_val; BumpSwap[i+1] = maskIn - 1.0f; BumpSwap[i+2] = label;
                    }
                } else {
                    if (maskIn > 0.0f) {
            			VMapBump[i] = VMapInn.x; VMapBump[i+1] = VMapInn.y; VMapBump[i+2] = VMapInn.z;
            		  } else {
            			VMapBump[i] = 0.0f; VMapBump[i+1] = 0.0f; VMapBump[i+2] = 0.0f;
            		  }
                }
            } else {
                float weight = 1.0f;
                
                float new_bump = (weight*best_state + bum_val*maskIn) / (maskIn + weight);
                float4 bumpOut = {new_bump, min(10.0f, maskIn + weight), label, 1.0f};
                if (maskIn < 2000.0f) {
                    BumpSwap[i] = bumpOut.x; BumpSwap[i+1] = bumpOut.y; BumpSwap[i+2] = bumpOut.z;
                } else {
                    new_bump = bum_val;
                }
                
                //Get color
                float p1[3];
                d = new_bump / fact_BP;
                p1[0] = pt_T[0] + d*nmle_T[0];
                p1[1] = pt_T[1] + d*nmle_T[1];
                p1[2] = pt_T[2] + d*nmle_T[2];
                
                float p1_T[3];
                p1_T[0] = p1[0] * Pose_D2RGB[0] + p1[1] * Pose_D2RGB[1] + p1[2] * Pose_D2RGB[2] + Pose_D2RGB[3];
                p1_T[1] = p1[0] * Pose_D2RGB[4] + p1[1] * Pose_D2RGB[5] + p1[2] * Pose_D2RGB[6] + Pose_D2RGB[7];
                p1_T[2] = p1[0] * Pose_D2RGB[8] + p1[1] * Pose_D2RGB[9] + p1[2] * Pose_D2RGB[10] + Pose_D2RGB[11];
                
                p_indx[0] = min(m_rgbd - 1, max(0, convert_int(round((p1_T[0] / fabs(p1_T[2]))*calib_rgb[0] + calib_rgb[2]))));
                p_indx[1] = min(n_rgbd - 1, max(0, convert_int(round((p1_T[1] / fabs(p1_T[2]))*calib_rgb[1] + calib_rgb[3]))));             	
                
                uint3 pixelRGB;
                pixelRGB.x = RGBMap[3*(p_indx[1]*m_rgbd + p_indx[0])]; 
                pixelRGB.y = RGBMap[3*(p_indx[1]*m_rgbd + p_indx[0]) + 1]; 
                pixelRGB.z = RGBMap[3*(p_indx[1]*m_rgbd + p_indx[0]) + 2]; 
                float3 RGBMapIn;
                RGBMapIn. x = RGBMapBump[i]; RGBMapIn.y = RGBMapBump[i+1]; RGBMapIn.z = RGBMapBump[i+2];
                if (maskIn < 2000.0f) {
                    RGBMapBumpSwap[i] = (weight*convert_float(pixelRGB.z) + RGBMapIn.x*maskIn) / (maskIn + weight); 
                    RGBMapBumpSwap[i+1] = (weight*convert_float(pixelRGB.y) + RGBMapIn.y*maskIn) / (maskIn + weight); 
                    RGBMapBumpSwap[i+2] = (weight*convert_float(pixelRGB.x) + RGBMapIn.z*maskIn) / (maskIn + weight); 
                }
                
                VMapInn.x = pt[0] + d*nmle[0];
                VMapInn.y = pt[1] + d*nmle[1];
                VMapInn.z = pt[2] + d*nmle[2];
                if (maskIn + weight > 0.0) {
                    VMapBump[i] = VMapInn.x; VMapBump[i+1] = VMapInn.y; VMapBump[i+2] = VMapInn.z;
                } else {
                    VMapBump[i] = 0.0f; VMapBump[i+1] = 0.0f; VMapBump[i+2] = 0.0f;
                }           
            }        
        }
    }
}
	
"""

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
}
"""

# Input:
#   res = rendered image
#   vmap = input vertex image
#   nmap = input normal image
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
                    atomic_xchg(&res[3*(int)(pix.y + pix.x*nbColumns)], (unsigned int)(color[i])); 
                    atomic_xchg(&res[3*(int)(pix.y + pix.x*nbColumns)+1], (unsigned int)(color[i+1])); 
                    atomic_xchg(&res[3*(int)(pix.y + pix.x*nbColumns)+2], (unsigned int)(color[i+2])); 
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