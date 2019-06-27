#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:05:11 2018

@author: 
"""
import cv2
import numpy as np
import os
import pickle
directory = os.path.dirname(os.path.abspath(__file__))
sift = cv2.xfeatures2d.SIFT_create()
ratio = 0.8
width = 10
length = 2*width

def calculation2(number,interval):
    img1 = cv2.imread(os.path.join(directory, 'haya0630_images', 'ryugu_image0630_%s.bmp' %str(3*number).zfill(3)),0)
    img2 = cv2.imread(os.path.join(directory, 'haya0630_images', 'ryugu_image0630_%s.bmp' %str(3*(number+interval)).zfill(3)),0)
    #img1 = cv2.imread(os.path.join(directory, 'save_folder', 'others', 'test_image01.bmp'),0)
    #img2 = cv2.imread(os.path.join(directory, 'save_folder', 'others', 'test_image02.bmp'),0)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_paras = dict(algorithm = FLANN_INDEX_KDTREE,trees=5)
    search_paras = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_paras,search_paras)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good=[]
    pts1=[]
    pts2=[]
    des1_list = []
    des2_list = []
    size1_list = []
    angle1_list = []
    response1_list = []
    octave1_list = []
    p_class1_list = []
    for i,(m,n) in enumerate(matches):
        if (m.distance < ratio*n.distance)and(420 < kp1[m.queryIdx].pt[0] < 600)and(360 < kp1[m.queryIdx].pt[1] < 670):
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            des2_list.append(des2[m.trainIdx])
            des1_list.append(des1[m.queryIdx])
            size1_list.append(kp1[m.queryIdx].size)
            angle1_list.append(kp1[m.queryIdx].angle)
            response1_list.append(kp1[m.queryIdx].response)
            octave1_list.append(kp1[m.queryIdx].octave)
            p_class1_list.append(kp1[m.queryIdx].class_id)
            
    pts1 = np.float32(pts1) 
    pts2 = np.float32(pts2)   
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR) 
    img2_sift = img2
    pts1_true = []
    pts2_true = []
    random_list = np.random.randint(0,len(pts1),5)
    for i in range(len(random_list)):
        num = random_list[i]
        img1 = cv2.circle(img1,(int(pts1[num][0]),int(pts1[num][1])),2,(0,255,0),3)
        img2_sift = cv2.circle(img2_sift,(int(pts2[num][0]),int(pts2[num][1])),2,(0,255,0),3)
        pts1_true.append(pts1[num])
        pts2_true.append(pts2[num])
    cv2.imwrite(os.path.join(directory, 'test','exp2','tracking_points1121(0-12).bmp'), img1) #original
    cv2.imwrite(os.path.join(directory, 'test','exp2','SIFT1121(0-12).bmp'), img2_sift) #SIFT画像保存
    print("save_original & SIFT image")
    
    pt_list = []
    for i in range(len(pts1_true)):
        n = 10*i
        print("%s percent"%str(n))
        pt1 = pts1_true[i]
        pt2 = pts2_true[i]
        temp = img1[int(pt1[1]-width):int(pt1[1]+width),int(pt1[0]-width):int(pt1[0]+width)]
        R_list = []
        xy_list = []
        T_sift = sift_matrix(temp,0,0,0)
        for j in range(70,150):
            for k in range(15):
                ptx0 = pt1[0]+j
                pty0 = pt1[1]-5+k
                I_sift = sift_matrix(img2,ptx0,pty0,1)
                R = calc_similarity(I_sift,T_sift)
                R_list.append(R)
                xy_list.append([j,k])
        index = R_list.index(max(R_list))
        x_ = xy_list[index][0]
        y_ = xy_list[index][1]
        ptx = pt1[0] + x_
        pty = pt1[1] + y_-5        
        img2 = cv2.circle(img2,(int(ptx),int(pty)),2,(0,255,0),5) 
        pt = [ptx,pty]
        pt_list.append(pt)
        pts = [pt1, pt2, pt]
        with open('save_folder/pickle/exp2/proposed(0-12)(%s)1121_2.pickle'%str(i), 'wb') as f:
            pickle.dump(pts, f)
    cv2.imwrite(os.path.join(directory, 'test','exp2','proposed(0-12).bmp'),img2)
    with open('save_folder/pickle/exp2/tracking_point(0-12)1121_2.pickle', 'wb') as f:
        pickle.dump(pts1_true, f)
    with open('save_folder/pickle/exp2/SIFT(0-12)1121_2.pickle', 'wb') as f:
        pickle.dump(pts2_true, f)
    #with open('save_folder/pickle/exp2/proposed(0-12)1117.pickle', 'wb') as f:
    #    pickle.dump(pt_list, f)
        
def sift_matrix(img,ptx,pty,num):
    A = np.arange(length*length*128).reshape(length,length,128)
    for i in range(length):
        for j in range(length):
            kp_fv = list()
            if num == 0:
                k=cv2.KeyPoint(x=i,y=j,_size=3, _angle=0) #temp用
            else:
                k=cv2.KeyPoint(x=ptx-width+i,y=pty-width+j,_size=3, _angle=0)  #I用
            kp_fv.append(k)
            kp, des = sift.compute(img,kp_fv)
            A[i][j] = des
    return A

def calc_similarity(I,T):
    R = 0
    for i in range(length):
        for j in range(length):
            T_sca = np.dot(T[i][j],T[i][j])
            I_sca = np.dot(I[i][j],I[i][j])
            R += np.dot(T[i][j],I[i][j])/np.sqrt(T_sca*I_sca)#Iの方注意
    return R
 
if __name__ == '__main__':
    calculation2(0,4)