# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:17:54 2023

@author: hsu12
"""

import cv2
from picamera2 import Picamera2
import numpy as np
import time
import pandas as pd
import os

# Definition 
def length_cal(layer_name):
    length_list = []
    for i in range(len(layer_name)-1):
        length = abs(layer_name.iloc[i]['x'] - layer_name.iloc[i+1]['x'])
        length_list.append(length)
    return np.average(length_list)

def length_cal_list(layer_name):
    length_list = []
    for i in range(len(layer_name)-1):
        length = abs(layer_name.iloc[i]['x'] - layer_name.iloc[i+1]['x'])
        length_list.append(length)            
    return length_list

def Euclidian_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def Euclidian_distance2(a_x, a_y, b_x, b_y):
    return (a_x - b_x) ** 2 + (a_y - b_y) ** 2

def get_index(a, b):
    index_dist = []
    for i in range(len(b)):
        dist = []
        for o in range(len(a)):
            dist.append(Euclidian_distance(a[o], b[i]))                
        index_dist.append(dist.index(min(dist)))
    return index_dist    

# 두 점 사이의 각도를 계산하는 함수
def calculate_angle(x, y):
    return np.degrees(np.arctan2(y, x))

def find_corresponding_points3(srcPoints, dstPoints, threshold=0.1):
    # FLANN matcher 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    srcPoints = srcPoints.astype('float32')
    dstPoints = dstPoints.astype('float32')
    matches = flann.match(srcPoints, dstPoints)
    
    # Among the matched points, the order is determined by excluding those with a distance greater than a certain distance and connecting the corresponding points
    srcPoints_new = np.zeros_like(srcPoints)
    dstPoints_new = np.zeros_like(dstPoints)
    visited = [False] * len(srcPoints)
    for m in matches:
        if m.distance <= threshold:
            src_idx = m.queryIdx
            dst_idx = m.trainIdx
            if not visited[src_idx]:
                srcPoints_new[src_idx] = srcPoints[src_idx]
                dstPoints_new[src_idx] = dstPoints[dst_idx]
                visited[src_idx] = True

    return srcPoints_new[visited], dstPoints_new[visited]

# 데이터 프레임 이름이 중복안되게 저장하는 함수
def save_dataframe_to_csv(df, base_filename, directory='.'):
    """
    DataFrame을 지정된 디렉토리에 저장하고, 중복된 파일 이름이 있으면 번호를 붙입니다.
    
    Parameters:
    - df: 저장할 DataFrame
    - base_filename: 저장할 파일의 기본 이름 (확장자 제외)
    - directory: 저장할 디렉토리 경로 (기본값은 현재 디렉토리)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    base_filename = os.path.splitext(base_filename)[0]  # 확장자 제거
    full_path = os.path.join(directory, base_filename + '.csv')
    
    counter = 1
    while os.path.exists(full_path):
        full_path = os.path.join(directory, f"{base_filename}_{counter}.csv")
        counter += 1
    
    df.to_csv(full_path, index=False)
    print(f"DataFrame saved to {full_path}")
    
# Video size
h = 240
w = 320

# Image mapping parameters
RAN = 12.0
nn = 40
frame_to_frame_threshold = 50

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={'size': (w, h)}, raw={'format': 'SRGGB8', 'size': (1640, 1232), 'crop_limits': (1000, 750, 3280, 2464)}, 
                                              controls={"FrameRate": 150, "ScalerCrop": (1000, 750,  3280, 2464), 'FrameDurationLimits': (3333, 33333)}, buffer_count=10) #fps 42 good
picam2.configure(config)
print('high speed active')
picam2.start()

# Camera calibration setting
mtx = np.array([[3.73639562e+03, 0.00000000e+00, 2.09275876e+02],
       [0.00000000e+00, 3.50019093e+03, 2.75594551e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.31613826e+01,  6.13595466e+02,  4.63391459e-02,
        -6.76091323e-01, -3.17320359e+04]])

newcameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
x_r,y_r,w_r,h_r = roi

# Global variables
sharpening_mask = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]]) #Custom sharpness filter
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8)) #CLAHE
cut = 5 # Boundary cutter  default : 10
median_area = 1500 # Median area 500
# ref_center_point = (140, 120) # Center points
ref_center_point = (160, 120)
initialize_time = 50000 # Initialize start time
area_threshold = w/4

# Virtual coordinate algorithm parameters
threshold_value_x = 20
threshold_value_y = 10
image_limit_x0 = 0+cut+threshold_value_x
image_limit_x1 = w-cut-threshold_value_x
image_limit_y0 = 0+cut+threshold_value_y
image_limit_y1 = h-cut-threshold_value_y
out_index=[]
index_dist = []

# etc
init_cord = []
time_save = []
init_cord_0to5 = []
center_point_contour_df_list = []
MOI_time_list = []
MOI_time_t1 = 0
execution_time  = 0
init_cord_list = []
pixel_dist_X_all = []
pixel_dist_Y_all = []
quadrant_1_x_all = []
quadrant_1_y_all = []
quadrant_2_x_all = []
quadrant_2_y_all = []     
quadrant_3_x_all = []
quadrant_3_y_all = []
quadrant_4_x_all = []
quadrant_4_y_all = []
directions = []
angle_dist_all = []
angle_dist_quadrant_1_all = []
angle_dist_quadrant_2_all = []
angle_dist_quadrant_3_all = []
angle_dist_quadrant_4_all = []

map_0to5 = []
map_0to5_TL = []
map_0to5_TR = []
map_0to5_BL = []
map_0to5_BR = []

frame_count = 0
MOI_time = 0
start_time = time.time()

start_time0 = time.time()
while True:
    frame = picam2.capture_array()
    MOI_start = time.time()
    img = frame
    
    frame_count += 1
    img = cv2.flip(img, 1)
    
    # Image processing process        
    img_clahe = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)
    
    img_shap = cv2.filter2D(img_clahe, -1, sharpening_mask) 
    img_gray = cv2.cvtColor(img_shap, cv2.COLOR_BGR2GRAY)  
    gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB, cv2.CV_32F) 
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    blur[h-cut:h, 0:w] = 255
    blur[0:cut, 0:w] = 255
    blur[0:h, 0:cut] = 255
    blur[0:h, w-cut:w] = 255    
    
    gray_blur  = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, imthres = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imthres = imthres.astype(np.uint8)
    
    # Sampling of contour area according to median area
    contour, hier = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    list_contour = list(contour)
    
    median_area = np.median([cv2.contourArea(list_contour[c]) for c in range(len(list_contour))])
    list_contour2 = []
    for i in range(len(list_contour)):
        if median_area*0.2 < cv2.contourArea(list_contour[i]) < median_area*50: list_contour2.append(list_contour[i]) 
        
    # Visualize contour boxes and calculate center points of contour boxes
    center_point_contour_list = []    
    for i in list_contour2:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        center_TRUE = (round((x1+x2)/2, 5), round((y1+y2)/2, 5))        
        
        cv2.circle(img, (int(center_TRUE[0]), int(center_TRUE[1])), 1, (0,255,0), -1)     
        cv2.drawContours(img, [box], 0, (0,255,0), 2) 
        center_point_contour_list.append(center_TRUE)
        
    center_point_contour_df = np.array(center_point_contour_list)
    
    # Initialize process (specific fps or keyboard 'w')     
    if len(time_save) == initialize_time or cv2.waitKey(1) & 0xFF == ord('w') :           
        cp_dist_list = []
        index_list = []
        init_cord = [] 
        
    # Reconsider of center point 
        cp_dist_list0 = []
        index_list0 = []
        for i in range(len(center_point_contour_df)):
            cp_dist = Euclidian_distance(center_point_contour_df[i], ref_center_point)  
            cp_dist_list0.append(cp_dist)
            index_list0.append(i)                
        cp_dist_list0 = np.stack([np.array(cp_dist_list0), np.array(index_list0)], -1)     
        cp_dist_list0 = cp_dist_list0[cp_dist_list0[:,0].argsort()]
        
        center_point = center_point_contour_df[int(cp_dist_list0[0][1])]
        
    # Basic center points
        center_point_contour_df_list = center_point_contour_df
        
        adjacent_distances = []
        for i in range(len(center_point_contour_df_list) - 1):
            dist = np.linalg.norm(center_point_contour_df_list[i] - center_point_contour_df_list[i + 1])
            adjacent_distances.append(dist)
        
        Q1 = np.percentile(adjacent_distances, 20)
        Q3 = np.percentile(adjacent_distances, 50)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR
        upper_bound = Q3 + IQR
        
        adjacent_distances = np.array(adjacent_distances)
        filtered_distances = adjacent_distances[(adjacent_distances >= lower_bound) & (adjacent_distances <= upper_bound)]        
        adjacent_length = np.mean(filtered_distances)
        
        x_min = (center_point[0] + adjacent_length/2 - area_threshold)*1
        x_max = (center_point[0] + adjacent_length/2 + area_threshold)*1
        y_min = (center_point[1] + adjacent_length/2 - area_threshold)*1
        y_max = (center_point[1] + adjacent_length/2 + area_threshold)*1
        
    # 지정된 범위 내에 있는 좌표들을 추출
        selected_coordinates = center_point_contour_df_list[
            (center_point_contour_df_list[:, 0] >= x_min) & (center_point_contour_df_list[:, 0] <= x_max) &
            (center_point_contour_df_list[:, 1] >= y_min) & (center_point_contour_df_list[:, 1] <= y_max)]
        changed_index = selected_coordinates
        
        refine_index_list = []
        for j in range(0,4):
            remaining_points_list = []
            remaining_points_list2 = []
            remaining_points_list3 = []
            remaining_points_list4 = []
                 
            p_first = np.array(sorted(changed_index, key=lambda p: (p[0]) + (p[1])))[0]
            p_fourth = np.array(sorted(changed_index, key=lambda p: (p[0]) - (p[1])))[-1]
            p_second_ref = np.array([(- p_first[0] + p_fourth[0])*1/3 + p_first[0], (p_first[1] + p_fourth[1])/2])
            p_third_ref = np.array([(- p_first[0] + p_fourth[0])*2/3 + p_first[0], (p_first[1] + p_fourth[1])/2])          
            
            changed_index_dist_list = []
            for i in changed_index:
                changed_index_dist = Euclidian_distance(i, p_second_ref)     
                changed_index_dist_list.append(changed_index_dist)
            p_second = changed_index[changed_index_dist_list.index(min(changed_index_dist_list))]
            
            changed_index_dist_list = []
            for i in changed_index:
                changed_index_dist = Euclidian_distance(i, p_third_ref)     
                changed_index_dist_list.append(changed_index_dist)
            p_third = changed_index[changed_index_dist_list.index(min(changed_index_dist_list))]
            
            for ii in range(len(changed_index)) :        
                if changed_index[ii][0] == p_first[0] and changed_index[ii][1] == p_first[1] :
                    refine_index_list.append(changed_index[ii])
                else :
                    remaining_points_list.append(changed_index[ii])
            
            for ii in range(len(remaining_points_list)) :               
                if remaining_points_list[ii][0] == p_second[0] and remaining_points_list[ii][1] == p_second[1] :
                    refine_index_list.append(remaining_points_list[ii])
                else :
                    remaining_points_list2.append(remaining_points_list[ii])    
                
            for ii in range(len(remaining_points_list2)) :          
                if remaining_points_list2[ii][0] == p_third[0] and remaining_points_list2[ii][1] == p_third[1] :
                    refine_index_list.append(remaining_points_list2[ii])    
                else :
                    remaining_points_list3.append(remaining_points_list2[ii])
                    
            for ii in range(len(remaining_points_list3)) :          
                if remaining_points_list3[ii][0] == p_fourth[0] and remaining_points_list3[ii][1] == p_fourth[1] :
                    refine_index_list.append(remaining_points_list3[ii])    
                else :
                    remaining_points_list4.append(remaining_points_list3[ii])        
                    
            changed_index = remaining_points_list4
        
        init_cord = np.array(refine_index_list)
        init_cord0 = init_cord.copy()
        print("Initialize finish")    

    # Caculate change of pixel in 16 contours [나중에 x y축 분리한거 없애기]
    if len(init_cord) != 0 :
        init_cord_t = init_cord.copy()
        init_cord_t0_flat = init_cord.flatten('F')
        init_cord_t_flat = init_cord_t.flatten('F')
        
        if len(out_index) > 0 :
            for i in range(len(out_index)): 
                if out_index[i] < 16:
                    init_cord_t_flat[out_index[i]] = init_cord_t0_flat[out_index[i]]
                    init_cord_t_flat[out_index[i]+16] = init_cord_t0_flat[out_index[i]+16]
                elif out_index[i] >= 16:
                    init_cord_t_flat[out_index[i]] = init_cord_t0_flat[out_index[i]]
                    init_cord_t_flat[out_index[i]-16] = init_cord_t0_flat[out_index[i]-16]
                    
            init_cord_t_flat.resize(2,16)
            init_cord_t = init_cord_t_flat.T            
            index_dist = get_index(center_point_contour_df, init_cord_t)         
        else :
            index_dist = get_index(center_point_contour_df, init_cord)         
        
        init_cord_list.append(init_cord)
        # t-1 out MOI checker
        out_index_x = [i for i, value in enumerate(init_cord[:,0]) if value > image_limit_x1 or value < image_limit_x0]
        out_index_y = [i for i, value in enumerate(init_cord[:,1]) if value > image_limit_y1 or value < image_limit_y0]
        out_index_y = list(np.array(out_index_y) + 16)
        out_index = list(set(out_index_x).difference(set(out_index_y))) + out_index_y
        out_index.sort()   
        
        for i in range(len(index_dist)):
            init_cord_t[i] = center_point_contour_df[index_dist[i]]               
            
        # Virtual coordinate algorithm
        if len(out_index) > 0 :           
            init_cord_t_flat = init_cord_t.flatten('F')
            
            for i in range(len(out_index)):
                init_cord_t_flat[out_index[i]] = init_cord_t0_flat[out_index[i]]    
                
                if out_index[i] < 16 : # x coordinate
                    if init_cord_t_flat[out_index[i]] > image_limit_x1 :
                        for i in range(len(out_index)):
                            if out_index[i] in [3, 7, 11, 15]:
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]-2] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]-2]
                                init_cord_t_flat[out_index[i]+9] = (init_cord_t_flat[out_index[i]-2+9] * init_cord_t0_flat[out_index[i]+9])/init_cord_t0_flat[out_index[i]-2+9]
                        
                            elif out_index[i] in [2, 6, 10, 14]:        
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]-1] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]-1]
                                init_cord_t_flat[out_index[i]+9] = (init_cord_t_flat[out_index[i]-1+9] * init_cord_t0_flat[out_index[i]+9])/init_cord_t0_flat[out_index[i]-1+9]
                            
                    elif init_cord_t_flat[out_index[i]] < image_limit_x0 :
                        for i in range(len(out_index)):
                            if out_index[i] in [0, 4, 8, 12]:
                                print(out_index[i] )
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]+2] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]+2]
                                init_cord_t_flat[out_index[i]+9] = (init_cord_t_flat[out_index[i]+2+9] * init_cord_t0_flat[out_index[i]+9])/init_cord_t0_flat[out_index[i]+2+9]
                            
                            elif out_index[i] in [1, 5, 9, 13]:    
                                print(out_index[i] )
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]+1] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]+1]
                                init_cord_t_flat[out_index[i]+9] = (init_cord_t_flat[out_index[i]+1+9] * init_cord_t0_flat[out_index[i]+9])/init_cord_t0_flat[out_index[i]+1+9]
                            
                if out_index[i] >= 16 : # y coordinate
                    if init_cord_t_flat[out_index[i]] > image_limit_y1 :
                        for i in range(len(out_index)):
                            if out_index[i] in [0, 1, 2, 3] :
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]+6] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]+6]
                                init_cord_t_flat[out_index[i]-9] = (init_cord_t_flat[out_index[i]+6-9] * init_cord_t0_flat[out_index[i]-9])/init_cord_t0_flat[out_index[i]+6-9]
                                
                            elif out_index[i] in [4, 5, 6, 7]:
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]+3] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]+3]
                                init_cord_t_flat[out_index[i]-9] = (init_cord_t_flat[out_index[i]+3-9] * init_cord_t0_flat[out_index[i]-9])/init_cord_t0_flat[out_index[i]+3-9]
                                
                    elif init_cord_t_flat[out_index[i]] < image_limit_y0 :
                        for i in range(len(out_index)):
                            if out_index[i] in [12, 13, 14, 15]:
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]-6] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]-6]
                                init_cord_t_flat[out_index[i]-9] = (init_cord_t_flat[out_index[i]-6-9] * init_cord_t0_flat[out_index[i]-9])/init_cord_t0_flat[out_index[i]-6-9]
                                
                            elif out_index[i] in [8, 9, 10, 11]:     
                                init_cord_t_flat[out_index[i]] = (init_cord_t_flat[out_index[i]-3] * init_cord_t0_flat[out_index[i]])/init_cord_t0_flat[out_index[i]-3]
                                init_cord_t_flat[out_index[i]-9] = (init_cord_t_flat[out_index[i]-3-9] * init_cord_t0_flat[out_index[i]-9])/init_cord_t0_flat[out_index[i]-3-9]

            init_cord_t_flat.resize(2,16)
            init_cord_t = init_cord_t_flat.T         
            index_dist = get_index(center_point_contour_df, init_cord_t) 
     
    # MOI visualization
        box_area = []
        for i in range(len(index_dist)):
            rect = cv2.minAreaRect(list_contour2[index_dist[i]])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0,0,255), 2)
            cv2.putText(img, '%.d'%int(i), tuple(np.int0(center_point_contour_df[index_dist[i]])), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),2)
            box_area.append(cv2.contourArea(list_contour2[index_dist[i]]))
        
        # Global x, y change
        pixel_dist_X = [init_cord_t[i][0] - init_cord0[i][0] for i in range(len(init_cord_t))]
        pixel_dist_Y = [init_cord_t[i][1] - init_cord0[i][1] for i in range(len(init_cord_t))]
        
        # 각도 변화량을 저장할 리스트 초기화
        angle_dist = []
        for i in range(len(init_cord_t)):
            angle_t = calculate_angle(init_cord_t[i][0], init_cord_t[i][1])
            angle_0 = calculate_angle(init_cord0[i][0], init_cord0[i][1])
            angle_diff = angle_t - angle_0
            angle_dist.append(angle_diff)
        
        # Local x, y change
        quadrant_1_indices = [2, 3, 6, 7]
        quadrant_2_indices = [0, 1, 4, 5]
        quadrant_3_indices = [8, 9, 12, 13]
        quadrant_4_indices = [10, 11, 14, 15]
        
        quadrant_1_x = [init_cord_t[i][0] - init_cord0[i][0] for i in quadrant_1_indices]
        quadrant_1_y = [init_cord_t[i][1] - init_cord0[i][1] for i in quadrant_1_indices]
        
        quadrant_2_x = [init_cord_t[i][0] - init_cord0[i][0] for i in quadrant_2_indices]
        quadrant_2_y = [init_cord_t[i][1] - init_cord0[i][1] for i in quadrant_2_indices]
                
        quadrant_3_x = [init_cord_t[i][0] - init_cord0[i][0] for i in quadrant_3_indices]
        quadrant_3_y = [init_cord_t[i][1] - init_cord0[i][1] for i in quadrant_3_indices]
        
        quadrant_4_x = [init_cord_t[i][0] - init_cord0[i][0] for i in quadrant_4_indices]
        quadrant_4_y = [init_cord_t[i][1] - init_cord0[i][1] for i in quadrant_4_indices]
        
        # 각도 변화량을 저장할 리스트 초기화
        angle_dist_quadrant_1 = [calculate_angle(init_cord_t[i][0], init_cord_t[i][1]) - calculate_angle(init_cord0[i][0], init_cord0[i][1]) for i in quadrant_1_indices]
        angle_dist_quadrant_2 = [calculate_angle(init_cord_t[i][0], init_cord_t[i][1]) - calculate_angle(init_cord0[i][0], init_cord0[i][1]) for i in quadrant_2_indices]
        angle_dist_quadrant_3 = [calculate_angle(init_cord_t[i][0], init_cord_t[i][1]) - calculate_angle(init_cord0[i][0], init_cord0[i][1]) for i in quadrant_3_indices]
        angle_dist_quadrant_4 = [calculate_angle(init_cord_t[i][0], init_cord_t[i][1]) - calculate_angle(init_cord0[i][0], init_cord0[i][1]) for i in quadrant_4_indices]

        init_cord = init_cord_t     
        init_cord_0to5.append(init_cord_t.copy())
        
        if len(init_cord_0to5) > 5 :
            del init_cord_0to5[0]
        
        # Weight for MOI         
        if len(init_cord_0to5) == 6 :
           init_cord = 0.075829384*init_cord_0to5[0] + 0.113744076*init_cord_0to5[1] + 0.170616114*init_cord_0to5[2] + 0.255924171*init_cord_0to5[3] + 0.383886256*init_cord_0to5[4]
       
        pixel_dist_X_all.append(np.average(pixel_dist_X))
        pixel_dist_Y_all.append(np.average(pixel_dist_Y))      
        quadrant_1_x_all.append(np.average(quadrant_1_x))
        quadrant_1_y_all.append(np.average(quadrant_1_y))
        quadrant_2_x_all.append(np.average(quadrant_2_x))
        quadrant_2_y_all.append(np.average(quadrant_2_y))
        quadrant_3_x_all.append(np.average(quadrant_3_x))
        quadrant_3_y_all.append(np.average(quadrant_3_y))
        quadrant_4_x_all.append(np.average(quadrant_4_x))
        quadrant_4_y_all.append(np.average(quadrant_4_y))          
        angle_dist_all.append(np.average(angle_dist))
        angle_dist_quadrant_1_all.append(np.average(angle_dist_quadrant_1))
        angle_dist_quadrant_2_all.append(np.average(angle_dist_quadrant_2))
        angle_dist_quadrant_3_all.append(np.average(angle_dist_quadrant_3))
        angle_dist_quadrant_4_all.append(np.average(angle_dist_quadrant_4))     
        MOI_time_list.append(time.time())

    if time.time() - start_time >= 1.0:
        fps = frame_count / (time.time() - start_time)
        print(f"FPS:{fps:.2f}")
        
        # soft reset
        frame_count = 0
        start_time = time.time()

    # Display the image
    cv2.imshow('strain_map', cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA))
   
    # Break rules
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('o'):
        pixel_dist_X_all.append(np.array(100))
        pixel_dist_Y_all.append(np.array(100))      
        quadrant_1_x_all.append(np.array(100))
        quadrant_1_y_all.append(np.array(100))
        quadrant_2_x_all.append(np.array(100))
        quadrant_2_y_all.append(np.array(100))
        quadrant_3_x_all.append(np.array(100))
        quadrant_3_y_all.append(np.array(100))
        quadrant_4_x_all.append(np.array(100))
        quadrant_4_y_all.append(np.array(100))
        angle_dist_all.append(np.array(100))
        angle_dist_quadrant_1_all.append(np.array(100))
        angle_dist_quadrant_2_all.append(np.array(100))
        angle_dist_quadrant_3_all.append(np.array(100))
        angle_dist_quadrant_4_all.append(np.array(100))
        MOI_time_list.append(time.time())
        init_cord_list.append(np.full((16, 2), 100))
        print('recording signal')

cv2.destroyAllWindows()

all_time = start_time - start_time0
print(f"full_time:{all_time:.2f}")
MOI_time_list = np.array(MOI_time_list)-MOI_time_list[0]

all_data = np.array([pixel_dist_X_all, pixel_dist_Y_all, quadrant_1_x_all, 
quadrant_1_y_all, quadrant_2_x_all, quadrant_2_y_all,quadrant_3_x_all, quadrant_3_y_all,
quadrant_4_x_all, quadrant_4_y_all, angle_dist_all, angle_dist_quadrant_1_all, angle_dist_quadrant_2_all,
angle_dist_quadrant_3_all, angle_dist_quadrant_4_all, MOI_time_list]).T

all_data_df = pd.DataFrame(all_data, columns=['X', 'Y', 'q1_x', 'q1_y', 'q2_x', 'q2_y', 'q3_x', 'q3_y', 'q4_x', 'q4_y', 
                                              'total_angle', 'q1_angle', 'q2_angle', 'q3_angle', 'q4_angle','Time'])

flatten_init_cord_list = [np.array(sublist).flatten() for sublist in init_cord_list]
flatten_init_cord_list = np.vstack(flatten_init_cord_list)
flatten_init_cord_list = pd.DataFrame(flatten_init_cord_list, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y', '6x', '6y', '7x', '7y', 
                                                          '8x', '8y', '9x', '9y', '10x', '10y', '11x', '11y', '12x', '12y', '13x', '13y', '14x', '14y', 
                                                          '15x', '15y'])
                                      
save_dataframe_to_csv(flatten_init_cord_list, '250120_performance_test_raw_data.csv')
save_dataframe_to_csv(all_data_df, '250120_performance_test.csv')
