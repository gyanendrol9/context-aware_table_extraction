import cv2
import cv2 as cv
import imutils
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

# Add code to sys.path
import matplotlib.pyplot as plt 

import sys
sys.path.append('../src')

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2 as cv
import collections

import dla.src.table_structure_analysis as tsa
import dla.src.xml_utils as xml_utils
from dla.src.image_utils import put_box, put_line
import pytesseract
import numpy as np

from statistics import mean

import importlib
import glosat_utils
importlib.reload(glosat_utils)

from glosat_utils import *

config_file = '/data/glosat/glosat_table_dataset/dla/config/cascadeRCNN.py'
table_checkpoint_file = '/data/glosat/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth'
cell_checkpoint_file = '/data/glosat/glosat_table_dataset/models/model_coarsecell_GloSAT.pth'

model = init_detector(config_file, table_checkpoint_file, device='cuda:0')
cell_model = init_detector(config_file, cell_checkpoint_file, device='cuda:0')

THRESHOLD = 0.5
CLASSES = ("table_body","cell","full_table","header","heading")

restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in restricted_labels]
label_idx_color_mapping = {restricted_labels.index(l) + 1: c for l, c in zip(restricted_labels, restricted_colors)}

directory = '/data/glosat/Code-Git/docformer/dataset/Finetuning/test/images'
outdirectory = '/data/glosat/Code-Git/docExtractor-master/demo/output_files'

files = os.listdir(directory)
cell_cordinates_info = {}
for filename in files:
	if '.jpg' in filename:
		img_path = f'{directory}/{filename}'
		img = cv.imread(img_path)
		img, h, w, c = image_preprocessing(img)
		masks, blend_img = find_text_region(img, label_idx_color_mapping, normalize)

		blend_img.save(f'{outdirectory}/{filename}_text_region.jpg')

		cells_by_image = collections.defaultdict(list)
		cells_by_image[filename] = detect_cell_pretrained(cell_model,img)

		full_tables, tables = detect_table_region(model,img)

		ori_correct_cells, exclude_cells = classify_single_cell(cells_by_image[filename], masks[1])

		img = cv.imread(img_path)
		image, height, width, _ = image_preprocessing(img)

		table_cells = get_x_y_initial(tables, cells_by_image[filename], THRESHOLD)
		cell_cordinates, average_cordinates = get_coordinates(table_cells,0.7)

		possible_cells, exclude_cells_pred = generate_cells(masks[1], tables, cell_cordinates, average_cordinates)
		correct_cells, blank_cells, exclude_cells = classify_cells(possible_cells, masks[1], img, average_cordinates)    

		not_ovellapped_blank_cells_pred =  find_overlapped_cell(correct_cells, blank_cells)
		considered_cells = correct_cells+not_ovellapped_blank_cells_pred

		not_ovellapped_exclude_cells_pred =  find_overlapped_cell(considered_cells, exclude_cells+exclude_cells_pred)

		new_corrected_excluded_cells, not_corrected_excluded_cells = classify_error_cells(not_ovellapped_exclude_cells_pred, tables, masks[1], average_cordinates,0.7)

		all_valid_cells = ori_correct_cells+correct_cells+not_ovellapped_blank_cells_pred+new_corrected_excluded_cells

		table_cells = get_x_y(tables, all_valid_cells)
		cell_cordinates, average_cordinates = get_coordinates(table_cells,0.7)

		possible_cells, exclude_cells_pred = generate_cells(masks[1], tables, cell_cordinates, average_cordinates)
		correct_cells, blank_cells, exclude_cells = classify_cells(possible_cells, masks[1], img, average_cordinates)  

		new_corrected_excluded_cells, not_corrected_excluded_cells = classify_error_cells(exclude_cells, tables, masks[1], average_cordinates,0.7)

		if len(exclude_cells)>1:
			neighbouring_cells, y_cords = get_x_y_neighbour_cells(exclude_cells, tables, average_cordinates)

			x_y_neighbouring_cells_dict = {}
			for yid in neighbouring_cells:
			    cells = neighbouring_cells[yid]
			    
			    y_neighbours = cells[0]
			    x_neighbours = cells[1]
			    
			    tid =x_neighbours[0][4]
			    
			    avg_x, avg_y = average_cordinates[tid]
			       
			    # find x_neighbours
			    ncells = []
			    imagedraw = []
			    next_cell_begin = True
			    starting_cell = []
			    
			    for xid, xcell in enumerate(x_neighbours):    

			        # Either split the cell or merge to begining cell
			        if next_cell_begin:
			            starting_cell = list(xcell)
			        else:
			            starting_cell[2] = xcell[2]
			        
			        # check next cell distance
			        if (xid+1)<len(x_neighbours):
			            next_cell = x_neighbours[xid+1]
			            if abs(next_cell[0]-xcell[2])<(avg_x/3):
			                next_cell_begin = False
			            else:
			                ncells.append(starting_cell)
			                starting_cell = []
			                next_cell_begin = True
			        elif len(x_neighbours)==1:
			            ncells.append(starting_cell)
			            starting_cell = []
			    
			    if len(starting_cell)>0:
			        ncells.append(starting_cell)
			                
			    x_y_neighbouring_cells_dict[yid] = (y_neighbours, ncells)
			    

			yids = list(x_y_neighbouring_cells_dict.keys())
			count = 0
			y_neighbours = {}
			cur_list = []

			for i, j in zip(yids[:-1],yids[1:]):
			    if count == 0:  # First index
			        cur_list.append(i)        
			    if (j-i) < avg_y:
			        cur_list.append(j)
			    else:
			        y_neighbours[cur_list[0]] = cur_list
			        cur_list = [j]
			    count+=1
			    
			if cur_list[0] not in y_neighbours:
			    y_neighbours[cur_list[0]] = cur_list

			y_neighbours_new = {}
			for yid in y_neighbours:
			    for yidn in y_neighbours[yid]:
			        y_neighbours_new[yidn] = y_neighbours[yid]

			error_corrected_cells = []
			not_corrected_cells = []
			for yid in x_y_neighbouring_cells_dict:
			    ynid, xnid = x_y_neighbouring_cells_dict[yid]
			    
			    if len(ynid) == 0: 
			        new_corrected, not_corrected = classify_error_cells(xnid, tables, masks[1], average_cordinates,0.7) # correct single error cell or declare error
			        
			        if len(not_corrected)==0:
			            error_corrected_cells+=new_corrected
			        else:
			            for cell in not_corrected:                
			                tid = cell[4]
			                avg_x, avg_y = average_cordinates[tid]
			                tab_boundary = list(map(int, tables[tid]))
			                
			                move = cell[6]
			                
			                for idx,i in enumerate(move):
			                    if i == 0:
			                        if abs(cell[idx]-tab_boundary[idx])<avg_y/3:
			                            cell[idx] = tab_boundary[idx]
			                            move[idx] = 5
			                            
			                if 0 in move:
			                    not_corrected_cells+=not_corrected
			                    
			                else:
			                    cell[6] =  move
			                    error_corrected_cells+=[cell]
			                    
			    elif len(xnid)>0: #multiple neighbours
			#         print(ynid, yid)
			        above_cells = []
			        cur_cells = []
			        below_cells = []
			        for y_n in ynid:
			            if y_n < yid:
			#                 print(y_n, yid)
			                above_cells = x_y_neighbouring_cells_dict[y_n][1]
			            else:
			                cur_cells = xnid
			            
			                below_cells = x_y_neighbouring_cells_dict[y_n][1]
			        print(f'============{yid}================== \nAbove cells:{above_cells} \nCurrent cells: {xnid} \nBelow cells: {below_cells}')
			        
			        for cell in xnid:
			            anchor = list(cell)      
			            
			            tid = cell[4]
			            avg_x, avg_y = average_cordinates[tid]
			            tab_boundary = list(map(int, tables[tid]))

			            #check above first:
			            for a_cell in above_cells:
			                anchor_a = list(a_cell)
			                anchor_a[1] = anchor[1] # change the y-coordinates to see overlapping or not and if overlapped the cell is immediate neighbour
			                anchor_a[3] = anchor[3]
			                score1 = tsa.how_much_contained(anchor_a, cell)
			                score2 = tsa.how_much_contained(cell,anchor_a)
			                
			                if score1>0.5:
			                    
			                    
			                    if a_cell[0] < cell[0]:  ## check if a_cell[0] >= cell[0] and change a_cell[0] to cell[0] and a_cell[2] to cell[2] for correction
			                        
			                        print(f'1--{yid} just above cell: {a_cell}, Current cell: {cell}, {anchor_a}, {score1}')
			                        above = True
			                        cur_corrected, n_cur_corrected = correct_boundary_multiple_cell(image, cell, a_cell, above, average_cordinates, tables, masks[1])
			                        error_corrected_cells.append(cur_corrected)
			                        
			                        merged_cell = list(cell)
			                        merged_cell[1] = a_cell[1]
			                        
			                        # croppedimage_full=image[int(merged_cell[1]):int(merged_cell[3]),int(merged_cell[0]):int(merged_cell[2])] 
			                        # plt.rcParams["figure.figsize"] = (10,5)
			                        # plt.imshow(croppedimage_full)
			                        # plt.title(f'{yid} merged cell of {len(y_cords[yid])} #images')
			                        # plt.show()
			                        
			                        # croppedimage_full=image[int(cur_corrected[1]):int(cur_corrected[3]),int(cur_corrected[0]):int(cur_corrected[2])]  
			                        # plt.rcParams["figure.figsize"] = (10,5)
			                        # plt.imshow(croppedimage_full)
			                        # plt.title(f'current cell images')
			                        # plt.show()


			                        # croppedimage_full=image[int(n_cur_corrected[1]):int(n_cur_corrected[3]),int(n_cur_corrected[0]):int(n_cur_corrected[2])]  
			                        # plt.rcParams["figure.figsize"] = (10,5)
			                        # plt.imshow(croppedimage_full)
			                        # plt.title(f'neighbour cell images')
			                        # plt.show()
			                        
			                        
			                        print('-----Above-----\n')
			                    
			                    else:                            
			                        pass ##wrong issue
			            
			            
			            #check below cell:
			            for b_cell in below_cells:
			                anchor_b = list(b_cell)
			                anchor_b[1] = anchor[1] # change the y-coordinates to see overlapping or not and if overlapped the cell is immediate neighbour
			                anchor_b[3] = anchor[3]
			                score1 = tsa.how_much_contained(cell,anchor_b)
			                score2 = tsa.how_much_contained(anchor_b, cell)
			                
			                if score1>0.5:
			                    
			                    if b_cell[0] < cell[0]: ##check if b_cell[0] >= cell[0] and change b_cell[0] to cell[0] and b_cell[2] to cell[2] for correction
			                        
			                        above = False
			                        cur_corrected, n_cur_corrected = correct_boundary_multiple_cell(image, cell, b_cell, above, average_cordinates, tables, masks[1])
			                        error_corrected_cells.append(cur_corrected)
			                        print(f'2--{yid} just below cell: {b_cell}, Current cell: {cell}, {anchor_b} {score1}')
			                        
			                        merged_cell = list(cell)
			                        merged_cell[3] = b_cell[3]
			                        
			                        # croppedimage_full=image[int(merged_cell[1]):int(merged_cell[3]),int(merged_cell[0]):int(merged_cell[2])]  
			                        # plt.rcParams["figure.figsize"] = (10,5)
			                        # plt.imshow(croppedimage_full)
			                        # plt.title(f'{yid} merged cell of {len(y_cords[yid])} #images')
			                        # plt.show()
			                        
			                        # croppedimage_full=image[int(cur_corrected[1]):int(cur_corrected[3]),int(cur_corrected[0]):int(cur_corrected[2])]  
			                        # plt.rcParams["figure.figsize"] = (10,5)
			                        # plt.imshow(croppedimage_full)
			                        # plt.title(f'current cell images')
			                        # plt.show()


			                        # croppedimage_full=image[int(n_cur_corrected[1]):int(n_cur_corrected[3]),int(n_cur_corrected[0]):int(n_cur_corrected[2])]  
			                        # plt.rcParams["figure.figsize"] = (10,5)
			                        # plt.imshow(croppedimage_full)
			                        # plt.title(f'neighbour cell images')
			                        # plt.show()
			                        
			                        print('-----Below-----\n')
			                    
			                    else:                        
			                        pass ##wrong issue


			not_overlap_error_cells =  find_overlapped_cell(error_corrected_cells, exclude_cells+exclude_cells_cor)

			new_corrected_excluded_cells, not_corrected_excluded_cells = classify_error_cells(not_overlap_error_cells, tables, masks[1], average_cordinates,0.7)
		else:
		    new_corrected_excluded_cells = []
		    not_corrected_excluded_cells = []


		img = cv.imread(img_path)
		image, height, width, _ = image_preprocessing(img)

		color = 255
		for box in tables:
		    box = list(map(int, box[0:4]))
		    put_box(image,box,(0,color,color)) # Cyan 
		    
		for box in correct_cells:
		    put_box(image,box,(0,0,color)) # Blue
		    
		for box in new_corrected_excluded_cells:
		    put_box(image,box,(0,color,0)) # Green

		for box in not_corrected_excluded_cells:
		    put_box(image,box,(color,0,0)) # Red
		    
		for box in blank_cells:   
		    put_box(image,box,(color,color,0)) # Yellow

		im_pil = Image.fromarray(image)
		im_pil.save(f"{outdirectory}/{filename}_temp_cell_multiple_cell_correction.jpg")


		considered_cells = correct_cells+new_corrected_excluded_cells

		cell_cordinates[filename] = considered_cells

