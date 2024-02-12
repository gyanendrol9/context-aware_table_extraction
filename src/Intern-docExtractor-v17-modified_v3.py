import cv2
import cv2 as cv
import imutils
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
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

from glosat_utils import *

config_file = '/data/glosat/glosat_table_dataset/dla/config/cascadeRCNN.py'
table_checkpoint_file = '/data/glosat/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth'
cell_checkpoint_file = '/data/glosat/glosat_table_dataset/models/model_coarsecell_GloSAT.pth'

model = init_detector(config_file, table_checkpoint_file, device='cuda:0')
cell_model = init_detector(config_file, cell_checkpoint_file, device='cuda:0')

THRESHOLD = 0.5
CLASSES = ("table_body","cell","full_table","header","heading")
color = 255

restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in restricted_labels]
label_idx_color_mapping = {restricted_labels.index(l) + 1: c for l, c in zip(restricted_labels, restricted_colors)}

directory = '/data/glosat/Code-Git/docformer/dataset/Finetuning/test/images'
outdirectory = '/data/glosat/Code-Git/docExtractor-master/demo/output_files'

file = sys.argv[1]

if 'jpg' in file:
    filename = file.replace('.jpg', '')
    img_path = f'{directory}/{filename}.jpg'
    img = cv.imread(img_path)
    img, h, w, c = image_preprocessing(img)
    masks, blend_img = find_text_region(img, label_idx_color_mapping, normalize)
    blend_img.save(f'{outdirectory}/{filename}_text_region.jpg')

    cells_by_image = collections.defaultdict(list)
    cells_by_image[filename] = detect_cell_pretrained(cell_model,img)

    full_tables, tables = detect_table_region(model,img)

    ori_correct_cells, exclude_cells = classify_single_cell(cells_by_image[filename], masks[1])
    table_cells = get_x_y_initial(tables, cells_by_image[filename], THRESHOLD)
    cell_cordinates, average_cordinates = get_coordinates(table_cells,0.7)

    # Average table cell in a page if multiple table exists 
    x_avg = []
    y_avg = []
    for xa,ya in average_cordinates:
        x_avg.append(xa)
        y_avg.append(ya)
        
    average_cordinates.append((mean(x_avg),mean(y_avg)))

    new_ori_correct_cells = []
    for cell in ori_correct_cells:
        idx, score = get_table_coord_initial(cell, tables)
        if score>0.5:
            cell[4] = idx
            cell = list(map(int, cell))
            new_cell, changed = correct_boundary_single_cell(cell, masks[1], tables[idx], average_cordinates[-1], 0.7)
        new_ori_correct_cells.append(new_cell+[idx])

    ori_correct_cells =  list(new_ori_correct_cells)

    possible_cells, exclude_cells_pred = generate_cells(masks[1], tables, cell_cordinates, average_cordinates)
    # possible_cells = find_overlapped_cell(ori_correct_cells, possible_cells, average_cordinates)

    if len(possible_cells)>0:
        correct_cells, blank_cells, exclude_cells = classify_cells(possible_cells, masks[1], average_cordinates) 
    else:
        correct_cells, blank_cells, exclude_cells = classify_cells(ori_correct_cells, masks[1], average_cordinates)

    correct_cells, ori_correct_cells = find_overlapped_cell(ori_correct_cells, correct_cells, average_cordinates)
    considered_cells = ori_correct_cells+correct_cells

    not_ovellapped_blank_cells_pred, considered_cells = find_overlapped_cell(considered_cells, blank_cells, average_cordinates)
    considered_cells = considered_cells+not_ovellapped_blank_cells_pred

    not_ovellapped_exclude_cells_pred, considered_cells = find_overlapped_cell(considered_cells, exclude_cells+exclude_cells_pred, average_cordinates)

    all_valid_cells = considered_cells

    if len(not_ovellapped_exclude_cells_pred)>2:
        corrected_cells_new, error_cells = classify_error_cells(not_ovellapped_exclude_cells_pred, tables, masks[1], average_cordinates,0.7)
        if len(corrected_cells_new)>0:
            corrected_cells_new, blank_cells_new, exclude_cells_new = classify_cells(corrected_cells_new, masks[1], average_cordinates)
        else:
            corrected_cells_new = []
            blank_cells_new = []
            exclude_cells_new = []
    else:
        corrected_cells_new = []
        blank_cells_new = []
        exclude_cells_new = []
        error_cells = []
        
    exclude_cells_new, corrected_cells_new = find_overlapped_cell(corrected_cells_new, not_ovellapped_exclude_cells_pred, average_cordinates)
    all_valid_cells += corrected_cells_new

    table_cells = get_x_y(tables, all_valid_cells) 
    cell_cordinates, average_cordinates = get_coordinates(table_cells,0.7)

    possible_cells, exclude_cells_pred = generate_cells(masks[1], tables, cell_cordinates, average_cordinates)
    possible_cells, all_valid_cells = find_overlapped_cell(all_valid_cells, possible_cells, average_cordinates)

    if len(possible_cells)>0:
        correct_cells, blank_cells, exclude_cells = classify_cells(possible_cells, masks[1], average_cordinates) 
    else:
        correct_cells, blank_cells, exclude_cells = classify_cells(all_valid_cells, masks[1], average_cordinates)

    correct_cells, all_valid_cells = find_overlapped_cell(all_valid_cells, correct_cells, average_cordinates)
    blank_cells, all_valid_cells = find_overlapped_cell(all_valid_cells, blank_cells, average_cordinates)
    exclude_cells, all_valid_cells = find_overlapped_cell(all_valid_cells, exclude_cells, average_cordinates)

    new_valid_cells = correct_cells+blank_cells
    new_valid_cells, all_valid_cells = find_overlapped_cell(all_valid_cells, new_valid_cells, average_cordinates)
    all_valid_cells += new_valid_cells

    not_corrected_excluded_cells, all_valid_cells = find_overlapped_cell(all_valid_cells, exclude_cells, average_cordinates)


    x_y_neighbouring_cells_dict = {}
    if len(not_corrected_excluded_cells)>2:
        neighbouring_cells, y_cords = get_x_y_neighbour_cells(not_corrected_excluded_cells, tables, average_cordinates)
        
        for yid in neighbouring_cells:
            cells = neighbouring_cells[yid]

            y_neighbours = cells[0]
            x_neighbours = cells[1]

            tid =x_neighbours[0][4]

            avg_x, avg_y = average_cordinates[tid]

            # print(yid,y_neighbours,x_neighbours,tid, average_cordinates[tid])

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

        #     print(x_neighbours,'---',ncells)   
            x_y_neighbouring_cells_dict[yid] = (y_neighbours, ncells)


    y_neighbours_new = {}
    if len(x_y_neighbouring_cells_dict)>2:
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

        for yid in y_neighbours:
            for yidn in y_neighbours[yid]:
                y_neighbours_new[yidn] = y_neighbours[yid]
        #         if yidn not in y_neighbours:
        #             print(y_neighbours[yid])

    # combine x and y neighbours v2

    error_corrected_cells = []
    not_corrected_cells = []

    if len(x_y_neighbouring_cells_dict)>2:
        for yid in x_y_neighbouring_cells_dict:
        #     print(yid)
            ynid, xnid = x_y_neighbouring_cells_dict[yid]

            if len(ynid) == 0:             
                new_corrected, not_corrected = classify_error_cells(xnid, tables, masks[1], average_cordinates,0.7) # correct single error cell or declare error            
                if len(error_corrected_cells)>0:
                    new_corrected, error_corrected_cells = find_overlapped_cell(error_corrected_cells, new_corrected, average_cordinates)
                
                error_corrected_cells+=new_corrected

                if len(not_corrected)>0:
                    for cell in not_corrected:                
                        tid = cell[4]
                        avg_x, avg_y = average_cordinates[tid]
                        tab_boundary = list(map(int, tables[tid]))

                        move = cell[5]

                        for idx,i in enumerate(move):
                            if i == 0:
                                if abs(cell[idx]-tab_boundary[idx])<avg_y/3:
                                    cell[idx] = tab_boundary[idx]
                                    move[idx] = 5

                        if 0 in move:
                            not_corrected_cells+=not_corrected

                        else:
                            cell[5] =  move

                            if len(error_corrected_cells)>0:
                                cell, error_corrected_cells = find_overlapped_cell(error_corrected_cells, [cell], average_cordinates)
                                error_corrected_cells+=cell
        #                     print(cell, imagedraw[idx],tab_boundary[idx], avg_y/3, abs(cell[idx]-tab_boundary[idx]))

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

                    step_x = int(avg_x*0.025)
                    step_y = int(avg_y*0.025)

                    tab_boundary = list(map(int, tables[tid]))

                    #check above first:
                    for a_cell in above_cells:
                        anchor_a = list(a_cell)
                        anchor_a[1] = anchor[1] # change the y-coordinates to see overlapping or not and if overlapped the cell is immediate neighbour
                        anchor_a[3] = anchor[3]
                        score1 = tsa.how_much_contained(anchor_a, cell)
                        score2 = tsa.how_much_contained(cell,anchor_a)
                        
                        print('1 overlap score:',score1,score2)

                        if score1>0.5:
                            merged_cell = list(a_cell)
                            merged_cell[3] = cell[3]
                        
                            print('1 cell info:',a_cell, cell, merged_cell)
                            
                            # croppedimage_full=image[int(merged_cell[1]):int(merged_cell[3]),int(merged_cell[0]):int(merged_cell[2])] 
                            # plt.rcParams["figure.figsize"] = (10,5)
                            # plt.imshow(croppedimage_full)
                            # plt.title(f'1 {yid} merged cell of {merged_cell} #images')
                            # plt.show()   

    #                         if a_cell[0] < cell[0]:  ## check if a_cell[0] >= cell[0] and change a_cell[0] to cell[0] and a_cell[2] to cell[2] for correction

                            print(f'1--{yid} just above cell: {a_cell}, Current cell: {cell}, {anchor_a}, {score1}')
                            above = True

                            cur_corrected = segment_image(merged_cell, masks[1], average_cordinates)

                            merged_cell = list(cell)
                            merged_cell[1] = a_cell[1]
                            cur_correct_final =[]

                            for cur_cell in cur_corrected:

                                cur_cell[0]+=step_x
                                cur_cell[1]+=step_y
                                cur_cell[2]+=step_x
                                cur_cell[3]+=step_y
                        
                                print('1 cur_cell info:',cur_cell)
                                if check_cell(cur_cell,average_cordinates):
                                    # croppedimage_full=image[int(cur_cell[1]):int(cur_cell[3]),int(cur_cell[0]):int(cur_cell[2])]  
                                    # plt.rcParams["figure.figsize"] = (10,5)
                                    # plt.imshow(croppedimage_full)
                                    # plt.title(f'1 current cell images')
                                    # plt.show()
                                    cur_correct_final.append(cur_cell)
                            
                            if len(error_corrected_cells)>0:
                                cur_correct_final, error_corrected_cells = find_overlapped_cell(error_corrected_cells, cur_correct_final, average_cordinates)
                                error_corrected_cells += cur_correct_final

                            print('-----Above-----\n')
        #                         pass

        #                         croppedimage_full=image[int(a_cell[1]):int(a_cell[3]),int(a_cell[0]):int(a_cell[2])]  ## check if a_cell[0] >= cell[0] and change a_cell[0] to cell[0] and a_cell[2] to cell[2] for correction
        #                         plt.rcParams["figure.figsize"] = (10,5)
        #                         plt.imshow(croppedimage_full)
        #                         plt.title(f'Issue with above cell image')
        #                         plt.show()

        #                         croppedimage_full=image[int(cell[1]):int(cell[3]),int(cell[0]):int(cell[2])]
        #                         plt.rcParams["figure.figsize"] = (10,5)
        #                         plt.imshow(croppedimage_full)
        #                         plt.title(f'Current cell image')
        #                         plt.show()


                    #check below cell:
                    for b_cell in below_cells:
                        anchor_b = list(b_cell)
                        anchor_b[1] = anchor[1] # change the y-coordinates to see overlapping or not and if overlapped the cell is immediate neighbour
                        anchor_b[3] = anchor[3]
                        score1 = tsa.how_much_contained(cell,anchor_b)
                        score2 = tsa.how_much_contained(anchor_b, cell)

                        if score1>0.5:
                            
                            merged_cell = list(b_cell)
                            merged_cell[1] = cell[1]
                            cur_correct_final =[]

                            cur_corrected = segment_image(merged_cell, masks[1], average_cordinates)
                            print(f'2--{yid} just below cell: {b_cell}, Current cell: {cell}, {anchor_b} {score1}')

                            merged_cell = list(cell)
                            # merged_cell[3] = b_cell[3]

                            # croppedimage_full=image[int(merged_cell[1]):int(merged_cell[3]),int(merged_cell[0]):int(merged_cell[2])]  
                            # plt.rcParams["figure.figsize"] = (10,5)
                            # plt.imshow(croppedimage_full)
                            # plt.title(f'2 {yid} merged cell of {len(y_cords[yid])} #images')
                            # plt.show()

                            for cur_cell in cur_corrected:

                                cur_cell[0]+=step_x
                                cur_cell[1]+=step_y
                                cur_cell[2]+=step_x
                                cur_cell[3]+=step_y
                        
                                print('2 cur_cell info:',cur_cell)
                                if check_cell(cur_cell,average_cordinates):
                                    # croppedimage_full=image[int(cur_cell[1]):int(cur_cell[3]),int(cur_cell[0]):int(cur_cell[2])]  
                                    # plt.rcParams["figure.figsize"] = (10,5)
                                    # plt.imshow(croppedimage_full)
                                    # plt.title(f'2 current cell images')
                                    # plt.show()
                                    cur_correct_final.append(cur_cell)
                            
                            if len(error_corrected_cells)>0:
                                cur_correct_final, error_corrected_cells = find_overlapped_cell(error_corrected_cells, cur_correct_final, average_cordinates)
                                error_corrected_cells += cur_correct_final

                            print('-----Below-----\n')

    new_corrected_excluded_cells = []
    not_corrected_cells = []

    error_corrected_cells, all_valid_cells =  find_overlapped_cell(all_valid_cells, error_corrected_cells, average_cordinates)

    considered_cells = all_valid_cells+error_corrected_cells
    if len(error_corrected_cells)>0:
        not_overlap_error_cells, considered_cells =  find_overlapped_cell(considered_cells, not_corrected_excluded_cells, average_cordinates)
        new_corrected_excluded_cells, not_corrected_cells = classify_error_cells(not_overlap_error_cells, tables, masks[1], average_cordinates,0.7)

    not_corrected_cells, not_corrected_excluded_cells =  find_overlapped_cell(not_corrected_excluded_cells, not_corrected_cells, average_cordinates)

    considered_cells = all_valid_cells + error_corrected_cells+new_corrected_excluded_cells
    not_corrected_excluded_cells, considered_cells =  find_overlapped_cell(considered_cells, not_corrected_excluded_cells+not_corrected_cells, average_cordinates)

    # img = cv.imread(img_path)
    # image, height, width, _ = image_preprocessing(img)

    # color = 255
    # for box in tables:
    #     box = list(map(int, box[0:4]))
    #     put_box(image,box,(0,color,color)) # Cyan 
        
    # for box in all_valid_cells:
    #     put_box(image,box,(0,0,color)) # Blue
        
    # for box in error_corrected_cells:
    #     put_box(image,box,(0,color,0)) # Green

    # for box in not_corrected_excluded_cells:
    #     put_box(image,box,(color,0,0)) # Red
        
    # for box in new_corrected_excluded_cells:
    #     put_box(image,box,(color,0,color))  # pink
        
    # for box in blank_cells:   
    #     put_box(image,box,(color,color,0)) # Yellow

    # im_pil = Image.fromarray(image)
    # im_pil.save(f"{outdirectory}/{filename}_temp_cell_multiple_cell_correction.jpg")

    all_valid_cells = all_valid_cells+error_corrected_cells+new_corrected_excluded_cells
    not_corrected_excluded_cells, all_valid_cells =  find_overlapped_cell(all_valid_cells, not_corrected_excluded_cells, average_cordinates)#

    correct_cells, blank_cells_new, exclude_cells_new = classify_cells(all_valid_cells, masks[1], average_cordinates)
    new_corrected_excluded_cells, not_corrected_excluded_cells = classify_error_cells(exclude_cells_new+not_corrected_excluded_cells, tables, masks[1], average_cordinates,0.7)

    exclude_cells_new, new_corrected_excluded_cells =  find_overlapped_cell(new_corrected_excluded_cells, not_corrected_excluded_cells, average_cordinates)#

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

    for box in exclude_cells_new:
        put_box(image,box,(color,0,0)) # Red
        
    # # for box in new_corrected_excluded_cells:
    # #     put_box(image,box,(color,0,color))  # pink
        
    for box in blank_cells_new:   
        put_box(image,box,(color,color,0)) # Yellow

    im_pil = Image.fromarray(image)
    im_pil.save(f"{outdirectory}/{filename}_temp_cell_multiple_cell_correction_v2.jpg")


    f = open(f"{outdirectory}/{filename}_cell_bounding_box_v2.pkl", 'wb')
    pickle.dump((correct_cells+new_corrected_excluded_cells, blank_cells_new, exclude_cells_new, tables),f)
    f.close()
