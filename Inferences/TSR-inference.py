import sys
workdir = 'Tabular-Data-Extraction'

sys.path.append(f'{workdir}/docExtractor-master/src')

import os
import numpy as np
from PIL import Image

from utils.image import resize

from utils.constant import LABEL_TO_COLOR_MAPPING
from utils.image import LabeledArray2Image
import torch

from models import load_model_from_path
from utils import coerce_to_path_and_check_exist
from utils.path import MODELS_PATH
from utils.constant import MODEL_FILE

import cv2
import cv2 as cv
import pickle as pkl

import imutils

from os import path


import matplotlib.pyplot as plt
import pickle
import torchvision.transforms.functional as F
import copy
# Add code to sys.path
import matplotlib.pyplot as plt 


import json

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

import collections

import dla.src.table_structure_analysis as tsa
import dla.src.xml_utils as xml_utils
from dla.src.image_utils import put_box, put_line



from statistics import mean

import importlib
import glosat_utils

from glosat_utils import *

TSR_path = 'active_learning'

config_file = f'{TSR_path}/mmdetection/dla/config/cascadeRCNN_ignore_all_but_cells.py'
table_checkpoint_file = f'{TSR_path}/dla_models/model_semi_supervised_fine_aclr_0/table_det/model_fulltables_only_GloSAT.pth'
table_hearder_checkpoint_file = f'{TSR_path}/dla_models/model_semi_supervised_fine_aclr_0/table_det/model_tables_enchanced_GloSAT.pth'
coarse_cell_checkpoint_file = f'{TSR_path}/dla_models/model_semi_supervised_fine_aclr_0/coarse_cell_det/best_model_manual.pth'
cell_checkpoint_file_dir = f'{TSR_path}/dla_models/model_semi_supervised_fine_aclr_0/supervised-checkpoint' #best_model_aclr_0_epc_601.pth'
maskpath = f'{TSR_path}/docExtractor_output/GloSAT-originalImage'
    
if not os.path.exists(maskpath):
    os.mkdir(maskpath)
    
cell_cordinates_info = {}
img_path = sys.argv[1] # Input image path

num_aclr = sys.argv[2] # Number of ensembled models

outdirectory = sys.argv[3] # Output directory

file = img_path.split('/')[-1]
filename = file.split('.')[0]

# outdirectory = f'{TSR_path}/dla_models/model_semi_supervised_fine_aclr_0/VOC2007/heuristics_aclr{num_aclr}_v3'

if not os.path.exists(outdirectory):
    os.mkdir(outdirectory)

outpathXML = f'{outdirectory}/XML-det'
if not os.path.exists(outpathXML):
    os.mkdir(outpathXML)
    
outpathJSON = f'{outdirectory}/JSON-det'
if not os.path.exists(outpathJSON):
    os.mkdir(outpathJSON)
    
def find_text_region(img, label_idx_color_mapping, normalize):

    im_pil = Image.fromarray(img)
    
    # Normalize and convert to Tensor
    inp = np.array(img, dtype=np.float32) / 255
    if normalize:
        inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(device)

    # compute prediction
    pred = model_extractor(inp.reshape(1, *inp.shape))[0].max(0)[1].cpu().numpy()

    # Retrieve good color mapping and transform to image
    pred_img = LabeledArray2Image.convert(pred, label_idx_color_mapping)

    # Blend predictions with original image    
    mask = Image.fromarray((np.array(pred_img) == (0, 0, 0)).all(axis=-1).astype(np.uint8) * 127 + 128)
    blend_img = Image.composite(im_pil, pred_img, mask)
    
    mask = torch.from_numpy(pred)
    mask = mask.unsqueeze(0)

    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask == obj_ids[:, None, None]

    return masks, blend_img 

def find_overlapped_cell_final_v2(corrected_cells, cells_pred, average_cordinates,pc=0.5):
    new_cells_pred = []
    large_cells = []
    missed_cells = []
    overlapped = copy.deepcopy(corrected_cells)
    
    for i, box in enumerate(cells_pred):
        count=0       
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
        tid = box[4]
        if isinstance(tid, int):
            avg_x, avg_y = average_cordinates[tid]
            
            if avg_x>0 and avg_y>0:
                if xi>(avg_x*pc*3) and yi>(avg_y*pc*3):        
                    for j, cells in enumerate(corrected_cells): 
                        xj = abs(cells[2]-cells[0])
                        yj = abs(cells[3]-cells[1])

                        if xj>(avg_x*pc*3) and yj>(avg_y*pc*3):   
                            if tsa.how_much_contained(cells,box)>pc:
                                count+=1
                                overlapped[j] = cells
                                break

                            elif tsa.how_much_contained(box, cells)>pc:
                                count+=1
                                overlapped[j] = box
                                break
                            else:
                                missed_cells.append(box)
                        else:
                            large_cells.append(box)
                else:
                    large_cells.append(box)
                        
            if count==0:
                new_cells_pred.append(box)
    seen = set()
    unique_lists = [x for x in missed_cells if tuple(x) not in seen and not seen.add(tuple(x))]

    return new_cells_pred, overlapped, large_cells,unique_lists

def find_overlapped_cell_final(corrected_cells, cells_pred, average_cordinates,pc=0.5):
    new_cells_pred = []
    large_cells = []
    overlapped = copy.deepcopy(corrected_cells)

    for i, box in enumerate(cells_pred):
        count=0       
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
        tid = box[4]
        if isinstance(tid, int):
            avg_x, avg_y = average_cordinates[tid]
            
            if avg_x>0 and avg_y>0:
                if xi>(avg_x*pc*3) and yi>(avg_y*pc*3):        
                    for j, cells in enumerate(corrected_cells): 
                        xj = abs(cells[2]-cells[0])
                        yj = abs(cells[3]-cells[1])

                        if xj>(avg_x*pc*3) and yj>(avg_y*pc*3):   
                            if tsa.how_much_contained(cells,box)>pc:
                                count+=1
                                overlapped[j] = cells
                                break

                            elif tsa.how_much_contained(box, cells)>pc:
                                count+=1
                                overlapped[j] = box
                                break
                        else:
                            large_cells.append(box)
                else:
                    large_cells.append(box)
                        
            if count==0:
                new_cells_pred.append(box)
    return new_cells_pred, overlapped, large_cells


def find_average_cellsize(tables, copy_cells, THRESHOLD=0.5, init=False, average_cordinates = []):
    #Separate cells for each tables
    table_cells = [[] for i in range(len(tables))]
    cell_size= [[[],[]] for i in range(len(tables))]
    xy_coords = [[[],[]] for i in range(len(tables))]
    
    cells = []
    if init:
        for cell in copy_cells:
            if cell[4] > THRESHOLD:
                table_idx, score = get_table_coord(cell, tables)
                if score > 0.5:
                    cell[4] = table_idx  #cells and its table index
                    if isinstance(table_idx, int):
                        table_cells[table_idx].append(cell)
                        cells.append(cell)
    else:
        for cell in copy_cells:
            table_idx = cell[4]
            if isinstance(table_idx, int):
                table_cells[table_idx].append(cell)
        
    # find average cell size
    if len(average_cordinates) == 0:
        for tid, tab_cells in enumerate(table_cells):
            if len(tab_cells)>0:
                x_size = []
                y_size = []
                for cell in tab_cells:
                    xax = abs(cell[0]-cell[2])
                    yax = abs(cell[1]-cell[3])
                    xy_coords[tid][0]+=[cell[0],cell[2]]
                    xy_coords[tid][1]+=[cell[1],cell[3]]
                    x_size.append(xax)
                    y_size.append(yax)
                x_avg = mean(x_size)
                y_avg = mean(y_size)
                average_cordinates.append((x_avg, y_avg))
                xy_coords[tid][0] = sort_coordinates(xy_coords[tid][0]+[tables[tid][0],tables[tid][2]])
                xy_coords[tid][1] = sort_coordinates(xy_coords[tid][1]+[tables[tid][1],tables[tid][3]]) 
                cell_size[tid][0] = sort_coordinates(x_size) 
                cell_size[tid][1] = sort_coordinates(y_size) 
            else:
                average_cordinates.append((0, 0))
        
    # Normalize coordinates
    for tid,tab_cells in enumerate(table_cells):
        cells_x, cells_y = xy_coords[tid]
        x_size, y_size = cell_size[tid]
        if len(cell_size[tid][0])>0 and len(cell_size[tid][1])>0:
            x_avg, y_avg = average_cordinates[tid] 
            cell_size_min = cell_size[tid][0][0]
            xy_coords[tid][0] = remove_small_coordinates(xy_coords[tid][0], cell_size_min)
            cell_size_min = cell_size[tid][1][0]
            xy_coords[tid][1] = remove_small_coordinates(xy_coords[tid][1], cell_size_min)
            
    return average_cordinates, cell_size, xy_coords, cells, table_cells


def find_coordinates(x_coords, x_avg, cell_size_pc):
    new_x_coords = []
    count = 0 

    for j in x_coords:
        if count==0:
            new_x_coords.append(j)
            count+=1
        else:
            i = new_x_coords[-1]
            if abs(i-j)>x_avg:
                new_x_coords.append(j)
                
            elif abs(i-j)>x_avg*0.9:
                new_x_coords.append(j)

            elif abs(i-j)>20 and abs(i-j)>x_avg*cell_size_pc:
                new_x_coords.append(j)

            count+=1        
    return new_x_coords

def get_coordinates(table_cells, pc, average_cordinates = []):    
    cell_cordinates = []
    new_x = []
    new_y = []
    for cells_x, cells_y in table_cells:
        if len(cells_x)>0 and len(cells_y)>0:
            cells_x = list(map(int, cells_x))
            cells_x = list(set(cells_x))
            values, _ = torch.Tensor(cells_x).int().sort()
            cells_x = values.tolist()    

            cells_y = list(map(int, cells_y))
            cells_y = list(set(cells_y))
            values, _ = torch.Tensor(cells_y).int().sort()
            cells_y = values.tolist()

            new_y = list(cells_y)
            new_y = avg_coordinates(new_y)

            new_x = list(cells_x)
            new_x = avg_coordinates(new_x)
            cell_cordinates.append([new_x,new_y])
        else:
            cell_cordinates.append([[],[]])
    
    if len(average_cordinates)==0:
        for tid, (x_cords, y_cords) in enumerate(cell_cordinates):
            
            if len(x_cords)>0 and len(y_cords)>0:

                x_size = [abs(i-j) for i, j in zip(x_cords[0:-1], x_cords[1:])]
                y_size = [abs(i-j) for i, j in zip(y_cords[0:-1], y_cords[1:])]

                x_avg = average_list(x_size)
                y_avg = average_list(y_size)

                average_cordinates.append((x_avg, y_avg))
            else:
                average_cordinates.append((0, 0))
        
    new_cell_cordinates = []
    for idx, coords in enumerate(cell_cordinates):
        x_coords, y_coords = coords
        x_avg, y_avg = average_cordinates[idx]
        if x_avg>0 and y_avg>0:
            new_x_coords = find_coordinates(x_coords, x_avg, pc)
            new_y_coords = find_coordinates(y_coords, y_avg, pc)

            new_cell_cordinates.append([new_x_coords, new_y_coords])
        else:
            new_cell_cordinates.append([[], []])
    
    return new_cell_cordinates, average_cordinates

def generate_cells(tables, cell_cordinates, average_coordinates, pc=0.7):

    exclude_cells_pred = []
    possible_cells  = []
    
    for idx, table in enumerate(tables):
        x_avg, y_avg = average_coordinates[idx]
        
        x0,y0,xn,yn = table
        cells_x, cells_y = cell_cordinates[idx]
        if len(cells_x)>0 and len(cells_y)>0:
            if cells_x[0]>x0:
                cells_x = [x0]+cells_x
            if cells_y[0]>y0:
                cells_y = [y0]+cells_y

            cells_x = [x for x in cells_x if x<xn]
            if cells_x[-1]<xn:
                cells_x+=[xn]
            cells_y = [y for y in cells_y if y<yn]
            if cells_y[-1]<yn:
                cells_y+=[yn]

            x_a = x0
            for r, x_b in enumerate(cells_x):
                if (x_b-x_a)>2:
                    row_a = [(int(x_a),y) for y in cells_y]
                    row_b = [(int(x_b),y) for y in cells_y]

                    for i in range(len(row_a)-1):
                        imagedraw = (row_a[i]+row_b[i+1])
                        imagedraw = list(map(int, imagedraw))

                        xax = abs(imagedraw[2] - imagedraw[0])
                        yax = abs(imagedraw[3] - imagedraw[1])

                        try:
                            if xax > x_avg*pc and yax > y_avg*pc:

                                possible_cells.append(imagedraw+[idx])
                                
                            else:
                                exclude_cells_pred.append(imagedraw+[idx])
                        except:
                            exclude_cells_pred.append(imagedraw+[idx])
                    x_a = int(x_b)

    return possible_cells, exclude_cells_pred

def has_object(imagedraw, mask, pc=0.005):
    imagedraw = list(map(int, imagedraw))
    h, w = mask[imagedraw[1]:imagedraw[3],imagedraw[0]:imagedraw[2]].shape
    
    count = 0
    for cell_mask in mask[imagedraw[1]:imagedraw[3],imagedraw[0]:imagedraw[2]]:
        for bol in cell_mask:  
            if bol:
                count+=1
    
    if w>2 and h>2:
        if (count/(w * h)) > pc: #5% object
            return True
        else:
            return False
    else:
        return False


def check_bounding_box(imagedraw,mask):
    scores = []
    count = 0
    key = imagedraw[1]
    for bol in mask[key,imagedraw[0]:imagedraw[2]]:  #Height
        if bol:
            count+=1
    score = count/len(mask[key,imagedraw[0]:imagedraw[2]])
    scores.append(score)
    if score<0.2: 
        top_height = True
    else:
        top_height = False
        
    count = 0
    key = imagedraw[3]
    for bol in mask[key,imagedraw[0]:imagedraw[2]]:  #Height
        if bol:
            count+=1
    score = count/len(mask[key,imagedraw[0]:imagedraw[2]])
    scores.append(score)
    if score<0.2:
        bottom_height = True
    else:
        bottom_height = False

    count = 0
    key = imagedraw[0]
    for bol in mask[imagedraw[1]:imagedraw[3],key]:  #width
        if bol:
            count+=1
    score = count/len(mask[imagedraw[1]:imagedraw[3],key])
    scores.append(score)
    if score<0.1:
        left_width = True
    else:
        left_width = False
        
    count = 0    
    key = imagedraw[2]
    for bol in mask[imagedraw[1]:imagedraw[3],key]:  #width
        if bol:
            count+=1
    score = count/len(mask[imagedraw[1]:imagedraw[3],key])
    scores.append(score)
    if score<0.1:
        right_width = True
    else:
        right_width = False
        
    return (top_height, bottom_height, left_width, right_width), scores

def check_cell(cell,average_cordinates,pc=0.8):
    tid = cell[4]
    avg_x, avg_y = average_cordinates[tid]

    if abs(cell[0]-cell[2])>avg_x*pc and abs(cell[1]-cell[3])> avg_y*pc:
        return True
    else:
        return False

def classify_cells(cells, mask, average_coordinates,pc=0.8):

    cells = [x[0:5] for x in cells]
    x = torch.IntTensor(cells)
    x = x[:,0:4]
    value, index = x[:,1].sort()

    corrected_cells_new = []
    exclude_cells_new = []
    blank_cells_new = []

    for idx in index:
        imagedraw = cells[idx][0:4]
        tid = cells[idx][4]
        imagedraw = list(map(int, imagedraw))

        x_avg,y_avg = average_coordinates[tid]
        
        if check_cell(imagedraw+[tid],average_coordinates,pc):

            boolean, scores = check_bounding_box(imagedraw,mask)

            objects = has_object(imagedraw, mask)

            if all(boolean) and objects:
                corrected_cells_new.append(cells[idx])

            elif not objects:
                blank_cells_new.append(cells[idx])

            else:
                exclude_cells_new.append(cells[idx])
        else:
            exclude_cells_new.append(imagedraw+[idx])
        # except:
        #     exclude_cells_new.append(cells[idx])

    return corrected_cells_new, blank_cells_new, exclude_cells_new

def sort_coordinates(cells_x):
    cells_x = list(map(int, cells_x))
    cells_x = list(set(cells_x))
    values, _ = torch.Tensor(cells_x).int().sort()
    cells_x = values.tolist() 
    return cells_x

def remove_small_coordinates(x_coords, cell_size_min):
    count = 0
    new_x_coords = []
    small_coords = []
    for i,j in zip(x_coords[0:-1],x_coords[1:]):
        if count == 0:
            new_x_coords.append(i)
            count+=1
        else:
            pc = abs(i-j) # next coord is larger than cellmin size
            pc2 = abs(i-new_x_coords[-1]) # Check boundary initial 
            pc3 = abs(j-new_x_coords[-1]) # Check boundary later
                        
            if pc2 > cell_size_min and len(small_coords)==0: # Check boundary initial okay?
                new_x_coords.append(i)
                    
            elif pc2 > cell_size_min and len(small_coords)>0: # Check boundary initial okay?
                if len(new_x_coords)>2:
                    mi = mean(small_coords+new_x_coords[-2:])
                    new_x_coords[-1] = int(mi)
                new_x_coords.append(i)
                small_coords = []   
                
            elif pc2 < cell_size_min and pc > cell_size_min:
                new_x_coords.append(j)
            
            elif pc < cell_size_min and pc3 > cell_size_min: # Check boundary later okay?
                new_x_coords.append(j)
                small_coords = []
                
            else:
                small_coords.append(i)
            count+=1
            
    pc2 = abs(x_coords[-1]-new_x_coords[-1]) ###check the last coords entry
    if pc2 < cell_size_min:
        new_x_coords[-1] = x_coords[-1]

    else:
        new_x_coords.append(x_coords[-1])

    return new_x_coords


def find_average_cellsize(tables, copy_cells, THRESHOLD=0.5, init=False, average_cordinates = []):
    #Separate cells for each tables
    table_cells = [[] for i in range(len(tables))]
    cell_size= [[[],[]] for i in range(len(tables))]
    xy_coords = [[[],[]] for i in range(len(tables))]
    cells = []
    if init:
        for cell in copy_cells:
            if cell[4] > THRESHOLD:
                table_idx, score = get_table_coord(cell, tables)
                if score > 0.5:
                    cell[4] = table_idx  #cells and its table index
                    if isinstance(table_idx, int):
                        table_cells[table_idx].append(cell)
                        cells.append(cell)
    else:
        for cell in copy_cells:
            table_idx = cell[4]
            if isinstance(table_idx, int):
                table_cells[table_idx].append(cell)
        
    # find average cell size
    for tid, tab_cells in enumerate(table_cells):
        # print('average_cordinates',len(average_cordinates))
        if len(tab_cells)>0:
            x_size = []
            y_size = []
            for cell in tab_cells:
                xax = abs(cell[0]-cell[2])
                yax = abs(cell[1]-cell[3])
                xy_coords[tid][0]+=[cell[0],cell[2]]
                xy_coords[tid][1]+=[cell[1],cell[3]]
                x_size.append(xax)
                y_size.append(yax)
            x_avg = mean(x_size)
            y_avg = mean(y_size)
            average_cordinates.append((x_avg, y_avg))
            xy_coords[tid][0] = sort_coordinates(xy_coords[tid][0]+[tables[tid][0],tables[tid][2]])
            xy_coords[tid][1] = sort_coordinates(xy_coords[tid][1]+[tables[tid][1],tables[tid][3]]) 
            cell_size[tid][0] = sort_coordinates(x_size) 
            cell_size[tid][1] = sort_coordinates(y_size) 
        else:
            average_cordinates.append((0, 0))
        
    # Normalize coordinates
    for tid,tab_cells in enumerate(table_cells):
        cells_x, cells_y = xy_coords[tid]
        x_size, y_size = cell_size[tid]
        if len(cell_size[tid][0])>0 and len(cell_size[tid][1])>0:
            x_avg, y_avg = average_cordinates[tid] 
            cell_size_min = cell_size[tid][0][0]
            xy_coords[tid][0] = remove_small_coordinates(xy_coords[tid][0], cell_size_min)
            cell_size_min = cell_size[tid][1][0]
            xy_coords[tid][1] = remove_small_coordinates(xy_coords[tid][1], cell_size_min)
            
    return average_cordinates, cell_size, xy_coords, cells, table_cells

def is_overlapped(overlapped_check, box,pc=0.5):
    for j, cell in enumerate(overlapped_check): 
        overlapped_score = tsa.how_much_contained(cell,box)
        overlapped_score2 = tsa.how_much_contained(box, cell)        
        if overlapped_score>pc:
            # print('Overlap Case 1',cell,box, overlapped_score, pc)
            return True
        elif overlapped_score2 > pc:
            # print('Overlap Case 2',cell, box, overlapped_score2, pc)
            return True
    return False

def get_table_coord(cell, tables):
    for idx,table in enumerate(tables):
        table = list(map(int, table))
        score = tsa.how_much_contained(cell[0:4],table)    
        if score > 0.5:
            return idx, score 
    return [], 0

def get_row_neighbour_cells(cellz, average_cordinates, pc=0.01):
    x_cords = []
    cells_idx = {}
    for box in cells:
        x_cords.append(box[0])
        if box[0] not in cells_idx:
            cells_idx[box[0]] = box
            
    merged_row_cells = merge_neighbour_rows(x_cords, cells_idx, average_cordinates,  pc)     
    return merged_row_cells  

def merge_neighbour_rows(x_cords, cells_idx, average_cordinates, pc = 0.01):
    x_cords = sorted(x_cords)
    merged_cells = []
    for xi in x_cords:
        boxi = cells_idx[xi]
        tidi = boxi[4]
        
        x_avg, y_avg = average_cordinates[tidi]
        if len(merged_cells) == 0:
            ncell = boxi  
            merged_cells.append(ncell)
        else:
            ncell = merged_cells.pop()      
            diff = abs(ncell[2]-boxi[0])
            # print(ncell, boxi, diff, x_avg*pc)     
            
            if diff<x_avg*pc:
                ncell[2] = boxi[2]           
                merged_cells.append(ncell)
            else:
                merged_cells.append(ncell)
                merged_cells.append(boxi)
    return merged_cells
 
if not os.path.exists(f'{outpathJSON}/{filename}.json'):
# if True:
    model = init_detector(config_file, table_checkpoint_file, device='cuda:0')
    tb_header_model = init_detector(config_file, table_hearder_checkpoint_file, device='cuda:0')
    coarse_cell_model = init_detector(config_file, coarse_cell_checkpoint_file, device='cuda:0')

    THRESHOLD = 0.5
    CLASSES = ("table_body","cell","full_table","header","heading")
    color = 255

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    TAG = 'default'
    model_path = coerce_to_path_and_check_exist(MODELS_PATH / TAG / MODEL_FILE)
    model_extractor, (img_size, restricted_labels, normalize) = load_model_from_path(model_path, device=device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
    _ = model_extractor.eval()

    restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in restricted_labels]
    label_idx_color_mapping = {restricted_labels.index(l) + 1: c for l, c in zip(restricted_labels, restricted_colors)}

    if not os.path.exists(f'{maskpath}/{filename}_mask.pkl'):
        print(f'Calculate text area regions of {img_path}')
        image = cv.imread(img_path)
        masks, blend_img = find_text_region(image, label_idx_color_mapping, normalize)
        blend_img.save(f'{maskpath}/{filename}_text_region.jpg')

        f = open(f'{maskpath}/{filename}_mask.pkl','wb')
        pkl.dump((masks, blend_img),f)
        f.close()
    else:
        print(f'Load text area regions of {img_path}')
        f = open(f'{maskpath}/{filename}_mask.pkl','rb')
        (masks, blend_img) = pkl.load(f)
        f.close()

    img = cv.imread(img_path)
    # Table detection
    result = inference_detector(model,img)
    result_header = inference_detector(tb_header_model,img)

    #Process table headers
    headers = []
    for box in result_header[CLASSES.index("header")]:
        if box[4]>THRESHOLD :
            headers.append(box[0:4])


    #Process table headers
    tables_body = []
    for box in result_header[CLASSES.index("table_body")]:
        if box[4]>THRESHOLD :
            tables_body.append(box[0:4])


    #Process table bodies
    tables = []
    for box in result[CLASSES.index("table_body")]:
        if box[4]>THRESHOLD :
            tables.append(box[0:4])

    #Process tables
    full_tables = []
    for box in result[CLASSES.index("full_table")]:
        if box[4]>THRESHOLD :
            full_tables.append(box[0:4])

            if all(tsa.how_much_contained(table,box)<0.5 for table in tables):
                tables.append(box[0:4])
                
                
    coarse_cells_by_image = detect_cell_pretrained(coarse_cell_model,img)
    cells_by_image = collections.defaultdict(list)
    for aclr in reversed(range(int(num_aclr)-2,int(num_aclr)+1)): 
        cell_checkpoint_file = f'{cell_checkpoint_file_dir}/aclr{aclr}.pth'
        # cell_checkpoint_file = f'{cell_checkpoint_file_dir}/best_epoch_200.pth'
        cell_model = init_detector(config_file, cell_checkpoint_file, device='cuda:0')
        cells_by_image[aclr] = detect_cell_pretrained(cell_model,img)

    cells_ori = []
    cells_ori_missed = []
    idx_cell = []

    for aclr in reversed(range(int(num_aclr)-2,int(num_aclr)+1)): 
        idx_cell = []
        for cell in cells_by_image[aclr]:
            idx, score = get_table_coord(cell, tables)
            if isinstance(idx, int):
                cell[4] =  idx
                idx_cell.append(cell)
                
        table_cells = get_x_y_initial(full_tables, cells_by_image[aclr], THRESHOLD)
        cell_cordinates, average_cordinates = get_coordinates(table_cells,0.5)

        not_overlapped, overlapped, large_cell, missed = find_overlapped_cell_final_v2(cells_ori, idx_cell, average_cordinates,pc=0.001)
        cells_ori+=not_overlapped
        not_overlapped, overlapped, large_cell, missed = find_overlapped_cell_final_v2(cells_ori_missed, missed, average_cordinates,pc=0.005)
        cells_ori_missed+=not_overlapped
        
    for cell in coarse_cells_by_image:
        idx, score = get_table_coord(cell, tables)
        cell[4] = idx
        
    not_overlapped, overlapped, large_cell, missed = find_overlapped_cell_final_v2(cells_ori, cells_ori_missed, average_cordinates,pc=0.005)
    average_cordinates, cell_sizes, xy_coords, cells, tablewise_cells = find_average_cellsize(full_tables, cells_ori+not_overlapped+coarse_cells_by_image, [])

    idx_cell = []
    cells_ori+=not_overlapped
    for cell in coarse_cells_by_image:
        idx, score = get_table_coord(cell, tables)
        cell[4] =  idx
        idx_cell.append(cell)

    not_overlapped, overlapped, large_cell, missed  = find_overlapped_cell_final_v2(cells_ori, idx_cell, average_cordinates,pc=0.005)

    average_cordinates = [(cell_size[0][0],cell_size[1][0]) for cell_size in cell_sizes]
    table_cells = get_x_y(full_tables, cells_ori+not_overlapped)
    cell_cordinates, _ = get_coordinates(table_cells,0.7,average_cordinates)

    possible_cells, exclude_cells_pred = generate_cells(full_tables, cell_cordinates, average_cordinates)

    not_overlapped = []
    overlapped = []
    for cell in possible_cells:
        if is_overlapped(cells_ori, cell[:4], pc=0.35):
            overlapped.append(cell)
        else:
            not_overlapped.append(cell)

    # not_overlapped_1, overlapped_header, large_cell, missed = find_overlapped_cell_final_v2(headers, not_overlapped, average_cordinates,pc=0.3)
    not_overlapped_1 = []
    overlapped_header = []
    for cell in not_overlapped:
        if is_overlapped(headers, cell[:4], pc=0.3):
            overlapped_header.append(cell)
        else:
            not_overlapped_1.append(cell)

    if len(overlapped_header)>0:
        corrected_hcells_new, blank_hcells_new, exclude_hcells_new = classify_cells(overlapped_header, masks[1], average_cordinates,pc=0.7)
    else:
        corrected_hcells_new, blank_hcells_new, exclude_hcells_new = ([],[],[])
        
    #Merge neighbour cells
    cellz = copy.deepcopy(exclude_hcells_new)
    same_rows = dict()
    for box in cellz:
        yi = (box[3]+box[1])/2
        if yi not in same_rows:
            same_rows[yi] = []
        same_rows[yi].append(box)   
                
    new_cells = []
    for row in same_rows:
        cells = copy.deepcopy(same_rows[row])
        new_cells += get_row_neighbour_cells(cells,average_cordinates, pc=0.01)   
        
    same_cols = dict()
    for box in new_cells:
        idx = f'{box[0]}_{box[2]}'
        if idx not in same_cols:
            same_cols[idx] = []
        same_cols[idx].append(box)   

    new_col_cells = []
    for idx in same_cols:
        ncell = []
        for cell in same_cols[idx]:
            if len(ncell) == 0:
                ncell = cell
            else:
                ncell[1] = min(ncell[1], cell[1])
                ncell[3] = max(ncell[3], cell[3])
        new_col_cells.append(ncell)


    if len(cells_ori+not_overlapped_1)>0:
        image = cv.imread(img_path)
        img, height, width, _ = image_preprocessing(img)

        color = 255
        for box in tables:
            box = list(map(int, box[0:4]))
            put_box(image,box,(0,color,color)) # Cyan 
            
        for box in headers:
            box = list(map(int, box[0:4]))
            put_box(image,box,(color,0,color), 'Header') # magenta 
            
        for box in list(cells_ori):
            put_box(image,box,(0,0,color)) # Blue
            
        for box in list(not_overlapped_1+corrected_hcells_new):
            put_box(image,box,(0,color,0)) # Green
                        
        for box in list(new_col_cells):
            put_box(image,box,(0,color,color)) # cyan
            
        im_pil = Image.fromarray(image)
        im_pil.save(f"{outdirectory}/{filename}_heuristic_correction.jpg")

        xml_utils.save_VOC_xml_from_cells(headers,headers,tables,tables,cells_ori+not_overlapped_1+corrected_hcells_new+blank_hcells_new+new_col_cells,f"{outpathXML}/{filename}.xml",width,height)
        print(f'XML save at: {outpathXML}/{filename}.xml')

        table_cells = cells_ori+not_overlapped_1+corrected_hcells_new+blank_hcells_new+new_col_cells
        # table_cells = table_cells.tolist()
        new_table_cells = []
        for cell in tables:
            if not isinstance(cell, list):
                cell = cell.tolist()
            new_table_cells.append(cell)
                    
        new_cells = []
        for cell in table_cells:
            if not isinstance(cell, list):
                cell = cell.tolist()
            new_cells.append(cell)
                    
        new_tables_body_cells = []
        for cell in tables_body:
            if not isinstance(cell, list):
                cell = cell.tolist()
            new_tables_body_cells.append(cell)

        new_headers_cells = []
        for cell in headers:
            if not isinstance(cell, list):
                cell = cell.tolist()
            new_headers_cells.append(cell)

        # Combine both lists into a dictionary for better JSON structure
        data = {
            "tables": new_table_cells,
            "table_cells": new_cells,
            'tables_body': new_tables_body_cells,
            'headers': new_headers_cells,
            'average_coordinates':average_cordinates
        }

        # Save to a JSON file
        with open( f'{outpathJSON}/{filename}.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f'XML save at: {outpathJSON}/{filename}.json')

