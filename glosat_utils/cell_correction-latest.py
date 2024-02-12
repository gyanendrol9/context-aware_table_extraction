from statistics import mean

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2 as cv
import collections
import matplotlib.pyplot as plt

import dla.src.table_structure_analysis as tsa
import dla.src.xml_utils as xml_utils
from dla.src.image_utils import put_box, put_line
import pytesseract
import numpy as np
import torch

THRESHOLD = 0.5
CLASSES = ("table_body","cell","full_table","header","heading")

def detect_cell_pretrained(cell_model,img):
    result = inference_detector(cell_model, img)
    return result[CLASSES.index("cell")].tolist()


def has_object(imagedraw, mask):
    imagedraw = list(map(int, imagedraw))
    h, w = mask[imagedraw[1]:imagedraw[3],imagedraw[0]:imagedraw[2]].shape
    
    count = 0
    for cell_mask in mask[imagedraw[1]:imagedraw[3],imagedraw[0]:imagedraw[2]]:
        for bol in cell_mask:  
            if bol:
                count+=1
    
    if w>2 and h>2:
        if 100*(count/(w * h)) > 1:
            return True
        else:
            return False
    else:
        return False

def classify_single_cell(cells, mask):    
    x = torch.IntTensor(cells)
    x = x[:,0:4]
    value, index = x[:,1].sort()

    corrected_cells = []
    exclude_cells = []
    blank_cells = []

    for idx in index:
        imagedraw = cells[idx][0:4]
        imagedraw = list(map(int, imagedraw))

        boolean, scores = check_bounding_box(imagedraw,mask)

        if all(boolean):
            corrected_cells.append(cells[idx])
        else:
            exclude_cells.append(cells[idx])
    return corrected_cells, exclude_cells


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

def detect_table_region(model,image):
    result = inference_detector(model,image)
    #Process table headings
    headings = []
    for box in result[CLASSES.index("heading")]:
        if box[4]>THRESHOLD :
            headings.append(box[0:4])

    #Process table headers
    headers = []
    for box in result[CLASSES.index("header")]:
        if box[4]>THRESHOLD :
            headers.append(box[0:4])

    #Process table bodies
    tables = []
    for box in result[CLASSES.index("table_body")]:
        if box[4]>THRESHOLD :
            tables.append(box[0:4])

    full_tables = []
    for box in result[CLASSES.index("full_table")]:
        if box[4]>THRESHOLD :
            full_tables.append(box[0:4])
            if all(tsa.how_much_contained(table,box)<0.5 for table in tables):
                tables.append(box[0:4])
                
    for table in tables:
        if all(tsa.how_much_contained(table,full_table)<0.5 for full_table in full_tables):
            full_tables.append(table)
            
    return full_tables, tables

def get_x_y_initial(tables, cells, THRESHOLD):   
    segment_headers =  False
    table_cells = []
    for tid, table in enumerate(tables):
        cells_x = []
        cells_y = []
        for box in cells:  #cells_by_image[img_path]: #
            cell = box[0:4] #(x0,y0,x1,y1)

            if box[4]>THRESHOLD:
                if tsa.how_much_contained(cell,table)>0.5:
                    cells_x.append(cell[0])
                    cells_x.append(cell[2])
                    cells_y.append(cell[1])
                    cells_y.append(cell[3])
                else:
                    pass
        table_cells.append([cells_x, cells_y])
    return table_cells

def get_x_y(tables, cells):   
    segment_headers =  False
    table_cells = []
    for table in tables:
        cells_x = []
        cells_y = []
        for box in cells:  #cells_by_image[img_path]: #
            cell = box[0:4] #(x0,y0,x1,y1)
            
            if tsa.how_much_contained(cell,table)>0.5:
                cells_x.append(cell[0])
                cells_x.append(cell[2])
                cells_y.append(cell[1])
                cells_y.append(cell[3])
            else:
                pass
        table_cells.append([cells_x, cells_y])
    return table_cells  

def average_list(cells):        
    x_avg = 0
    for i in cells:
        x_avg+=i
    x_avg = x_avg/len(cells)
    return x_avg

def find_coordinates(x_coords, x_avg, cell_size_pc):
    new_x_coords = []
    flag_j = False
    last_flag = True
    count = 0 

    for i,j in zip(x_coords[0:-1],x_coords[1:]):
        if count==0:
            new_x_coords.append(i)

        if abs(i-j)>x_avg*cell_size_pc:
            new_x_coords.append(j)
            flag_j = True
            last_flag = True

        elif not last_flag:
            new_x_coords.append(j)
            last_flag = True

        else:
            flag_j = False
            last_flag =  False

        count+=1
    return new_x_coords

def avg_coordinates(cells_x,pc=0.8):
    new_x = []   
    avg = mean([j-i for i,j in zip(cells_x[0:-1],cells_x[1:])])*pc  ###compare with 1/3 size of the average cell size
    
    while len(cells_x)>0:
        x=cells_x.pop()
        
        if len(new_x)>0 and abs(x - new_x[-1])<avg:  # last insert is near to current value then skip current value pop out
            new_x[-1] = int((x + new_x[-1])/2)
        else:
            if len(cells_x)>0:
                if abs(x - cells_x[-1])<avg:
                    n_c = int((x + cells_x[-1])/2)
                    x = cells_x.pop()
                    new_x.append(n_c)
                else:
                    new_x.append(x)
            else:
                new_x.append(x)

    new_x = [i for i in reversed(new_x)] 
    return new_x

def get_coordinates(table_cells, pc):    
    cell_cordinates = []
    new_x = []
    new_y = []
    for cells_x, cells_y in table_cells:
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
        
    average_cordinates = []
    for tid, (x_cords, y_cords) in enumerate(cell_cordinates):

        x_size = [abs(i-j) for i, j in zip(x_cords[0:-1], x_cords[1:])]
        y_size = [abs(i-j) for i, j in zip(y_cords[0:-1], y_cords[1:])]

        x_avg = average_list(x_size)
        y_avg = average_list(y_size)

        average_cordinates.append((x_avg, y_avg))
        
    new_cell_cordinates = []
    for idx, coords in enumerate(cell_cordinates):
        x_coords, y_coords = coords
        x_avg, y_avg = average_cordinates[idx]
        new_x_coords = find_coordinates(x_coords, x_avg, pc)
        new_y_coords = find_coordinates(y_coords, y_avg, pc)

        new_cell_cordinates.append([new_x_coords, new_y_coords])
    
    return new_cell_cordinates, average_cordinates

# average nearby corodinates:
def find_average_coordinates(cells_x):
    new_x = []   
    avg = mean([abs(i-j) for i,j in zip(cells_x[0:-1],cells_x[1:])])/3
    
    while len(cells_x)>0:
        x=cells_x.pop()
        
        if len(new_x)>0 and abs(x - new_x[-1])<avg:  # last insert is near to current value then skip current value pop out
            new_x[-1] = int((x + new_x[-1])/2)
        else:
            if len(cells_x)>0:
                if abs(x - cells_x[-1])<avg:
                    n_c = int((x + cells_x[-1])/2)
                    x = cells_x.pop()
                    new_x.append(n_c)
                else:
                    new_x.append(x)
            else:
                new_x.append(x)
    
    new_x = [i for i in reversed(new_x)] 
    return new_x

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
            exclude_cells_new.append(cells[idx])

    return corrected_cells_new, blank_cells_new, exclude_cells_new

def merge_cells(cells, tables, average_cordinates):    
    x_y_neighbouring_cells_dict = {}
    if len(cells)>2:
        neighbouring_cells, y_cords, x_cords = get_x_y_neighbour_cells(cells, tables, average_cordinates)

        for yid in neighbouring_cells:
            cells = neighbouring_cells[yid]

            x_neighbours = cells[1]

            tid = x_neighbours[0][4]

            avg_x, avg_y = average_cordinates[tid]

            # print(yid,y_neighbours,x_neighbours,tid, average_cordinates[tid])

            # find x_neighbours
            ncells = []
            imagedraw = []
            next_cell_begin = True
            starting_cell = []

            y_neighbours = []
            for y_nid in cells[0]:
                if abs(y_nid - yid) < (avg_y + avg_y * 0.2):  # Filter column neighbour ids
                    y_neighbours.append(y_nid)

            for xid, xcell in enumerate(x_neighbours):  # Filter row neighbour cells
                # Either split the cell or merge to begining cell
                if next_cell_begin:
                    starting_cell = list(xcell)
                elif abs(starting_cell[2] - xcell[0]) < avg_x * 0.01:
                    starting_cell[2] = xcell[2]

                # check next cell distance
                if (xid + 1) < len(x_neighbours):
                    next_cell = x_neighbours[xid + 1]
                    if abs(next_cell[0] - xcell[2]) < avg_x * 0.01:
                        next_cell_begin = False
                    else:
                        ncells.append(starting_cell)
                        starting_cell = []
                        next_cell_begin = True
                elif len(x_neighbours) == 1:
                    ncells.append(starting_cell)
                    starting_cell = []

            if len(starting_cell) > 0:
                ncells.append(starting_cell)

            #     print(x_neighbours,'---',ncells)
            x_y_neighbouring_cells_dict[yid] = (y_neighbours, ncells)

    # combine x and y neighbours v2
    neighbours = []
    if len(x_y_neighbouring_cells_dict) > 0:
        for yid in x_y_neighbouring_cells_dict:
            ynid, xnid = x_y_neighbouring_cells_dict[yid]
            above_cells = []
            cur_cells = []
            below_cells = []
            for y_n in ynid:
                if y_n < yid:
                    above_cells = x_y_neighbouring_cells_dict[y_n][1]
                else:
                    cur_cells = xnid
                    if y_n > yid:
                        below_cells = x_y_neighbouring_cells_dict[y_n][1]

            print(
                f'============{yid}================== \nAbove cells:{above_cells} \nCurrent cells: {xnid} \nBelow cells: {below_cells}')

            for cur_cell2 in xnid:
                xc0, yc0, xc1, yc1, tid = cur_cell2[0:5]
                avg_x, avg_y = average_cordinates[tid]
                cur_neigh = []
                for cur_cell_a in above_cells:
                    xa0, ya0, xa1, ya1, tid_a = cur_cell_a[0:5]
                    cell_a = [xa0, ya0, xa1, ya1]
                    cell_b = [xc0, ya0, xc1, ya1]

                    if tsa.how_much_contained(cell_a, cell_b) > 0.5:
                        cur_neigh.append(cur_cell_a)
                    elif tsa.how_much_contained(cell_b, cell_a) > 0.5:
                        cur_neigh.append(cur_cell_a)

                cur_neigh.append(cur_cell2)
                for cur_cell_b in below_cells:
                    xb0, yb0, xb1, yb1, tid_b = cur_cell_b[0:5]
                    cell_a = [xc0, yc0, xc1, yc1]
                    cell_b = [xb0, yc0, xb1, yc1]

                    if tsa.how_much_contained(cell_a, cell_b) > 0.5:
                        cur_neigh.append(cur_cell_b)
                    elif tsa.how_much_contained(cell_b, cell_a) > 0.5:
                        cur_neigh.append(cur_cell_b)

                neighbours.append(cur_neigh)


    merged_cells = []
    for neigh_cells in neighbours:
        if len(neigh_cells)>1:
            cell = neigh_cells[0]
            end_cell =  neigh_cells[-1]
            print(cell, end_cell)
            cur_cell2 = list(cell)
            cur_cell2[3] = end_cell[3]
            merged_cells.append(cur_cell2)
        else:
            merged_cells+=neigh_cells
            
    return merged_cells


def generate_cells(mask, tables, cell_cordinates, average_coordinates):

    exclude_cells_pred = []
    possible_cells  = []
    
    for idx, table in enumerate(tables):
        x_avg, y_avg = average_coordinates[idx]
        
        x0,y0,xn,yn = table
        cells_x, cells_y = cell_cordinates[idx]
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

                select_cells = []

                for i in range(len(row_a)-1):
                    imagedraw = (row_a[i]+row_b[i+1])
                    imagedraw = list(map(int, imagedraw))

                    xax = abs(imagedraw[2] - imagedraw[0])
                    yax = abs(imagedraw[3] - imagedraw[1])

                    if xax > x_avg/2 and yax > y_avg/2:

                        boolean, scores = check_bounding_box(imagedraw,mask)

                        if all(boolean):
                            possible_cells.append(imagedraw+[idx])
                        else:
                            exclude_cells_pred.append(imagedraw+[idx])
                    else:
                        exclude_cells_pred.append(imagedraw+[idx])
                        
                x_a = int(x_b)
            
    return possible_cells, exclude_cells_pred

# def check_cell(cell,average_cordinates):
#     tid = cell[4]
#     avg_x, avg_y = average_cordinates[tid]

#     if abs(cell[0]-cell[2])>avg_x/3 and abs(cell[1]-cell[3])> avg_y/3:
#         return True
#     else:
#         return False

def check_cell(cell,average_cordinates,pc=0.8):
    tid = cell[4]
    avg_x, avg_y = average_cordinates[tid]

    if abs(cell[0]-cell[2])>avg_x*pc and abs(cell[1]-cell[3])> avg_y*pc:
        return True
    else:
        return False

def filter_overlapped_cell(corrected_cells, average_cordinates):
    new_cells_pred = []
    corrected_cells = filter_cells_overlap_check(corrected_cells, average_cordinates)
    for i, box in enumerate(corrected_cells):
        count=0       
        tid = box[4]
        avg_x, avg_y = average_cordinates[tid]
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
        
        if len(new_cells_pred)==0:
            new_cells_pred.append(box)
        else:
            for j, cells in enumerate(new_cells_pred): 
                xj = abs(cells[2]-cells[0])
                yj = abs(cells[3]-cells[1])

                if tsa.how_much_contained(cells,box)>0.5:
                    if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                        count+=1
                        break

                elif tsa.how_much_contained(box, cells)>0.5:
                    if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                        count+=1
                        break

            if count==0:
                new_cells_pred.append(box)
    return new_cells_pred

def filter_overlapped_cell_final(corrected_cells, average_cordinates):
    new_cells_pred = []
    large_cells = []
    corrected_cells = filter_cells_overlap_check(corrected_cells, average_cordinates)
    for i, box in enumerate(corrected_cells):
        count=0       
        tid = box[4]
        avg_x, avg_y = average_cordinates[tid]
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
                
        if xi/avg_x<3 and yi/avg_y<3:
            if len(new_cells_pred)==0:
                new_cells_pred.append(box)
            else:
                for j, cells in enumerate(new_cells_pred): 
                    xj = abs(cells[2]-cells[0])
                    yj = abs(cells[3]-cells[1])

                    if tsa.how_much_contained(cells,box)>0.5:
#                         if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                        count+=1
                        break

                    elif tsa.how_much_contained(box, cells)>0.5:
#                         if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                        count+=1
                        break

        else:
            large_cells.append(box)
            count+=1
            
        if count==0:
            new_cells_pred.append(box)
            
    return new_cells_pred,large_cells

def find_overlapped_cell_final(corrected_cells, cells_pred, average_cordinates):
    new_cells_pred = []
    for i, box in enumerate(cells_pred):
        count=0       
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
        
        for j, cells in enumerate(corrected_cells): 
            xj = abs(cells[2]-cells[0])
            yj = abs(cells[3]-cells[1])
            
            if tsa.how_much_contained(cells,box)>0.5:
#                     if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                count+=1
                if xj < xi:
                    cells[0] = box[0]  
                    cells[2] = box[2]
                if yj < yi:
                    cells[1] = box[1]  
                    cells[3] = box[3]
                corrected_cells[j] = cells
                break
                
            elif tsa.how_much_contained(box, cells)>0.5:
#                     if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                count+=1
                if xj < xi:
                    cells[0] = box[0]  
                    cells[2] = box[2]
                if yj < yi:
                    cells[1] = box[1]  
                    cells[3] = box[3]
                corrected_cells[j] = cells
                break
                    
        if count==0:
            new_cells_pred.append(box)
    return new_cells_pred, corrected_cells
    
def filter_cells_overlap_check(cells_pred, average_cordinates):
    new_cells_pred = []
    for i, box in enumerate(cells_pred):
        count=0       
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
        tid = box[4]
        x_avg,y_avg = average_cordinates[tid]

        if check_cell(box, average_cordinates):
            new_cells_pred.append(box)
        elif xi>x_avg*0.5 and yi > y_avg*0.5:
            new_cells_pred.append(box)
    return new_cells_pred

def find_overlapped_cell(corrected_cells, cells_pred, average_cordinates):
    new_cells_pred = []
    cells_pred = filter_cells_overlap_check(cells_pred, average_cordinates)
    for i, box in enumerate(cells_pred):
        count=0       
        xi = abs(box[2]-box[0])
        yi = abs(box[3]-box[1])
        
        for j, cells in enumerate(corrected_cells): 
            xj = abs(cells[2]-cells[0])
            yj = abs(cells[3]-cells[1])
            
            if tsa.how_much_contained(cells,box)>0.5:
                if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                    count+=1
                    if xj < xi:
                        cells[0] = box[0]  
                        cells[2] = box[2]
                    if yj < yi:
                        cells[1] = box[1]  
                        cells[3] = box[3]
                    corrected_cells[j] = cells
                break
                
            elif tsa.how_much_contained(box, cells)>0.5:
                if (max(yj,yi)/min(yj,yi))<3 and (max(xj,xi)/min(xj,xi))<3:
                    count+=1
                    if xj < xi:
                        cells[0] = box[0]  
                        cells[2] = box[2]
                    if yj < yi:
                        cells[1] = box[1]  
                        cells[3] = box[3]
                    corrected_cells[j] = cells
                break
                    
        if count==0:
            new_cells_pred.append(box)
    return new_cells_pred, corrected_cells

def segment_image(cell, mask, average_cordinates,table,pc=0.4,segmenter=0.05):   

    roi = mask[cell[1]:cell[3],cell[0]:cell[2]]
    h,w = roi.shape

    tid = cell[4]
    avg_x, avg_y = average_cordinates[tid]
    tx0, ty0, tx1, ty1 = table[tid]
    
    # check row by row first
    seg_x_region = []
    starting_flag = True
    count = 0
    if h*0.8 > avg_y:
        for i in range(h):
            if sum(roi[i,:]) > w*segmenter and starting_flag:
                if count>0:
                    last = seg_x_region[-1]
                    seg_x_region.append(i-last*0.05)
                else:
                    seg_x_region.append(i*0.2)
                    count+=1
                starting_flag = False
            elif sum(roi[i,:]) < w*segmenter and not starting_flag and i>avg_y*0.8:
                seg_x_region.append(i+1)
                starting_flag = True
        if not starting_flag and len(seg_x_region)<=1:
            seg_x_region.append(h)
    else:
        seg_x_region+=[0+h*0.05,h]

    new_row_cells = []

    if len(seg_x_region)>1:
        last_end = 0
        flag = False
        for start,end in zip(seg_x_region[0:-1],seg_x_region[1:]):
            if abs(start - end)>avg_y*pc:
                if last_end == 0:
                    new_row_cells.append(cell[1]+start) 
                    last_end = end
                else:
                    start = int((last_end+start)/2)
                    new_row_cells.append(cell[1]+start) 
                    last_end = end
                flag = True
            else:
                flag = False

        if flag:
            new_row_cells.append(cell[1]+end) 

    if len(new_row_cells)<2:
        new_row_cells = [cell[1],cell[3]]     
    else:
        if abs(new_row_cells[0]-ty0)<avg_y*0.7:
            new_row_cells = find_coordinates([cell[1]]+new_row_cells, avg_y, 0.7)
        if abs(new_row_cells[-1]-ty1)<avg_y*0.7:
            new_row_cells = find_coordinates(new_row_cells[0:-1]+[cell[3]], avg_y, 0.7)
        else:
            new_row_cells = find_coordinates(new_row_cells+[cell[3]], avg_y, 0.7)  
        
    # check col by col 
    seg_y_region = []
    starting_flag = True
    count = 0

    if w*0.8 > avg_x:
        for i in range(w):
            if sum(roi[:,i]) > h*segmenter and starting_flag:
        #         print(f'starting index {i} conditioned on {sum(roi[:,i])} > {rows*0.05}')
                if count>0:
                    last = seg_y_region[-1]
                    seg_y_region.append(i-last*0.05)
                else:
                    seg_y_region.append(i*0.2)
                    count+=1
                starting_flag = False
            elif sum(roi[:,i])< h*segmenter and not starting_flag and i>avg_x*0.8:
        #         print(f'ending index {i} conditioned on {sum(roi[:,i])} > {rows*0.05}')
                seg_y_region.append(i+1)
                starting_flag = True
        if not starting_flag and len(seg_y_region)<=1:
            seg_y_region.append(w)
    else:
        seg_y_region+=[0+w*0.05,w]
            
    new_col_cells = []
    if len(seg_y_region)>1:
        last_end = 0
        flag = False
        for start,end in zip(seg_y_region[0:-1],seg_y_region[1:]):
            if abs(start - end)>avg_x*pc:
                if last_end == 0:
                    new_col_cells.append(cell[0]+start) 
                    last_end = end
                else:
                    start = int((last_end+start)/2)
                    new_col_cells.append(cell[0]+start) 
                    last_end = end
                flag = True
            else:
                flag = False

        if flag:
            new_col_cells.append(cell[0]+end) 
            
            
    if len(new_col_cells)<2:
        new_col_cells= [cell[0],cell[2]] 
        
    else:
        if abs(new_col_cells[0]-tx0)<avg_x*0.7:
            new_col_cells = find_coordinates([cell[0]]+new_col_cells, avg_x, 0.7)
        if abs(new_col_cells[-1]-tx1)<avg_x*0.7:
            new_col_cells = find_coordinates(new_col_cells[0:-1]+[cell[2]], avg_x, 0.7)
        else:
            new_col_cells = find_coordinates(new_col_cells+[cell[2]], avg_x, 0.7)

    return generate_new_cells(sorted(set(new_row_cells)),sorted(set(new_col_cells)),cell[4])



def get_table_coord_initial(cell, tables):
    for idx,table in enumerate(tables):
        table = list(map(int, table))
        score = tsa.how_much_contained(cell[0:4],table)    
        if score > 0.5:
            return idx, score 
    return [], score


def get_table_coord(cell, tables, average_cordinates):
    for idx,table in enumerate(tables):
        table = list(map(int, table))
        if check_cell(cell,average_cordinates):
            score = tsa.how_much_contained(cell[0:4],table)    
            if score > 0.5:
                return idx, score 
    return [], 0
    
def generate_new_cells(y_coords, x_coords,idx):
    possible_cells = []
    x_a = x_coords[0]
    for r, x_b in enumerate(x_coords[1:]):
        if abs(x_b-x_a)>2:
            row_a = [(int(x_a),y) for y in y_coords]
            row_b = [(int(x_b),y) for y in y_coords]
            
            for i in range(len(row_a)-1):
                imagedraw = (row_a[i]+row_b[i+1])
                imagedraw = list(map(int, imagedraw))

                xax = abs(imagedraw[2] - imagedraw[0])
                yax = abs(imagedraw[3] - imagedraw[1])
                possible_cells.append(imagedraw+[idx])
            x_a = int(x_b)
    return possible_cells

# check boundary
def correct_boundary_single_cell(imagedraw, mask, table,average_cordinate, pc):
    cell = imagedraw[0:4]
    x_avg, y_avg = average_cordinate
    mask_area = mask[cell[1]:cell[3],cell[0]:cell[2]]
    
    x0,y0,x1,y1 = list(map(int, table))
    new_cell = list(cell)
    changed = [0]*4
        
    # Fix y-axis boundary
    count = 0
    lower = 0
    xc0,yc0,xc1,yc1 = list(map(int, cell))
    for idx, mask_cell in enumerate(mask_area):  # row by row check ascending order
    #     print(len(mask_cell),len(mask_area))
        if not any(mask_cell): # not text region
            count+=1

        elif lower == 0 and count>0:  # check y0
            if count>5:
                new_cell[1]= new_cell[1]+5
            else:
                new_cell[1]= new_cell[1]+count
            changed[1] = 1
            break

        elif float(sum(mask_cell))/len(mask_cell)< 0.2:
            count+=1
            
        elif count == 0:
            break

    count = 0
    lower = 0
    for idx in reversed(range(len(mask_area))):  # row by row check descending order
        mask_cell = mask_area[idx]
    #     print(len(mask_cell),len(mask_area))
        if not any(mask_cell): # not text region
            count+=1

        elif lower == 0 and count>0:  # check y0
            if count>5:
                new_cell[3]= new_cell[3]-5
            else:
                new_cell[3]= new_cell[3]-count
            changed[3] = 1
            break

        elif float(sum(mask_cell))/len(mask_cell)< 0.2:
            count+=1
            
        elif count == 0:
            break

    # Fix x-axis boundary
    count = 0
    lower = 0

    for col_check in range(len(mask_area[0])): # column by column check x0 to xn
        mask_cell = mask_area[:,col_check]
    #     print(len(mask_cell),len(mask_area[0]))
        if not any(mask_cell): # not text region
            count+=1

        elif lower == 0 and count>0:  # check x0
            if count>5:
                new_cell[0]= new_cell[0]+5
            else:
                new_cell[0]= new_cell[0]+count
            changed[0] = 1
            break

        elif float(sum(mask_cell))/len(mask_cell)< 0.2:
            count+=1
            
        elif count == 0:
            break

    count = 0
    lower = 0

    for col_check in reversed(range(len(mask_area[0]))): # column by column check reverse xn to x0
        mask_cell = mask_area[:,col_check]
        if not any(mask_cell): # not text region
            count+=1

        elif lower == 0 and count>0:  # check x0
            if count>5:
                new_cell[2]= new_cell[2]-5
            else:
                new_cell[2]= new_cell[2]-count
            changed[2] = 1
            break

        elif float(sum(mask_cell))/len(mask_cell)< 0.2:
            count+=1

        elif count == 0:
            break

    # if cell is in table border area
    if abs(x0-cell[0])<x_avg*pc:
        new_cell[0] = x0
        changed[0] = 5
    if abs(x1-cell[2])<x_avg*pc:
        new_cell[2] = x1
        changed[2] = 5
    if abs(y0-cell[1])<y_avg*pc:
        new_cell[1] = y0
        changed[1] = 5
    if abs(y1-cell[3])<y_avg*pc:
        new_cell[3] = y1
        changed[3] = 5

    return new_cell, changed


def sort_coord(new_excluded_cells_pred, tables, average_cordinates):
    x = [i[0:4] for i in new_excluded_cells_pred]
    x = torch.IntTensor(x)
    value, index = x[:,1].sort()

    y_cords = {}
    for idx, i in zip(index,value):
        idx, i = (int(idx),int(i))

        if i not in y_cords:
            y_cords[i]=[]
        y_cords[i].append(idx)
        
    y_cords_new = {}
    for i in y_cords:
        mat = [new_excluded_cells_pred[idx][0:5] for idx in y_cords[i]]
        mat = torch.IntTensor(mat)
        _, index = mat[:,0].sort()

        mat_new = []

        for idx in index:
            cell =  mat[int(idx)].tolist()
            table_idx, score = get_table_coord(cell, tables, average_cordinates)
            if score > 0.5:
                mat_new.append(cell[0:4]+[table_idx])  #cells and its table index
        y_cords_new[i]=mat_new
    return y_cords_new

# def classify_error_cells(new_excluded_cells_pred, tables, mask, average_cordinates,pc):
#     if len(new_excluded_cells_pred) == 0:
#         return [],[]
#     y_cords = sort_coord(new_excluded_cells_pred, tables, average_cordinates)

#     new_corrected_excluded_cells = []
#     not_corrected_excluded_cells = []
#     yidx = list(y_cords.keys())

#     for yid, y_idx in enumerate(y_cords):
#         yneighbours_idx = []
#         if yid>0 and yid+1 < len(y_cords):
#             yneighbours_idx = [yidx[yid-1],yidx[yid+1]]
#         elif yid+1 < len(y_cords):
#             yneighbours_idx = [yidx[yid+1]]

#         xneighbours_idx = []

#         for xid, imagedraw in enumerate(y_cords[y_idx]):

#             if xid>0 and xid+1 < len(y_cords[y_idx]):
#                 xneighbours_idx = [yneighbours_idx, [y_cords[y_idx][xid-1],y_cords[y_idx][xid+1]]]

#             elif xid+1 < len(y_cords[y_idx]):
#                 xneighbours_idx = [yneighbours_idx,[y_cords[y_idx][xid+1]]]
#             else:
#                 xneighbours_idx = [yneighbours_idx,[]]

#             table_idx = imagedraw[-1]

#             table = tables[table_idx]
#             new_cell, changed = correct_boundary_single_cell(imagedraw, mask, table, average_cordinates[table_idx], 1-pc)
            
#             x_avg, y_avg = average_cordinates[table_idx]

#             if abs(new_cell[0]-new_cell[2])<x_avg*pc or abs(new_cell[1]-new_cell[3])<y_avg*pc:
#                 not_corrected_excluded_cells.append(new_cell+[0]+[xneighbours_idx]+[changed])
#             elif sum(changed)==4 and 5 not in changed:
#                 new_corrected_excluded_cells.append(new_cell+[1]+[xneighbours_idx]+[changed])
#             elif sum(changed)> 5 and 0 not in changed:
#                 new_corrected_excluded_cells.append(new_cell+[1]+[xneighbours_idx]+[changed])
#             else:
#                 not_corrected_excluded_cells.append(new_cell+[0]+[xneighbours_idx]+[changed])
    
#     return new_corrected_excluded_cells, not_corrected_excluded_cells
def classify_error_cells(new_excluded_cells_pred, tables, mask, average_cordinates,pc):
    new_corrected_excluded_cells = []
    not_corrected_excluded_cells = []

    for cell in new_excluded_cells_pred:
        tid = cell[4]
        changed = [0]*4
        
        if check_cell(cell,average_cordinates):
            new_cell, changed = correct_boundary_single_cell(cell, mask, tables[tid], average_cordinates[-1], 1-pc)

            x_avg, y_avg = average_cordinates[-1]

            if abs(new_cell[0]-new_cell[2])<x_avg*pc or abs(new_cell[1]-new_cell[3])<y_avg*pc:
                not_corrected_excluded_cells.append(new_cell+[tid]+[changed])
            elif sum(changed)==4 and 5 not in changed:
                new_corrected_excluded_cells.append(new_cell+[tid]+[changed])
            elif sum(changed)> 5 and 0 not in changed:
                new_corrected_excluded_cells.append(new_cell+[tid]+[changed])
            else:
                not_corrected_excluded_cells.append(new_cell+[tid]+[changed])

        else:
            not_corrected_excluded_cells.append(cell+[changed])
    
    return new_corrected_excluded_cells, not_corrected_excluded_cells

# Correct cell boundary considering multiple cells
def get_neighbouring_coord(cells, tables, average_cordinates):
    new_excluded_cells_pred = []
    
    for cell in cells:
        new_excluded_cells_pred.append(cell[0:5])
    
    x = torch.IntTensor(new_excluded_cells_pred)
    value, index = x[:,1].sort()

    y_cords = {}
    for idx, i in zip(index,value):
        idx, i = (int(idx),int(i))

        if i not in y_cords:
            y_cords[i]=[]
        y_cords[i].append(idx)
        
    y_cords_new = {}
    for i in y_cords:
        mat = [new_excluded_cells_pred[idx] for idx in y_cords[i]]
        mat = torch.IntTensor(mat)
        _, index = mat[:,0].sort()

        mat_new = []

        for idx in index:
            cell =  mat[int(idx)].tolist()
            mat_new.append(cell)  #cells and its table index
        if len(mat_new)>0:
            y_cords_new[i]=mat_new
    
    new_excluded_cells_pred = []
    
    for cell in cells:
        new_excluded_cells_pred.append(cell[0:5])
    
    x = torch.IntTensor(new_excluded_cells_pred)
    value, index = x[:,0].sort()
    
    x_cords = {}
    for idx, i in zip(index,value):
        idx, i = (int(idx),int(i))

        if i not in x_cords:
            x_cords[i]=[]
        x_cords[i].append(idx)

    x_cords_new = {}
    for i in x_cords:
        mat = [new_excluded_cells_pred[idx] for idx in x_cords[i]]
        mat = torch.IntTensor(mat)
        _, index = mat[:,0].sort()

        mat_new = []

        for idx in index:
            cell =  mat[int(idx)].tolist()
            mat_new.append(cell)  #cells and its table index
        if len(mat_new)>0:
            x_cords_new[i]=mat_new

    return y_cords_new, x_cords_new


# sort x and y coordinates of excluded_cells_pred
def get_x_y_neighbour_cells(new_excluded_cells_pred, tables, average_cordinates):
    y_cords, x_cords = get_neighbouring_coord(new_excluded_cells_pred, tables, average_cordinates)
    neighbouring_cells = {}
    yidx = list(y_cords.keys())

    for yid, y_idx in enumerate(y_cords):
        yneighbours_idx = []
        for cell in y_cords[y_idx]:
            tid = cell[4]
            avg_x, avg_y = average_cordinates[tid]
            mat = [idx for idx in x_cords[cell[0]]]
            mat = torch.IntTensor(mat)
            value, index = mat[:,1].sort()
            for idx, val in zip(index,value):
                if int(val) not in yneighbours_idx:
                    yneighbours_idx.append(int(val))

        ### remove if condition if filtering is not required

        xneighbours_idx = []
        if len(y_cords[y_idx])>=1:
            xneighbours_idx = [yneighbours_idx, y_cords[y_idx]]
        else:
            xneighbours_idx = [yneighbours_idx,[]]  # no neighbours

        neighbouring_cells[y_idx]=xneighbours_idx
    
    return neighbouring_cells, y_cords, x_cords
            
def correct_boundary_neighbour(cell, tables, mask, average_cordinates, step_x, step_y):    
    anchor = list(cell)
    notfound = True
    count = 0 
    step_x0 = 0 
    step_x1 = 0
    step_y0 = 0
    step_y1 = 0
    changed = [1,1,1,1]
    
    while notfound:
        
        if anchor[0]<anchor[2] and anchor[1]< anchor[3]:
            new_corrected, not_corrected = classify_error_cells([anchor], tables, mask, average_cordinates,0.7) # correct single error cell or declare error
        else:
            return []

        if len(new_corrected)>0:
            count+=1
            
            anchor[0]+=step_x0
            anchor[1]+=step_y0
            anchor[2]+=step_x1
            anchor[3]+=step_y1
            
            if count>0:
                notfound = False
        elif len(not_corrected)>0:
            
            changed = not_corrected[0][6]   
            if 0 not in changed: 
                notfound = False
            else:
                if changed[0]==0:
                    step_x0+=step_x
                    anchor[0]+=step_x0
                if changed[1]==0:
                    step_y0+=step_y
                    anchor[1]+=step_y0
                if changed[2]==0:
                    step_x1+=step_x
                    anchor[2]+=step_x1
                if changed[3]==0:
                    step_y1+=step_y
                    anchor[3]+=step_y1           
    
    cur_corrected = list(anchor)
    return cur_corrected

def find_proper_cell(cell, avg_x, avg_y):
    if abs(cell[0]-cell[2])>(avg_x*2)/3 and abs(cell[1]-cell[3])>(avg_y*2)/3 :
        return True
    else:
        return False
    
def correct_boundary_multiple_cell(cell, n_cell, above, average_cordinates, tables, mask):
    
    tid = cell[4]
    avg_x, avg_y = average_cordinates[tid]
    tab_boundary = list(map(int, tables[tid]))
    
    step_x = int(avg_x*0.025)
    step_y = int(avg_y*0.025)
    
    if find_proper_cell(cell, avg_x, avg_y):
        cur_corrected = correct_boundary_neighbour(cell, tables, mask, average_cordinates, step_x, step_y)
        if len(cur_corrected) == 0:
            cur_corrected = cell
    else:
        cur_corrected = cell

    anchor = list(n_cell) ## neighbour update check
    if len(cur_corrected)>1:
        if above: # check cell by crossing boundary of above cell i.e. decrease y0 and y1
            anchor[3] = cur_corrected[1]
            anchor[0] = cur_corrected[0]

        else: # increase y0 and y1 by avg_y*0.05 (5%) cell size
            anchor[1] = cur_corrected[3]
            anchor[0] = cur_corrected[0]

    if find_proper_cell(anchor, avg_x, avg_y):
        n_cur_corrected = correct_boundary_neighbour(anchor, tables, mask, average_cordinates, step_x, step_y)
        if len(n_cur_corrected) == 0:
            n_cur_corrected = anchor
    else:
        n_cur_corrected = anchor
    
    return cur_corrected, n_cur_corrected


def display_cells(cells, image):
    for cur_cell2 in cells:
        croppedimage_full=image[int(cur_cell2[1]):int(cur_cell2[3]),int(cur_cell2[0]):int(cur_cell2[2])] 
        plt.rcParams["figure.figsize"] = (10,5)
        plt.imshow(croppedimage_full)
        plt.title(f'Cell coords {cur_cell2}')
        plt.show()  