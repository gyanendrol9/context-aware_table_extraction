import os
import json
import dla.src.table_structure_analysis as tsa
from dla.src.image_utils import put_box, put_line

import torch
from statistics import mean
import cv2 as cv
import pickle as pkl
from os import path

from PIL import Image, ImageDraw, ImageFont

workdir = 'Tabular-Data-Extraction'

import sys
sys.path.append(f'{workdir}/docExtractor-master/src')

from utils.image import resize
from utils.constant import LABEL_TO_COLOR_MAPPING
from utils.image import LabeledArray2Image

from models import load_model_from_path
from utils import coerce_to_path_and_check_exist
from utils.path import MODELS_PATH
from utils.constant import MODEL_FILE
import numpy as np

image_source = sys.argv[1]
out_dir = sys.argv[2]

font_size = 20
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", size=font_size)

dataset = 'GloSAT'
reconstruction_op = 'reconstruction_op'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

outpathTSV = f'{out_dir}/{reconstruction_op}'
if not os.path.exists(outpathTSV):
    os.mkdir(outpathTSV)

outpathTSV = f'{out_dir}/{reconstruction_op}/TSV'
if not os.path.exists(outpathTSV):
    os.mkdir(outpathTSV)

outpathIMG = f'{out_dir}/{reconstruction_op}/Images'
if not os.path.exists(outpathIMG):
    os.mkdir(outpathIMG)

op_path = f'{out_dir}/JSON-OCR'
table_dir = f'{out_dir}/JSON-det'

appen_table_dir = 'data/Table'
maskpath = 'docExtractor_output/originalImage'

color_flag = {}
color_flag[0] = (0, 0, 0)       # Black correct
color_flag[1] =  (255, 5, 0)     # Red Not easy to transcribe 
color_flag[2] = (255, 0, 255)   #Magenta Partial text
color_flag[3] =  (128, 128, 0) #Yellow Blank


if not os.path.exists(maskpath):
    os.mkdir(maskpath)

# from deeppavlov import build_model, configs
import pandas as pd
from rouge_score import rouge_scorer
import jiwer
from autocorrect import Speller

import re
from nltk.corpus import wordnet


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

def find_closest_idx(xi,L):
    diffs = [abs(Li - xi) for Li in L]
    return diffs.index(sorted(diffs)[0])

from textblob import TextBlob

# Function to correct the OCR text using TextBlob
def correct_spelling(text):
    blob = TextBlob(text)
    
    # Correct the spelling in the OCR text
    corrected_text = str(blob.correct())
    
    return corrected_text

def remove_dot_strings(text):
    # This regular expression matches strings that consist entirely of dots
    clean_text = re.sub(r'\.{1,}', '.', text) 
    
    # Removing extra spaces that might appear after removing dots
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def remove_consecutive_duplicates(text):
    # Use regex to find repeated words and replace with a single occurrence
    corrected_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    return corrected_text

def remove_repeated_substrings(text):
    # This regex detects repeated substrings like "townstownstownstown..."
    pattern = re.compile(r'(\b\w+)(\1\b)+')
    corrected_text = pattern.sub(r'\1', text)
    return corrected_text


class RepeatReplacer:
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

def evaluate_TDE(pd_cells, gt_cells):
    
    not_overlapped_gt = []
    for cell in gt_cells:
        gcell, text = gt_cells[cell]
        matched = 0
        for pcell in pd_cells:
            pdcell, ptext = pd_cells[pcell]
            if is_overlapped([gcell[:4]], pdcell[:4], pc=0.5):
                matched+=1
                # if matched>1:
                #     print(gcell, pdcell, text, ptext, matched)
        if matched == 0:
            not_overlapped_gt.append(cell)
            
    recall= (len(gt_cells)-len(not_overlapped_gt))/len(gt_cells)

    
    gt_text_all=''
    pd_text_all=''
    spell_correction = ''

    # Calculate ROUGE scores for each pair of texts
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # Initialize lists to hold WER and CER values
    wer_scores = []
    cer_scores = []

    ground_truths = []
    output_pred = []
    not_overlapped_pd = []
    for cell in pd_cells:
        pdcell, ptext = pd_cells[cell]
        
        idx = f'{pdcell[0]}_{pdcell[1]}_{pdcell[2]}_{pdcell[3]}'         
        if idx in blank_cell_idx:
            ptext = ''

        matched = 0
        for pcell in gt_cells:
            gcell, text = gt_cells[pcell]
            
            if is_overlapped([gcell[:4]], pdcell[:4], pc=0.5):
                ground_truths.append(text)
                output_pred.append(ptext)
                
                # print('Overlapped', text, text_info[0])
                text = text.replace("\n", "\\n")
                gt_text_all+=f'{text}\t'
                pd_text_all+=f'{ptext}\t'
                spell_correction+=f'{text}\t{ptext}\n\n'
                
                scores = scorer.score(text.lower(), ptext.lower())
                rouge_1_scores.append(scores['rouge1'].fmeasure)
                rouge_2_scores.append(scores['rouge2'].fmeasure)
                rouge_l_scores.append(scores['rougeL'].fmeasure)
                
                
                # Calculate CER
                text = text.replace(' ','').replace('\t','').replace('·','.').lower()
                ptext = ptext.replace(' ','').replace('\t','').replace('·','.').lower()
                text = re.sub(r'\.{2,}', '.', text)
                ptext = re.sub(r'\.{2,}', '.', ptext)
                cer = jiwer.cer(text, ptext)
                cer_scores.append(cer)
                # print(f'CER score({text},{ptext}): {cer}')
                
                if cer>0.3:
                    # Calculate WER
                    wer = jiwer.wer(text.lower(), ptext.lower())
                else:
                    wer = 0
                
                wer_scores.append(wer)
            
                matched+=1
                # if matched>1:
                # print(gcell, pdcell, text, ptext, matched)
                gt_text_all+=f'{text}, '
                pd_text_all+=f'{ptext}, '
                
        if matched == 0:
            not_overlapped_pd.append(cell)
            # gt_text_all+=f'{text}\t'
            # pd_text_all+=f'\t'
            
    precicion= (len(pd_cells)-len(not_overlapped_pd))/len(pd_cells)

    # Create a DataFrame to display the results
    results_df = pd.DataFrame({
        'Ground Truth': ground_truths,
        'Prediction': output_pred,
        'ROUGE-1': rouge_1_scores,
        'ROUGE-2': rouge_2_scores,
        'ROUGE-L': rouge_l_scores,
        'WER': wer_scores,
        'CER': cer_scores
    })

    # Calculate average scores
    average_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
    average_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    average_wer_scores = sum(wer_scores) / len(wer_scores)
    average_cer_scores = sum(cer_scores) / len(cer_scores)
    f1score = 2*precicion*recall/(precicion+recall)
    
    return precicion, recall, f1score, average_rouge_1, average_rouge_2, average_rouge_l, average_wer_scores, average_cer_scores, gt_text_all, pd_text_all 

def contains_digit(s):
    return any(char.isdigit() for char in s)

def contains_majority_digits(s):
    digit_count = sum(char.isdigit() for char in s)
    non_digit_count = len(s) - digit_count  # Total length minus digit count gives non-digit count

    # Return True if there are more digits, otherwise False
    return digit_count > non_digit_count

def sort_tables(tablez):
    ty_cord = []
    for table_cord in tablez:
        ty_cord.append(table_cord[1])

    sty_cord = sorted(ty_cord)

    indexes = []
    for table_cord in tablez:
        diff = [abs(table_cord[1]-y) for y in sty_cord]
        index_mapping = diff.index(0.0)
        indexes.append(index_mapping)

    sorted_table = [[] for i in tablez]
    for idx, i in enumerate(indexes):
        sorted_table[i] = tablez[idx] 

    return sorted_table

def sort_col_cells(col_cells):
    tx_cord = []
    for table_cord in col_cells:
        tx_cord.append(table_cord[0])
    sty_cord = sort_coordinates(tx_cord)

    diffs = [x-sty_cord[0] for x in tx_cord]
    index_mapping = [sty_cord.index(int(x)) for x in tx_cord]
    sorted_col_cells = [[] for i in col_cells]
    for i,idx in enumerate(index_mapping):
        sorted_col_cells[idx] = col_cells[i]
        
    return sorted_col_cells


def sort_by_rows_cells(tables, y_avg, pc=0.5):
    ty_cord = []
    for table_cord in tables:
        ty_cord.append(table_cord[1])
    sty_cord = sort_coordinates(ty_cord)
    
    diffs = [x-sty_cord[0] for x in ty_cord]
    
    index_mapping = []
    for diff in diffs:
        if diff>y_avg*0.3:
            index_mapping.append(1)
        else:
            index_mapping.append(0)

    sorted_table = [[] for idx in tables]
    for idx, i in enumerate(index_mapping):
        sorted_table[i].append(tables[idx])
    for i, cells in enumerate(sorted_table):
        sorted_table[i] = sort_col_cells(sorted_table[i])
    
    return sorted_table, sty_cord, index_mapping


def find_average_cellsize(copy_cells):
    # find average cell size
    if len(copy_cells)>0:
        x_size = []
        y_size = []
        for cell in copy_cells:
            xax = abs(cell[0]-cell[2])
            yax = abs(cell[1]-cell[3])
            x_size.append(xax)
            y_size.append(yax)
        x_avg = mean(x_size)
        y_avg = mean(y_size)
        return (x_avg, y_avg)
    else:
        return (0, 0)        

def get_table_coord(cell, tables):
    for idx,table in enumerate(tables):
        table = list(map(int, table))
        score = tsa.how_much_contained(cell[0:4],table)    
        if score > 0.5:
            return idx, score 
    return [], 0

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
            
#             print(new_x_coords, small_coords, '\n',new_x_coords[-1], i, pc2, cell_size_min, '\n', i, j, pc, cell_size_min, '\n')
            
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

def thresholding(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def image_preprocessing(img):
#     img = cv.resize(img, (800,1200), interpolation = cv.INTER_AREA)
    #Binarization
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Do dilation and erosion to eliminate unwanted noises
    kernel = np.ones((1,1), np.uint8)
    img = cv.dilate(img, kernel, iterations=30)
    img = cv.erode(img, kernel, iterations=30)

    #thresholding
    img = thresholding(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    w,h,c = img.shape
    
    return img,h,w,c

# Load and preprocess the image
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def preprocess_image(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values

import textwrap
def plot_text_image(draw, annotated_image,bbox,digitized_text):
    # Create a new image with the same height as the original image and double the width
    x,y,x_max,y_max = bbox[:4]
    y0 = y
    
    digitized_text = digitized_text.encode('utf-8')
    
    text = digitized_text.decode('utf-8')

    # Get the bounding box for the text, which returns a tuple (left, top, right, bottom)
    tbox = draw.textbbox((0, 0), text, font=font)

    # Calculate the width and height
    text_width = tbox[2] - tbox[0]
    text_height = tbox[3] - tbox[1]

    # Store the text size as (width, height)
    text_size = (text_width, text_height)
    
    # Calculate the position for the bottom-right corner of the rectangle
    rectangle_size = (x + text_size[0], y + text_size[1])

    # Calculate the new x-coordinate for the digitized text
    new_x = annotated_image.width // 2 + x
        
    box_x1, box_y1 = x, y - text_height
    box_x2, box_y2 = x + text_width, y

    special_flag = 0
    if '@@@' in text:
        special_flag = 1
    elif '$$$' in text:
        special_flag = 2
    elif '###' in text:
        special_flag = 3

        
    if box_x2 > im_width:
        # wrapped_text = textwrap.wrap(text, width=int(width*0.2))
        try:
            wrapped_text = textwrap.wrap(text, width=int(0.15*(x_max-x)))
        except:
            wrapped_text = text
    #     print(new_x)

        # Draw the red rectangle behind the text on the new image
        # draw.rectangle([(new_x, y), (new_x + text_size[0], y + text_size[1])])


        for line in wrapped_text:
            # Draw the line of text
            draw.text((new_x, y), line, font=font, fill="blue", align="left", spacing=10, multiline=True)

            # Update the y-coordinate for the next line
            y += font.getsize(line)[1]

        # text_box = (x, y, x + max_width, y + max_height) x,y,x_max,y_max
        if special_flag>0:
            draw.rectangle([(new_x, y0), (new_x + text_size[0], y + text_size[1])], outline=color_flag[special_flag],  width=6)
        else:
            draw.rectangle([(new_x, y0), (new_x + text_size[0], y + text_size[1])], outline=color_flag[special_flag])
        # draw.rectangle(text_box, outline="red")
        
    else:

        # Calculate the position for the bottom-right corner of the rectangle
        rectangle_size = (x + text_size[0], y + text_size[1])

        # Calculate the new x-coordinate for the digitized text
        # new_x = annotated_image.width // 2 + x

        # Draw the red rectangle behind the text on the new image
        if special_flag:
            draw.rectangle([(new_x, y), (new_x + text_size[0], y_max + text_size[1])], outline=color_flag[special_flag],  width=6)
        else:
            draw.rectangle([(new_x, y), (new_x + text_size[0], y_max + text_size[1])], outline=color_flag[special_flag])

        # Draw the text on top of the red rectangle at the new position on the new image
        draw.text((new_x, y), text, font=font, fill="green", align="left", spacing=10, multiline=True)

    return 0

from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the date and time as a string
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

device = torch.device('cpu')

TAG = 'default'
model_path = coerce_to_path_and_check_exist(MODELS_PATH / TAG / MODEL_FILE)
model_extractor, (img_size, restricted_labels, normalize) = load_model_from_path(model_path, device=device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
_ = model_extractor.eval()


color = 255
restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in restricted_labels]
label_idx_color_mapping = {restricted_labels.index(l) + 1: c for l, c in zip(restricted_labels, restricted_colors)}

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
spell = Speller() 

files = os.listdir(op_path)
cfiles = len(files)
print(cfiles)

for c, file in enumerate(files):
    try:
        print(f'Processing {file}.....{c}/{cfiles}...')
        filename = file.split('.')[0]
        img_dir = f'{image_source}/{filename}.jpg'
            
        ann_jsonl = f"{op_path}/{filename}.json"
        print(f'load OCR output file: {ann_jsonl}')
        with open(ann_jsonl, 'r') as json_file:
            data = json.load(json_file)
            tablez = data['tables']
            cellz = data['cell_texts']

        tables = sort_tables(tablez)
        #Load text area regions
        print(f'Load text area regions for {file}')
        if not os.path.exists(f'{maskpath}/{filename}_mask.pkl'):
            print(f'Calculate text area regions of {img_dir}')
            image = cv.imread(img_dir)
            masks, blend_img = find_text_region(image, label_idx_color_mapping, normalize)
            blend_img.save(f'{maskpath}/{filename}_text_region.jpg')

            f = open(f'{maskpath}/{filename}_mask.pkl','wb')
            pkl.dump((masks, blend_img),f)
            f.close()
        else:
            f = open(f'{maskpath}/{filename}_mask.pkl','rb')
            (masks, blend_img) = pkl.load(f)
            f.close()
            
        #sort the coordinates 
        y_centroids = [[] for i in tables]
        table_cells = dict()
        for box,text_info in cellz:
            yi = (box[3]+box[1])/2
            tid, score = get_table_coord(box, tables)
            if score>0.5:
                if len(box)>=5:
                    box[4] = tid #Error here
                else:
                    box.append(tid)
                y_centroids[tid].append(yi)
                if tid not in table_cells:
                    table_cells[tid] = []
                table_cells[tid].append(box)
            else:
                print(f'Error in finding table index for {file}!')
                
        average_cordinates = dict()
        for tid in table_cells:
            average_cordinates[tid] = find_average_cellsize(table_cells[tid])
            
        cells = [list(map(int, cell[:5])) for cell,text_info in cellz]
        correct_cells, blank_cells, exclude_cells = classify_cells(cells, masks[1], average_cordinates,pc = 0.9) 

        blank_cell_idx = {}
        for cell in blank_cells:
            cell = list(map(int, cell))
            idx = f'{cell[0]}_{cell[1]}_{cell[2]}_{cell[3]}' 
            blank_cell_idx[idx] = cell

        sort_y_centroids = {}
        for tid in range(len(tables)):
            sort_y_centroids[tid] = sort_coordinates(y_centroids[tid])
            pc = 0.8 
            cell_size_min = average_cordinates[tid][1]*pc
            sort_y_centroids[tid] = remove_small_coordinates(sort_y_centroids[tid], cell_size_min)
            
        ycentroids_cells = dict()
        for tid in table_cells:   
            # print(f"Table {tid} in processing....") 
            ycords = sort_y_centroids[tid]
            
            if tid not in ycentroids_cells:
                ycentroids_cells[tid] = [[] for i in ycords]

            x_avg, y_avg = average_cordinates[tid]
            count = 0
            
            for box in table_cells[tid]:
                if box[4] != tid:
                    print("ERRROR HERE!")
                    
                yi = (box[3]+box[1])/2
                pc = 0.8
                cell_size_min = y_avg*pc
                
                found = 0
                for yc, ycord in enumerate(ycords):
                    ydist = abs(ycord-yi)
                    
                    if ydist<cell_size_min:  
                        found = 1
                        if len(ycentroids_cells[tid][yc])==0 or not is_overlapped(ycentroids_cells[tid][yc], box, pc=0.5):
                            ycentroids_cells[tid][yc].append(box)
                            count+=1
                            break
                        
                if found==0:
                    print(f'{box} Not located')
                    
            tot_cells = len(table_cells[tid])
            # print(f"Table {tid} has {tot_cells} cells. After filtering overlapped cells, it has {count} cells")

        sort_x_cols = {}
        for tid in sort_y_centroids:
            ycords = sort_y_centroids[tid]
            x_cords = []
            if tid in ycentroids_cells:
                numrows = len(ycords)
                for id in range(numrows):
                    num_cells_row = len(ycentroids_cells[tid][id])
                    for cell in ycentroids_cells[tid][id]:
                        x_cords.append(cell[0])
                        if cell[4] != tid:
                            print("ERRROR HERE!")
                    
            sort_x = sort_coordinates(x_cords)
            pc = 0.3 
            cell_size_min = average_cordinates[tid][0]*pc
            sort_x_coords = remove_small_coordinates(sort_x, cell_size_min)
            sort_x_cols[tid] = sort_x_coords
            numcols = len(sort_x_coords)
            print(f'Table {tid} has #{numrows} rows and #{numcols} columns') 
            

        count = 0
        check_overlapped = dict()
        text_cell_idx = {}
        skipped_cells = []
        total_data_new = {}
        color = 255    

        for cell, text_info in cellz:
            tid = cell[4]
            xi = cell[0]
            yi = (cell[3]+cell[1])/2
            
            cell = list(map(int, cell))
            idx = f'{cell[0]}_{cell[1]}_{cell[2]}_{cell[3]}' 
            
            ctext = text_info[0][0].replace('WSW','').replace('WNW','').replace('###','')
            ctext = remove_consecutive_duplicates(ctext)
            ctext = remove_dot_strings(ctext)
            if ctext == '.':
                ctext = ''
            
            text_info[0][0] = ctext
            text_cell_idx[idx] = text_info
            
            xpc, ypc = (0.3, 0.8)
            x_avg, y_avg = average_cordinates[tid]
            
            numrows = len(sort_y_centroids[tid])
            numcols = len(sort_x_cols[tid])
                
            if tid not in total_data_new:
                total_data_new[tid] = {}
                
            if tid not in check_overlapped:
                check_overlapped[tid] = []
                
            x_idx = find_closest_idx(xi,sort_x_cols[tid])
            y_idx = find_closest_idx(yi,sort_y_centroids[tid])
            
            t_data_idx = sort_y_centroids[tid][y_idx]
            if t_data_idx not in total_data_new[tid]:
                total_data_new[tid][t_data_idx]=[]
                
            if len(total_data_new[tid][t_data_idx])==0:
                total_data_new[tid][t_data_idx].append(cell)
            
            elif not is_overlapped(total_data_new[tid][t_data_idx], cell, pc=0.3):
                total_data_new[tid][t_data_idx].append(cell)
            else:
                skipped_cells.append(cell)


        image = Image.open(img_dir)
        annotated_image = Image.new("RGB", (image.width * 2, image.height), color="white")

        im_width, im_height = image.size
        annotated_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(annotated_image)

        row_by_row_cells = {}
        for tid in total_data_new:
            sorted_rowss = sorted(total_data_new[tid].keys())
            cols = sort_x_cols[tid]
            rows = []
            
            if tid not in row_by_row_cells:
                row_by_row_cells[tid] = []
                
            for sidx in sorted_rowss:
                x_avg, y_avg = average_cordinates[tid]
                sorted_row_cells, _, _ = sort_by_rows_cells(total_data_new[tid][sidx], y_avg)
                
                for scells in sorted_row_cells:
                    rows = []
                    for cell in scells: 
                        if len(cell)>3:
                            idx = f'{cell[0]}_{cell[1]}_{cell[2]}_{cell[3]}' 
                            diff_x = abs(cell[0]-cell[2])
                            diff_y = abs(cell[1]-cell[3])
                            if idx not in blank_cell_idx and diff_x > x_avg*0.3 and diff_y >y_avg*0.3:
                                rows.append(cell)
                                
                                text_info = text_cell_idx[idx]
                                ctext = text_info[0][0] #.replace('WSW','').replace('WNW','')
                                # ctext = remove_consecutive_duplicates(ctext)
                                # ctext = remove_dot_strings(ctext)
                                x,y,x_max,y_max = cell[:4]
                                draw.rectangle([(x, y), (x_max, y_max)], outline='blue',  width=6)
                                plot_text_image(draw, annotated_image, cell, ctext)


                    if len(scells)>0:
                        row_by_row_cells[tid].append(rows)

        annotated_image.save(f'{outpathIMG}/{filename}.jpg')

        # Get the current date and time
        now = datetime.now()

        # Format the date and time as a string
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        config_ocr = f'''source-file\t{filename}
source-folder\t{dataset}
process-timestamp\t{timestamp}
process-engineer\t<Gyanendro Loitongbam>\n'''


        for tid in range(len(row_by_row_cells)):
            print(f'\n---------------Table {tid}--------------')
            cols = sort_x_cols[tid]
            x_avg, y_avg = average_cordinates[tid]
            
            numrows = len(sort_y_centroids[tid])
            numcols = len(sort_x_cols[tid])
            
            config_ocr_new = config_ocr
            config_ocr_new+=f'''data-number-of-tables-per-page\t{len(row_by_row_cells)}
data-table-number\t{tid}
data-number-of-rows\t{numrows}
data-number-of-columns\t{numcols}\n'''
            
            f = open(f'{outpathTSV}/{filename}_TID_{tid}-OCR_op.tsv', 'w')
            f.write(config_ocr_new+'\n') 

            for cells in row_by_row_cells[tid]:
                text = ''
                print_bool = [0 for col_cord in cols]
                for p, col_cord in enumerate(cols):
                    for cell in cells:    
                        diff = abs(cell[0]-col_cord)
                        if diff<x_avg*0.3:
                            print_bool[p] = cell

                for cell in print_bool:
                    if cell==0:
                        text+='\t'
                    else:
                        idx = f'{cell[0]}_{cell[1]}_{cell[2]}_{cell[3]}' 
                        diff_x = abs(cell[0]-cell[2])
                        diff_y = abs(cell[1]-cell[3])
                        
                        # if diff<x_avg*0.3:
                        # print('difference: -- ', diff_x,x_avg*0.3,diff_y,y_avg*0.3)
                        if idx not in blank_cell_idx and diff_x > x_avg*0.3 and diff_y >y_avg*0.3:
                            text_info = text_cell_idx[idx]
                            ctext = text_info[0][0] #.replace('WSW','').replace('WNW','')
                            # ctext = remove_consecutive_duplicates(ctext)
                            text+=ctext.replace('\n','\\n')+'\t'

                        else:
                            text+='\t'
                print(text)
                f.write(text+'\n') 
            f.close()
            
    except Exception as e:
        print(f'Error {e} in {file}')