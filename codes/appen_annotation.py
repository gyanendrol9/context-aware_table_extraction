import cv2
import cv2 as cv
import imutils
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append('../src')

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2 as cv
import collections

import dla.src.table_structure_analysis as tsa
import dla.src.xml_utils as xml_utils
from dla.src.image_utils import put_box, put_line

import importlib
import glosat_utils

from glosat_utils import *


config_file = '/data/glosat/glosat_table_dataset/dla/config/cascadeRCNN.py'
table_checkpoint_file = '/data/glosat/glosat_table_dataset/models/model_tables_enchanced_GloSAT.pth'

model = init_detector(config_file, table_checkpoint_file, device='cuda:0')

CLASSES = ("table_body","cell","full_table","header","heading")
color = 255

directory = '/data/glosat/Appen/Annotation/to_annotate'
outdirectory = '/data/glosat/Appen/Annotation/to_annotate_table_boundary_with_header'

files = os.listdir(directory)
for filename in files:
    filename = filename.replace('.jpg', '')
    if not os.path.exists(outdirectory):
        os.mkdir(outdirectory)
    if not os.path.exists(outdirectory+'/XML-VOC'):
        os.mkdir(outdirectory+'/XML-VOC')
    if not os.path.exists(outdirectory+'/images'):
        os.mkdir(outdirectory+'/images')


    img_path = f'{directory}/{filename}.jpg'

    img = cv.imread(img_path)
    image, height, width, _ = image_preprocessing(img)

    # full_tables, tables = detect_table_region(model,img)  
    result = inference_detector(model, img)  

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

    #Process tables
    full_tables = []
    for box in result[CLASSES.index("full_table")]:
        if box[4]>THRESHOLD :
            full_tables.append(box[0:4])

            if all(tsa.how_much_contained(table,box)<0.5 for table in tables):
                tables.append(box[0:4])

    for table in tables:
        if all(tsa.how_much_contained(table,full_table)<0.5 for full_table in full_tables):
            full_tables.append(table)

    for count,box in enumerate(headers):
        box = list(map(int, box[0:4]))
        box = box[0:4]+np.asarray([-10,-10,10,10])
        img = cv.imread(img_path)
        put_box(img,box,(color,0,0)) # Red

        im_pil = Image.fromarray(img)
        im_pil.save(f"{outdirectory}/images/{filename}-header-{count}.jpg")
    #     xml_utils.save_ICDAR_xml( full_tables, [], [], f"{outdirectory}/XML/{filename}-{count}.xml" )
        xml_utils.save_VOC_xml_from_cells([],[],[box],[box],[],f"{outdirectory}/XML-VOC/{filename}-header-{count}.xml",width,height)

    for count,box in enumerate(tables):
        box = list(map(int, box[0:4]))
        box = box[0:4]+np.asarray([-10,-10,10,10])
        img = cv.imread(img_path)
        put_box(img,box,(color,0,0)) # Red

        im_pil = Image.fromarray(img)
        im_pil.save(f"{outdirectory}/images/{filename}-table-{count}.jpg")
    #     xml_utils.save_ICDAR_xml( full_tables, [], [], f"{outdirectory}/XML/{filename}-{count}.xml" )
        xml_utils.save_VOC_xml_from_cells([],[],[box],[box],[],f"{outdirectory}/XML-VOC/{filename}-table-{count}.xml",width,height)

    print(filename)
    


