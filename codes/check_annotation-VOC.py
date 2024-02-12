import cv2
import cv2 as cv
import imutils
import os
from os import path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
# Add code to sys.path
import matplotlib.pyplot as plt 

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

import bs4 as bs
from PIL import Image

directory = '/data/glosat/active_learning/dla_models/model_table_struct_fine_aclr_2/VOC2007'
outdirectory = '/data/glosat/active_learning/dla_models/model_table_struct_fine_aclr_2/VOC2007/check_annotation'
annotation_type = 'Annotations'

f = open('/data/glosat/active_learning/dla_models/model_table_struct_fine_aclr_2/VOC2007/ImageSets/main.txt','r')
files = f.readlines()
f.close()

if not os.path.exists(f"{outdirectory}"):
    os.mkdir(f"{outdirectory}")

for filename in files:
    filename = filename.strip()
    img_path = f'{directory}/JPEGImages/{filename}.jpg'

    img_xmlpath = f'{directory}/{annotation_type}/{filename}.xml'
    voc_parsed = xml_utils.load_VOC_xml( img_xmlpath )

    headings = []
    headers = []
    tables = []
    full_tables = []
    cells = []

    for entry in voc_parsed :
        if entry['name'] == 'header' :
            headers.append( entry['bbox'] )
        elif entry['name'] == 'table_body' :
            tables.append( entry['bbox'] )
        elif entry['name'] == 'heading' :
            headings.append( entry['bbox'] )
        elif entry['name'] == 'full_table' :
            full_tables.append( entry['bbox'] )
        elif entry['name'] == 'cell' :
            cells.append( entry['bbox'] )


    img = cv.imread(img_path)
    image, height, width, _ = image_preprocessing(img)

    color = 255
    for box in tables:
        box = list(map(int, box[0:4]))
        put_box(image,box,(0,color,color)) # Cyan 
        
    for box in list(cells):
        put_box(image,box,(0,0,color)) # Blue

    im_pil = Image.fromarray(image)
    im_pil.save(f"{outdirectory}/{filename}.jpg")

