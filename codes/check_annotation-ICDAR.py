import cv2
import cv2 as cv
import imutils
import os
from os import path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

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

directory = '/data/glosat/active_learning/dla_models/model_table_struct_fine_aclr_0/VOC2007'
outdirectory = '/data/glosat/active_learning/dla_models/model_table_struct_fine_aclr_0/VOC2007'

if not os.path.exists(f"{outdirectory}/check_annotation"):
    os.mkdir(f"{outdirectory}/check_annotation")

file = sys.argv[1]
filename = file.replace('.jpg','')

img_path = f'{directory}/JPEGImages/{filename}.jpg'

img_xmlpath = f'{directory}/Annotations/{filename}.xml'
icdar_parsed = xml_utils.load_VOC_xml( img_xmlpath )

tables  = []
cells = []

for entry in icdar_parsed:
    tables.append(entry["region"])
    cells += entry["cells"]


img = cv.imread(img_path)
image, height, width, _ = image_preprocessing(img)

color = 255
for box in tables:
    box = list(map(int, box[0:4]))
    put_box(image,box,(0,color,color)) # Cyan 
    
for box in list(cells):
    put_box(image,box,(0,0,color)) # Blue

im_pil = Image.fromarray(image)
im_pil.save(f"{outdirectory}/check_annotation/{filename}.jpg")

