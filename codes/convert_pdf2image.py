from pdf2image import convert_from_path
import os

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

import importlib
import glosat_utils

from glosat_utils import *
import xml_utils as xml_utils

config_file = '/data/glosat/active_learning-2/mmdetection/dla/config/cascadeRCNN_ignore_all_but_cells.py'
table_checkpoint_file = '/data/glosat/active_learning/models/model_fulltables_only_GloSAT.pth'

model = init_detector(config_file, table_checkpoint_file, device='cuda:0')

THRESHOLD = 0.5
CLASSES = ("table_body","cell","full_table","header","heading")
color = 255

# iterate over PDF pages
pdf_directory = '/data/glosat/glosat_table_dataset/datasets/new/stations#Catherine_Ross_DWR#1900A'

files = os.listdir(pdf_directory)
for filename in files:
	if '.pdf' in filename: 
		pdf_file = f'{pdf_directory}/{filename}'
		filename = filename.split('.')[0]
		pages = convert_from_path(pdf_file)
		for image_index, page in enumerate(pages):
			if not os.path.exists(f"{pdf_directory}/images"):
				os.mkdir(f"{pdf_directory}/images")

			img = page.convert()
			image = np.asarray(img)
			full_tables, tables = detect_table_region(model,image)
			if len(tables)>0:
				page.save(f"{pdf_directory}/images/{filename}_{image_index}.jpg", 'JPEG')

				for box in tables:
					box = list(map(int, box[0:4]))
					put_box(image,box,(0,color,color)) # Cyan 
				
				im_pil = Image.fromarray(image)
				im_pil.save(f"{pdf_directory}/images_plot/{filename}_{image_index}.jpg")
