import sys
sys.path.append('../src')

import glosat_utils
from glosat_utils import *
import matplotlib.pyplot as plt 
import pickle
import bs4 as bs

import cv2
import cv2 as cv
import imutils
import os
from os import path
import numpy as np
import pickle

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
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

from bs4 import BeautifulSoup, Tag
try:
    import xml_utils as xml_utils
except:
    import dla.src.xml_utils as xml_utils

import argparse
import os
import copy 


def get_coordinates_transkribus(cordinates):
    imagedraw=[]
    points=cordinates.split()
    x_pt = []
    y_pt = []
    for cord in points:
        (x,y)=cord.split(',')
        x_pt.append(int(x))
        y_pt.append(int(y))
        
    imagedraw.append(min(x_pt))
    imagedraw.append(min(y_pt))
    imagedraw.append(max(x_pt))
    imagedraw.append(max(y_pt))
    return(imagedraw)

datapath = '/data/glosat/Code-Git/docExtractor-master/glosat/Data_conversion/Fine'
transkribus = f'{datapath}/Transkribus'
icdar = f'{datapath}/ICDAR'
output_path = f'{datapath}/ICDAR_Transkribus'

files = os.listdir(transkribus)
for filename in files:

    # Read Transkribus XML file
    f = open(f'{transkribus}/{filename}', 'r')
    file = f.read()
    soup = bs.BeautifulSoup(file,'xml')

    metadata = soup.find('Metadata')
    page = soup.find('Page')
    print(page['imageFilename'],page['imageWidth'],page['imageHeight'])

    table_regions = soup.findAll('TableRegion')
    header = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"></PcGts>'''

    temp = table_regions[0].find('Coords')
    custom = table_regions[0]['custom']

    # Read ICDAR XML File
    f = open(f'{icdar}/{filename}', 'r')
    file = f.read()
    icdar_soup = bs.BeautifulSoup(file,'xml')

    tables = icdar_soup.findAll('table')

    corner = '''<CornerPts>0 1 2 3</CornerPts>'''
    corner_new = bs.BeautifulSoup(corner, 'xml')

    corner_pts =  corner_new.find('CornerPts')

    soup_new = bs.BeautifulSoup(header, 'xml')
    soup_new.select_one('PcGts').append(metadata)

    page_point = Tag(builder=soup_new.builder, 
                   name='Page', 
                   attrs={'imageFilename':page['imageFilename'],'imageWidth':page['imageWidth'],'imageHeight':page['imageHeight']})

    soup_new.select_one('PcGts').append(page_point)

    for table in tables:
        coords =  table.find('Coords')
        table_bbox =  get_coordinates_transkribus(coords['points'])
        trans_cell = f'{table_bbox[0]},{table_bbox[1]} {table_bbox[2]},{table_bbox[1]} {table_bbox[2]},{table_bbox[3]} {table_bbox[0]},{table_bbox[3]}'
        temp_copy = copy.copy(temp) 
        temp_copy['points'] = trans_cell
    #     print(table['id'],coords['points'],table_bbox,trans_cell)
        
        in_point = Tag(builder=soup_new.builder, 
                   name='TableRegion', 
                   attrs={'id':table['id'],'custom':custom})
        in_point.append(temp_copy)
        
        cells = table.findAll('cell')
        count = len(cells)
        print(f'Table: {trans_cell} #cells:{count}')
        #Cell addition
        for cell in cells:
            rowspan = abs(int(cell['start-row'])-int(cell['end-row']))
            colspan = abs(int(cell['start-col'])-int(cell['end-col']))
            print(rowspan,colspan)
            if cell.has_attr('header'):
                print(2,'---',cell)
                if cell['header'] == 'true':
                    cell_point = Tag(builder=soup_new.builder, name='TableCell', 
                                 attrs={'row':cell['start-row'], 'col':cell['start-col'], 'id': cell['id'], 'rowSpan':rowspan, 'colSpan':colspan, 'type':"header", 'custom':'''structure {type:header;}'''})
                else:
                    cell_point = Tag(builder=soup_new.builder, name='TableCell', 
                                 attrs={'row':cell['start-row'], 'col':cell['start-col'], 'id': cell['id'], 'rowSpan':rowspan, 'colSpan':colspan})
            else:
                print(1,'---',cell)
                cell_point = Tag(builder=soup_new.builder, name='TableCell', 
                             attrs={'row':cell['start-row'], 'col':cell['start-col'], 'id': cell['id'], 'rowSpan':rowspan, 'colSpan':colspan})
                
            cell_bbox = cell.find('Coords')
            cell_bbox = get_coordinates_transkribus(cell_bbox['points'])
            trans_cell = f'{cell_bbox[0]},{cell_bbox[1]} {cell_bbox[0]},{cell_bbox[3]} {cell_bbox[2]},{cell_bbox[3]} {cell_bbox[2]},{cell_bbox[1]}'
    #         trans_cell = f'{cell_bbox[0]},{cell_bbox[1]} {cell_bbox[0]},{cell_bbox[3]} {cell_bbox[2]},{cell_bbox[3]} {cell_bbox[2]},{cell_bbox[1]}'
            temp_copy = copy.copy(temp) 
            temp_copy['points'] = trans_cell
            cell_point.append(temp_copy)
            temp_copy = copy.copy(corner_pts) 
            cell_point.append(temp_copy)
            in_point.append(cell_point) 
        soup_new.select_one('Page').append(in_point)

    with open(f"{output_path}/{filename}", "w") as file:
        file.write(str(soup_new))