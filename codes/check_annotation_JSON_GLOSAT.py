import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image
from PIL import Image, ImageDraw, ImageFont
import textwrap
import json
import os

from dla.src.image_utils import put_box, put_line

annotation_file = '/data/glosat/Appen/Annotation/cell_check/GlOSAT/batches/batch1/pilot-224/job_2114013.json'

info = []
for line in open(annotation_file, 'r'):
    info.append(json.loads(line))

def thresholding(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def image_preprocessing(img):
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

def plot_text_image(image, annotated_image,bbox,digitized_text):
    # Create a new image with the same height as the original image and double the width
    x,y,x_max,y_max = bbox
    y0 = y
    font_size = 30
    # font_size = int(image.size[1]*0.01)
    # Choose a font and font size for the text
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", size=font_size)
    
    # Paste the original image on the left side of the new image
    annotated_image.paste(image, (0, 0))
    
    width, height = image.size
    
    digitized_text = digitized_text.encode('utf-8')
    
    text = digitized_text.decode('utf-8')

    # Create a drawing context for the new image
    draw = ImageDraw.Draw(annotated_image)
    # Calculate the position for the top-left corner of the rectangle
    rectangle_position = (x, y)

    # Calculate the size of the rectangle needed to fit the text
    text_size = draw.textsize(digitized_text)

    # Calculate the position for the bottom-right corner of the rectangle
    rectangle_size = (x + text_size[0], y + text_size[1])

    # Calculate the new x-coordinate for the digitized text
    new_x = annotated_image.width // 2 + x
    
    text_width, text_height = font.getsize(text)
    
    box_x1, box_y1 = x, y - text_height
    box_x2, box_y2 = x + text_width, y
        
    if box_x2 > image.width:
        # wrapped_text = textwrap.wrap(text, width=int(width*0.2))
        wrapped_text = textwrap.wrap(text, width=int(0.15*(x_max-x)))
    #     print(new_x)

        # Draw the red rectangle behind the text on the new image
        # draw.rectangle([(new_x, y), (new_x + text_size[0], y + text_size[1])])

        for line in wrapped_text:
            # Draw the line of text
            draw.text((new_x, y), line, font=font, fill="blue", align="left", spacing=10, multiline=True)

            # Update the y-coordinate for the next line
            y += font.getsize(line)[1]

        # text_box = (x, y, x + max_width, y + max_height)
        draw.rectangle([(new_x, y0), (new_x + text_size[0], y + text_size[1])], outline="black")
        # draw.rectangle(text_box, outline="red")
        
    else:

        # Calculate the position for the bottom-right corner of the rectangle
        rectangle_size = (x + text_size[0], y + text_size[1])

        # Calculate the new x-coordinate for the digitized text
        new_x = annotated_image.width // 2 + x

        # Draw the red rectangle behind the text on the new image
        draw.rectangle([(new_x, y), (new_x + text_size[0], y_max + text_size[1])], outline="red")

        # Draw the text on top of the red rectangle at the new position on the new image
        draw.text((new_x, y), text, font=font, fill="green", align="left", spacing=10, multiline=True)

    return annotated_image


work_dir = f'/data/glosat/Appen/Annotation/cell_check/GlOSAT/batches/batch1'

for i in range(len(info)):
    worker_id =info[i]['results']['judgments'][0]['worker_id']
    image_name = info[i]['data']['image_name']
    json_object = json.loads(info[i]['results']['judgments'][0]['data']['annotation'])
    image_id = info[i]['id']
    
    image_path = f'{work_dir}/Image/{image_name}.jpg'
    
    
    # print(image_name)

    # image = cv.imread(image_path)
    # image, height, width, _ = image_preprocessing(img)
    image = Image.open(image_path)
    annotated_image = Image.new("RGB", (image.width * 2, image.height), color="white")

    cells = []
    err_cells = []
    color = 255
    missed = False
    for obj in json_object:
        bbox = obj['coordinates']
        bbox = [bbox['x'],bbox['y'],bbox['x']+bbox['w'],bbox['y']+bbox['h']]
        
        if 'text' in obj['metadata']['shapeTranscription']:
            text = obj['metadata']['shapeTranscription']['text']
            annotated_image = plot_text_image(image, annotated_image,bbox,text)
            cells.append(bbox)
        else:
            err_cells.append(bbox)
            missed = True

    open_cv_image = np.array(annotated_image) 

    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    for bbox in cells:
        put_box(open_cv_image,bbox,(0,0,color))

    for bbox in err_cells:
        put_box(open_cv_image,bbox,(color,0,0))

    im_pil = Image.fromarray(open_cv_image)

    if not os.path.exists(f"{work_dir}/Appen/{worker_id}/"):
        os.mkdir(f"{work_dir}/Appen/{worker_id}")
    

    if missed:
        im_pil.save(f"{work_dir}/Appen/{worker_id}/missed_{image_id}_{image_name}.jpg")
        print(f'missed {image_id}\t{image_name}')

    else:
        im_pil.save(f"{work_dir}/Appen/{worker_id}/{image_name}.jpg")