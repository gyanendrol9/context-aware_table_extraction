from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration

from PIL import Image, ImageDraw, ImageFont

def draw_bbox(image, bbox):
    new_width = image.width
    new_height = image.height
    new_image = Image.new("RGB", (new_width, new_height), color="white")
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    for box in bbox:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
    return new_image


#Training case
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.metrics import distance

import os
import sys
import json
import jsonlines

import cv2 as cv
import numpy as np


img_dir = sys.argv[1]
out_dir = sys.argv[2]

workdir = 'Tabular-Data-Extraction'


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

import matplotlib.pyplot as plt 
import textwrap


def plot_text_image(draw, annotated_image,bbox,digitized_text):
    # Create a new image with the same height as the original image and double the width
    x,y,x_max,y_max = bbox[:4]
    y0 = y
    
    digitized_text = digitized_text.encode('utf-8')
    
    text = digitized_text.decode('utf-8')

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
            wrapped_text = [text]

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
        
    else:

        # Calculate the position for the bottom-right corner of the rectangle
        rectangle_size = (x + text_size[0], y + text_size[1])

        # Draw the red rectangle behind the text on the new image
        if special_flag:
            draw.rectangle([(new_x, y), (new_x + text_size[0], y_max + text_size[1])], outline=color_flag[special_flag],  width=6)
        else:
            draw.rectangle([(new_x, y), (new_x + text_size[0], y_max + text_size[1])], outline=color_flag[special_flag])

        # Draw the text on top of the red rectangle at the new position on the new image
        draw.text((new_x, y), text, font=font, fill="green", align="left", spacing=10, multiline=True)

    return 0


def get_ocr_output(image):
    pixel_values = preprocess_image(image)

    pixel_values = pixel_values.to(device)
    # Generate prediction with logits
    outputs = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True, max_length=16)

    # Decode the predictions
    generated_ids = outputs.sequences
    predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    inputs = ocr_tokenizer(predicted_texts, return_tensors="pt", padding=True).to(device)
    output_sequences = ocr_model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        do_sample=False,  # disable sampling to test if batching affects output

    )
    spell_correction = ocr_tokenizer.batch_decode(output_sequences.to(device), skip_special_tokens=True)


    # Get the logits (confidence scores)
    logits = outputs.scores

    # Convert logits to probabilities (softmax)
    probabilities = [F.softmax(logit, dim=-1) for logit in logits]

    # Extract confidence scores for the predicted tokens and combine tokens with their confidence scores
    tokens_with_confidence = []
    for i, token_id in enumerate(generated_ids[0]):
        text = tokenizer.decode(token_id)
        if i < len(probabilities):  # Skip if index out of range
            score = probabilities[i][0, token_id].item()
        # if text!='<s>' and text!='</s>':
        tokens_with_confidence.append((text,score))

    scores = []
    for token, score in tokens_with_confidence:
        scores.append(score)
        # print(f"Token: {token}, Confidence Score: {score:.4f}")

    scores = sum(scores)/len(tokens_with_confidence)
    del pixel_values, probabilities, logits, image
    return predicted_texts, spell_correction, scores, tokens_with_confidence

def get_ocr_output_v1(image):
    pixel_values = preprocess_image(image)

    pixel_values = pixel_values.to(device)
    # Generate prediction with logits
    outputs = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True, max_length=16)

    # Decode the predictions
    generated_ids = outputs.sequences
    predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Get the logits (confidence scores)
    logits = outputs.scores

    # Convert logits to probabilities (softmax)
    probabilities = [F.softmax(logit, dim=-1) for logit in logits]

    # Extract confidence scores for the predicted tokens and combine tokens with their confidence scores
    tokens_with_confidence = []
    for i, token_id in enumerate(generated_ids[0]):
        text = tokenizer.decode(token_id)
        if i < len(probabilities):  # Skip if index out of range
            score = probabilities[i][0, token_id].item()
        # if text!='<s>' and text!='</s>':
        tokens_with_confidence.append((text,score))

    scores = []
    for token, score in tokens_with_confidence:
        scores.append(score)
        # print(f"Token: {token}, Confidence Score: {score:.4f}")

    scores = sum(scores)/len(tokens_with_confidence)
    del pixel_values, probabilities, logits, image
    return predicted_texts, scores, tokens_with_confidence

#Load model
model_name = "microsoft/trocr-large-handwritten"

tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

ocr_model = T5ForConditionalGeneration.from_pretrained('yelpfeast/byt5-base-english-ocr-correction')
ocr_tokenizer = AutoTokenizer.from_pretrained("yelpfeast/byt5-base-english-ocr-correction")

# Specify the path to the checkpoint file
checkpoint_path = f"{workdir}/TrOCR-GloSAT-DRAfrica-without-augmentation/combined_dataset_checkpoint_epoch_2.pth"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Remove 'module.' prefix from keys if loading on a single GPU
new_state_dict = {}
for key, value in checkpoint['model_state_dict'].items():
    new_key = key.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[new_key] = value

# Load the new_state_dict into the model
model.load_state_dict(new_state_dict)

if 'ocr_model_state_dict' in checkpoint:
    ocr_model.load_state_dict(checkpoint['ocr_model_state_dict'])
else:
    ocr_model.load_state_dict(checkpoint['ocr_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ocr_model.to(device)

color_flag = {}
color_flag[0] = (0, 0, 0)       # Black correct
color_flag[1] =  (255, 5, 0)     # Red Not easy to transcribe 
color_flag[2] = (255, 0, 255)   #Magenta Partial text
color_flag[3] =  (128, 128, 0) #Yellow Blank


# img_source = f'{img_dir}/{file}'
outpathJSON = f'{out_dir}/JSON-OCR'
if not os.path.exists(outpathJSON):
    os.mkdir(outpathJSON)

# img_source = f'{img_dir}/{file}'
outpathJPG = f'{out_dir}/Image-OCR'
if not os.path.exists(outpathJPG):
    os.mkdir(outpathJPG)
    
# img_source = f'{img_dir}/{file}'
outpathJPGspell = f'{out_dir}/Image-OCR-spell'
if not os.path.exists(outpathJPGspell):
    os.mkdir(outpathJPGspell)


# files =  os.listdir(f'{img_dir}')
files =  os.listdir(f'{out_dir}/JSON-det')
for file in files:
    print(f'\nText extraction of image: {file}\n')
    filename = file.split('.')[0]
    img_source = f'{img_dir}/{filename}.jpg'
    ann_jsonl = f"{out_dir}/JSON-det/{filename}.json"
        
    if os.path.exists(img_source) and not os.path.exists(f'{outpathJSON}/{filename}.json'):

        with open(ann_jsonl, 'r') as json_file:
            # Load the contents of the file into a Python object (dictionary or list)
            data = json.load(json_file)

        # Print the contents of the JSON file
        print(data.keys())

        img = cv.imread(img_source)
        w,h,c = img.shape
        # img, h, w, c = image_preprocessing(img)
        print('Image shape:',w,h,c)

        text_cells = list()
        all_cells = len(data['table_cells'])
        for i, cell in enumerate(data['table_cells']):
            print(f'Extracting text from cell {cell} -- {i}/{all_cells}')

            # Check if the bounding box extends outside the image boundaries
            if cell[0]>cell[2] or cell[1]>cell[3]:
                print(f"Bounding box is out of bounds {cell}, discarding.")
            else:
                if int(cell[1]) > 5 and int(cell[3])+25<h:
                    croppedimage=img[int(cell[1])-5:int(cell[3])+25,int(cell[0]):int(cell[2])] #img[y:y+h, x:x+w]
                elif int(cell[3])+25<h:
                    croppedimage=img[int(cell[1]):int(cell[3])+25,int(cell[0]):int(cell[2])] #img[y:y+h, x:x+w]
                else:
                    croppedimage=img[int(cell[1]):int(cell[3]),int(cell[0]):int(cell[2])] #img[y:y+h, x:x+w]
                    
                im_pil = Image.fromarray(croppedimage)
                text_info = get_ocr_output(im_pil)
                # text_info = (predicted_texts, spell_correction, scores, tokens_with_confidence)
                text_cells.append((cell, text_info))
            

        image = Image.open(img_source)
        annotated_image = Image.new("RGB", (image.width * 2, image.height), color="white")
        annotated_image_spell = Image.new("RGB", (image.width * 2, image.height), color="white")

        im_width, im_height = image.size
        annotated_image.paste(image, (0, 0))
        annotated_image_spell.paste(image, (0, 0))

        # Create a drawing context for the new image
        draw = ImageDraw.Draw(annotated_image)
        draw_spell = ImageDraw.Draw(annotated_image_spell)

        font_size = 20
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", size=font_size)

        for cell, text_info in text_cells:
            (predicted_texts, spell_correction, scores, tokens_with_confidence) = text_info
            x,y,x_max,y_max = cell[:4]
            draw.rectangle([(x, y), (x_max, y_max)], outline='blue',  width=6)
            plot_text_image(draw, annotated_image, cell, predicted_texts[0])
            plot_text_image(draw_spell, annotated_image_spell, cell, spell_correction[0])
        
        annotated_image.save(f"{outpathJPG}/{filename}-text-extract.jpg")
        annotated_image_spell.save(f"{outpathJPGspell}/{filename}-text-extract_spell.jpg")


        data['cell_texts'] = text_cells
        
        # Save to a JSON file
        with open( f'{outpathJSON}/{filename}.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f'JSON save at: {outpathJSON}/{filename}.json')
