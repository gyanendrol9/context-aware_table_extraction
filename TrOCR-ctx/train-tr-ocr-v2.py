from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from transformers import AutoTokenizer, AutoProcessor, AutoModelForTokenClassification

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt 
    
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

    TrOCRConfig,

    TrOCRProcessor,

    TrOCRForCausalLM,

    ViTConfig,

    ViTModel,

    VisionEncoderDecoderModel,
    ViTFeatureExtractor,

)

import torch
import torch.nn as nn
from nltk.metrics import distance
import cv2 as cv
import random

import os
import json
import jsonlines

data_dict = []
glosat_data_dict = []

work_dir = '/home/gyanendro/Desktop/mm-ocr-update/data/'
outJSON = f'{work_dir}/Augmented-JSON'

files = os.listdir(f'{work_dir}/Images')

with open(f"{work_dir}/completed_annotated_dataset.json", 'r') as json_file:
    total_data_dict = json.load(json_file)
    
train_data_dict = []
images = {}
for data_dict in total_data_dict:
    for data in data_dict['cell_info']:
        img_path = data['img_path']
        file = img_path.split('/')[-1]
        box = data['cell']
        text_info = data['text']

        img_dir = f'{work_dir}/{img_path}'
        
        if file not in images:
            img = cv.imread(img_dir)
            images[file] = img
        else:
            img = images[file]
            
        if '@@@' not in text_info  or '$$$' not in text_info or '###' not in text_info:
            croppedimage=img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            pil_image = Image.fromarray(croppedimage)
            train_data_dict.append(dict(img=pil_image, text=text_info))
            
augmented_cells_complete = []

cfiles = len(files)
for c, file in enumerate(files):
    filename = file.split('.')[0]
    ann_jsonl = f"{outJSON}/{filename}.json"
    
    if os.path.exists(ann_jsonl):
        print(f'Processing {file} {c}/{cfiles}...')
        with open(ann_jsonl, 'r') as json_file:
            # Load the contents of the file into a Python object (dictionary or list)
            augmented_cells_c = json.load(json_file)
            
            img_path = augmented_cells_c['img_path']
            file = img_path.split('/')[-1]
            box = data['cell']
            text_info = data['text']
            if os.path.exists(f'{work_dir}/Images/{file}'):
                img_dir = f'{work_dir}/Images/{file}'
            else:
                img_dir = f'{dr_africa_dir}/Images/{file}'
           
            augmented_cells = augmented_cells_c['augmented_cells']
            
            if file not in images:
                img = cv.imread(img_dir)
                images[file] = img
            else:
                img = images[file]

            for (cell, text) in augmented_cells:
                # try:
                    croppedimage=img[int(cell[1]):int(cell[3]),int(cell[0]):int(cell[2])] #img[y:y+h, x:x+w]
                    im_pil = Image.fromarray(croppedimage)
                    augmented_cells_complete.append(dict(img=im_pil, text=text))
                # except:
                #     pass
                
from torch.utils.data import Dataset
from PIL import Image
# import nlpaug.augmenter.char as nac
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

class MyTrainDataset(Dataset):
    def __init__(self, images, texts, processor, max_len, image_size=(224, 224)):
        self.images = images
        self.texts = texts
        self.processor = processor
        self.max_len = max_len
        self.image_size = image_size  # Desired fixed image size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and resize the image
        image = self.images[idx].convert('RGB')
        image = image.resize(self.image_size)  # Resize the image to the fixed size}

        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)  # Remove the batch dimension

        # Tokenize and pad the text
        label_ids = self.processor.tokenizer(self.texts[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_len).input_ids
        label_tensor = label_ids.squeeze(0).float()  # Convert to tensor and remove batch dimension

        # Add the label tensor and text to the inputs
        inputs["texts_ids"] = label_tensor
        inputs["texts"] = self.texts[idx]

        return inputs
    
aug_cell_count = len(augmented_cells_complete)
pc = 0.6
considered_aug_cells = int(aug_cell_count * pc)

train_cell_count = len(train_data_dict)
pc = 0.6
considered_train_cells = int(train_cell_count * pc)

random.shuffle(train_data_dict)
random.shuffle(augmented_cells_complete)

image_paths = [id['img'] for id in train_data_dict[:considered_train_cells]+augmented_cells_complete[:considered_aug_cells]]
texts = [id['text'] for id in train_data_dict[:considered_train_cells]+augmented_cells_complete[:considered_aug_cells]]
word_len = [len(id['text'].split()) for id in train_data_dict[:considered_train_cells]+augmented_cells_complete[:considered_aug_cells]]

average_size = (120, 80)
max_len = 190

train_dataset = MyTrainDataset(image_paths, texts, processor, max_len, image_size=average_size)
# train_dataset = MyTrainDataset(image_paths, texts, processor, max_len)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
del train_dataset

print(f'Loading pretrained models')

cache_dir = "/home/gyanendro/.cache/huggingface/hub/"
model_name = "microsoft/trocr-large-handwritten"

processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ocr_model = T5ForConditionalGeneration.from_pretrained('yelpfeast/byt5-base-english-ocr-correction')
ocr_tokenizer = AutoTokenizer.from_pretrained("yelpfeast/byt5-base-english-ocr-correction")
ocr_model.to(device)

optimizer = Adam(model.parameters(), lr=1e-5)  # Adjust the learning rate as needed
loss_function = nn.CrossEntropyLoss()  # Define the appropriate loss function for your OCR task

max_input_length = 1200

tot_epochs = 40
for epoch in range(tot_epochs):
    model.train()
    total_loss = 0

    print(f"Starting training epoch: {epoch+1}")

    for batch_num, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        images = batch["pixel_values"].to(device)
        labels = batch["texts_ids"].squeeze(1).long().to(device)
        outputs = model(pixel_values= images, labels=labels)
        
        generated_ids = model.generate(images)
        label_tokens = processor.batch_decode(labels, skip_special_tokens=True)
        generated_tokens = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        generated_tensor_list = [torch.tensor(list(text.encode("utf-8"))) for text in generated_tokens]
        padded_tensors = pad_sequence([torch.cat((tensor, torch.zeros(max_input_length - len(tensor)))) for tensor in generated_tensor_list], batch_first=True)
        
        label_tensor_list = [torch.tensor(list(text.encode("utf-8"))) for text in label_tokens]
        label_padded_tensors = pad_sequence([torch.cat((tensor, torch.zeros(max_input_length - len(tensor)))) for tensor in label_tensor_list], batch_first=True)
        
        loss = ocr_model(padded_tensors.long().to(device), labels=label_padded_tensors.long().to(device)).loss # forward pass
        print(f"Batch: {batch_num + 1} Epoch:{epoch}/{tot_epochs} Loss: {loss}", end='\r')

        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    if epoch%2 == 1:
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'ocr_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            't5_state_dict': ocr_model.state_dict(),
            # Add any other information you want to save
        }
        torch.save(checkpoint, f'TR-OCR_models/combined_dataset_checkpoint_epoch_{epoch}.pth')
        print(f'Checkpoint saved: TR-OCR_models/combined_dataset_checkpoint_epoch_{epoch}.pth')

    print(f"\n")
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch: {epoch+1}, Loss: {average_loss}")