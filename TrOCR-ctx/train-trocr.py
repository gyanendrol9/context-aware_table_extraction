from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer

from PIL import Image, ImageDraw
    
def draw_bbox(image, bbox):
    new_width = image.width
    new_height = image.height
    new_image = Image.new("RGB", (new_width, new_height), color="white")
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    for box in bbox:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
    return new_image


import torch
import torch.nn as nn
import cv2 as cv
from sklearn.model_selection import train_test_split

import os
import json
import sys


work_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

files = os.listdir(f'{work_dir}/Images')

train_ids, test_ids = train_test_split(files, test_size=0.1, random_state=42)
# Note: save these ids for evaluation purpose

with open(f"{work_dir}/UoS_Data_Rescue_TDE_dataset.json", 'r') as json_file:
    total_data_dict = json.load(json_file)
    
train_data_dict = []
images = {}
cfiles = len(total_data_dict)

for c, data_dict in enumerate(total_data_dict):
    img_path = data_dict['img_path']
    file = img_path.split('/')[-1]
    if file in train_ids:
        print(f'Processing {file} {c}/{cfiles}...')
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
            
image_paths = [id['img'] for id in train_data_dict]
texts = [id['text'] for id in train_data_dict]

del train_data_dict, total_data_dict

from torch.utils.data import Dataset
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
    
average_size = (120, 80)
max_len = 190
batch_size = 32
max_input_length = 190
tot_epochs = 40

print(f'Loading pretrained models')
cache_dir = "pretrained_model"

model_name = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use DataParallel to utilize multiple GPUs
gpus = torch.cuda.device_count()
if gpus > 1:# Check if PyTorch can access GPUs
    print(f"Number of GPUs available: {gpus}")
    model = torch.nn.DataParallel(model)

model.to(device)

optimizer = Adam(model.parameters(), lr=1e-5)  # Adjust the learning rate as needed
loss_function = nn.CrossEntropyLoss()  # Define the appropriate loss function for your OCR task

train_dataset = MyTrainDataset(image_paths, texts, processor, max_len, image_size=average_size)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

del train_dataset, image_paths, texts

for epoch in range(tot_epochs):
    model.train()
    total_loss = 0

    print(f"Starting training epoch: {epoch+1}")

    for batch_num, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        images = batch["pixel_values"].to(device)
        labels = batch["texts_ids"].squeeze(1).long().to(device)
        outputs = model(pixel_values= images, labels=labels)
        l1 = outputs.loss.mean()  # Make sure l1 is a scalar
        
        l1.backward()
        # Step 5: Optimizer step to update parameters
        optimizer.step()

        # Optionally, accumulate total loss
        total_loss += l1.item()   
        
        print(f"Batch: {batch_num} Epoch:{epoch}/{tot_epochs} TrOCR Loss: {l1}", end='\r')
        # optimizer.step()

    # Save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    torch.save(checkpoint, f'{out_dir}/combined_dataset_checkpoint_epoch_{epoch}.pth')
    print(f'Checkpoint saved: {out_dir}/combined_dataset_checkpoint_epoch_{epoch}.pth')

    average_loss = total_loss / len(train_dataloader)
    print(f"\nEpoch: {epoch}, Loss: {average_loss}")
