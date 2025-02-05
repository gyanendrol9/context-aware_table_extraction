from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer
from PIL import Image


#Training case
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence


import os
import pickle as pkl

import os
import json
import jsonlines
import sys

cache_dir = "pretrained_model" #cache dir pretrained models

img_source = sys.argv[1]
outdir = sys.argv[2]

if not os.path.exists(outdir):
    os.mkdir(outdir)
    
ann_jsonl = f"{img_source}/textrecog_train.json" # This dataset also contain the neighbour cell information added for each target cell to train.

data_dict = []
with jsonlines.open(ann_jsonl) as f:
    for line in f.iter():
        for annotation in line['data_list']:
            img_path = os.path.join(img_source, annotation['img_path'])
            text = []
            for txt in annotation['instances']:
                label = txt['text']
                if label != '@@@' or label != '$$$' or label != '###':
                    text.append(label)
                else:
                    print(annotation)
            data_dict.append(dict(img=img_path, text='\n'.join(text)))
train_data_dict = data_dict


ann_jsonl = f"{img_source}/textrecog_val.json" # This dataset also contain the neighbour cells information added for each target cell to train.

data_dict = []
with jsonlines.open(ann_jsonl) as f:
    for line in f.iter():
        for annotation in line['data_list']:
            img_path = os.path.join(img_source, annotation['img_path'])
            text = []
            for txt in annotation['instances']:
                label = txt['text']
                if label != '@@@' or label != '$$$' or label != '###':
                    text.append(label)
                else:
                    print(annotation)
            data_dict.append(dict(img=img_path, text='\n'.join(text)))
val_data_dict = data_dict

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
        image = image.resize(self.image_size)  # Resize the image to the fixed size

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

def calculate_average_image_size(image_paths):
    total_height = 0
    total_width = 0
    num_images = len(image_paths)

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
            total_width += width
            total_height += height

    avg_width = total_width // num_images
    avg_height = total_height // num_images

    return avg_width, avg_height

image_paths = [id['img'] for id in train_data_dict+val_data_dict]
texts = [id['text'] for id in train_data_dict+val_data_dict]
word_len = [len(id['text'].split()) for id in train_data_dict+val_data_dict]

average_size = (120, 80)
max_len = 190
batch_size = 32

ocr_model_name = "microsoft/trocr-large-handwritten"

processor = TrOCRProcessor.from_pretrained(ocr_model_name, cache_dir=cache_dir)
ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_name, cache_dir=cache_dir)

ocr_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
ocr_model.config.pad_token_id = processor.tokenizer.pad_token_id
ocr_model.config.vocab_size = ocr_model.config.decoder.vocab_size

train_dataset = MyTrainDataset(image_paths, texts, processor, max_len, image_size=average_size)
# train_dataset = MyTrainDataset(image_paths, texts, processor, max_len)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

del train_dataset

t5_model_name = 'yelpfeast/byt5-base-english-ocr-correction'

t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name, cache_dir=cache_dir)
ocr_tokenizer = AutoTokenizer.from_pretrained(t5_model_name, cache_dir=cache_dir)

optimizer = Adam(ocr_model.parameters(), lr=1e-5)  # Adjust the learning rate as needed
loss_function = nn.CrossEntropyLoss()  # Define the appropriate loss function for your OCR task

checkpoint_path = f"{outdir}/pretrained_previous_checkpoint.pth"
print(f'Loading checkpoint {checkpoint_path}')

if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())
    
    # Remove 'module.' prefix from keys if loading on a single GPU
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the new_state_dict into the model
    ocr_model.load_state_dict(new_state_dict)

    # Load the ocr model's state dictionary from the checkpoint
    pre_epoch = checkpoint['epoch']
    del checkpoint
else:
    pre_epoch = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use DataParallel to utilize multiple GPUs
gpus = torch.cuda.device_count()
if gpus > 1:# Check if PyTorch can access GPUs
    print(f"Number of GPUs available: {gpus}")
    ocr_model = torch.nn.DataParallel(ocr_model)
    t5_model = torch.nn.DataParallel(t5_model)

ocr_model.to(device)
t5_model.to(device)

max_input_length = 1200
tot_epochs = 40
for epoch in range(pre_epoch+1,tot_epochs):
    ocr_model.train()
    total_loss = 0

    print(f"Starting training epoch: {epoch+1}")

    for batch_num, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        images = batch["pixel_values"].to(device)
        labels = batch["texts_ids"].squeeze(1).long().to(device)
        outputs = ocr_model(pixel_values= images, labels=labels)
        l1 = outputs.loss.mean()  # Make sure l1 is a scalar
        
        generated_ids = ocr_model.module.generate(images)
        label_tokens = processor.batch_decode(labels, skip_special_tokens=True)
        generated_tokens = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        generated_tensor_list = [torch.tensor(list(text.encode("utf-8"))) for text in generated_tokens]
        padded_tensors = pad_sequence([torch.cat((tensor, torch.zeros(max_input_length - len(tensor)))) for tensor in generated_tensor_list], batch_first=True)
        
        label_tensor_list = [torch.tensor(list(text.encode("utf-8"))) for text in label_tokens]
        label_padded_tensors = pad_sequence([torch.cat((tensor, torch.zeros(max_input_length - len(tensor)))) for tensor in label_tensor_list], batch_first=True)
        
        loss = t5_model(padded_tensors.long().to(device), labels=label_padded_tensors.long().to(device)).loss.mean()   # forward pass
        avg_loss = 0.5*(loss+l1)
        
        # First, backpropagate TrOCR loss
        l1.backward(retain_graph=True)  # retain the graph to allow backprop for the second model

        # Then, backpropagate the ByT5 (t5_model) loss
        loss.backward()

        # avg_loss.backward()
        # Step 5: Optimizer step to update parameters
        optimizer.step()

        # Optionally, accumulate total loss
        total_loss += (l1.item() + loss.item())  # sum the two losses for reporting
        
        print(f"Batch: {batch_num + 1} Epoch:{epoch}/{tot_epochs} avg_loss: {avg_loss} OCR Loss: {l1}, T5 Loss: {loss}", end='\r')
        # optimizer.step()

    # Save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': ocr_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        't5_model_state_dict': t5_model.state_dict()        # Add any other information you want to save
    }
    torch.save(checkpoint, f'{outdir}/combined_dataset_sep-loss_epoch_{epoch}.pth')
    print(f'Checkpoint saved: {outdir}/combined_dataset_sep-loss_epoch_{epoch}.pth')

    average_loss = total_loss / len(train_dataloader)
    print(f"\nEpoch: {epoch}, Loss: {average_loss}")
    