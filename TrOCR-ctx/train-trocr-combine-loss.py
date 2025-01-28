from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from transformers import AutoTokenizer, AutoProcessor, AutoModelForTokenClassification

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

from torch.utils.data import Dataset

# import nlpaug.augmenter.char as nac
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence


import os
import pickle as pkl

print('Loading training data')
f = open('/scratch/gsl1r22/TDR/data/dr_africa_train_data_dict_new.pkl','rb')
dr_africa_cells_complete = pkl.load(f)
f.close()

# f = open('/scratch/gsl1r22/GloSAT/text-recognition/data/pickle_data-iridis/glosat_train_data_dict.pkl','rb')
# glosat_cells_complete = pkl.load(f)
# f.close()

outdir = '/scratch/gsl1r22/TDR/finetuned_model/TrOCR-GloSAT-DRAfrica-without-augmentation'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
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

image_paths = [id['img'] for id in dr_africa_cells_complete]
texts = [id['text'] for id in dr_africa_cells_complete]
word_len = [len(id['text'].split()) for id in dr_africa_cells_complete]

del dr_africa_cells_complete

# image_paths = [id['img'] for id in dr_africa_cells_complete+glosat_cells_complete]
# texts = [id['text'] for id in dr_africa_cells_complete+glosat_cells_complete]
# word_len = [len(id['text'].split()) for id in dr_africa_cells_complete+glosat_cells_complete]
# del dr_africa_cells_complete, glosat_cells_complete

average_size = (120, 80)
max_len = 190
batch_size = 32

print(f'Loading pretrained models')
cache_dir = "/scratch/gsl1r22/TDR/pretrained_model/"

model_name = '/scratch/gsl1r22/TDR/pretrained_model/models--microsoft--trocr-large-handwritten/snapshots/e68501f437cd2587ae5d68ee457964cac824ddee'
# model_name = "microsoft/trocr-large-handwritten"

processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

train_dataset = MyTrainDataset(image_paths, texts, processor, max_len, image_size=average_size)
# train_dataset = MyTrainDataset(image_paths, texts, processor, max_len)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

del train_dataset

ocr_model_name = '/scratch/gsl1r22/TDR/pretrained_model/models--yelpfeast--byt5-base-english-ocr-correction/snapshots/19d5c2fd86b87f0a0febb7d2574878a0d68d5294'
ocr_model = T5ForConditionalGeneration.from_pretrained(ocr_model_name, cache_dir=cache_dir)
ocr_tokenizer = AutoTokenizer.from_pretrained(ocr_model_name, cache_dir=cache_dir)
# ocr_model.to(device)

optimizer = Adam(model.parameters(), lr=1e-5)  # Adjust the learning rate as needed
loss_function = nn.CrossEntropyLoss()  # Define the appropriate loss function for your OCR task

checkpoint_path = f"/scratch/gsl1r22/TDR/finetuned_model/TrOCR-GloSAT-DRAfrica-without-augmentation/combined_dataset_sep-loss_epoch_14.pth"
# checkpoint_path = f"/scratch/gsl1r22/GloSAT/text-recognition/finetuned_model/TrOCR-GloSAT-DRAfrica-without-augmentation/combined_dataset_checkpoint_epoch_0.pth"
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
    model.load_state_dict(new_state_dict)

    # Load the ocr model's state dictionary from the checkpoint
    # ocr_model.load_state_dict(checkpoint['ocr_state_dict'])
    pre_epoch = checkpoint['epoch']
else:
    pre_epoch = 0
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use DataParallel to utilize multiple GPUs
gpus = torch.cuda.device_count()
if gpus > 1:# Check if PyTorch can access GPUs
    print(f"Number of GPUs available: {gpus}")
    model = torch.nn.DataParallel(model)
    ocr_model = torch.nn.DataParallel(ocr_model)

model.to(device)
ocr_model.to(device)

del checkpoint
max_input_length = 1200

tot_epochs = 40
for epoch in range(pre_epoch+1,tot_epochs):
    model.train()
    total_loss = 0

    print(f"Starting training epoch: {epoch+1}")

    for batch_num, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        images = batch["pixel_values"].to(device)
        labels = batch["texts_ids"].squeeze(1).long().to(device)
        outputs = model(pixel_values= images, labels=labels)
        l1 = outputs.loss.mean()  # Make sure l1 is a scalar
        
        generated_ids = model.module.generate(images)
        label_tokens = processor.batch_decode(labels, skip_special_tokens=True)
        generated_tokens = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        generated_tensor_list = [torch.tensor(list(text.encode("utf-8"))) for text in generated_tokens]
        padded_tensors = pad_sequence([torch.cat((tensor, torch.zeros(max_input_length - len(tensor)))) for tensor in generated_tensor_list], batch_first=True)
        
        label_tensor_list = [torch.tensor(list(text.encode("utf-8"))) for text in label_tokens]
        label_padded_tensors = pad_sequence([torch.cat((tensor, torch.zeros(max_input_length - len(tensor)))) for tensor in label_tensor_list], batch_first=True)
        
        loss = ocr_model(padded_tensors.long().to(device), labels=label_padded_tensors.long().to(device)).loss.mean()   # forward pass
        avg_loss = 0.5*(loss+l1)
        
        # First, backpropagate TrOCR loss
        l1.backward(retain_graph=True)  # retain the graph to allow backprop for the second model

        # Then, backpropagate the ByT5 (ocr_model) loss
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
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ocr_model_state_dict': ocr_model.state_dict()        # Add any other information you want to save
    }
    torch.save(checkpoint, f'{outdir}/combined_dataset_sep-loss_epoch_{epoch}.pth')
    print(f'Checkpoint saved: {outdir}/combined_dataset_sep-loss_epoch_{epoch}.pth')

    average_loss = total_loss / len(train_dataloader)
    print(f"\nEpoch: {epoch}, Loss: {average_loss}")
    