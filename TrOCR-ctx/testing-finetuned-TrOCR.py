
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from transformers import AutoTokenizer

from PIL import Image


import torch
import os
import json
import jsonlines
import sys


workdir = 'Text-recognition/'
img_source = f"{workdir}/data/glosat/"

model_name = "microsoft/trocr-large-handwritten"

tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the image
def load_image(image_path):
    return Image.open(image_path)

def preprocess_image(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values

def get_ocr_output(image):
    pixel_values = preprocess_image(image)

    pixel_values = pixel_values.to(device)
    # Generate prediction with logits
    outputs = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True, max_length=190)

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

class MyTrainDataset(Dataset):
    def __init__(self, image_paths, texts, processor, max_len):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        cimage = image.convert('RGB')
        inputs = self.processor(images=cimage, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        
        label_ids = processor.tokenizer(self.texts[idx], return_tensors="pt").input_ids
        label_ids = label_ids.squeeze(0).tolist()
        label_tensor = torch.tensor(label_ids + [0] * (self.max_len - len(label_ids))).unsqueeze(0).float()
        inputs["texts_ids"] = label_tensor
        inputs["texts"] = self.texts[idx]
        inputs["image_paths"] = self.image_paths[idx]
        
        return inputs

out_dir = f'{workdir}/TR-OCR-drafrica-result'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

ann_jsonl = f"{img_source}/textrecog_test.json"

data_dict = []
with open(ann_jsonl, 'r') as json_file:
    line = json.load(json_file)
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
test_data_dict = data_dict

image_paths = [id['img'] for id in test_data_dict]
texts = [id['text'] for id in test_data_dict]
word_len = [len(id['text'].split()) for id in test_data_dict]

test_dataset = MyTrainDataset(image_paths, texts, processor, 190)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

import pickle as pkl
from rouge_score import rouge_scorer
import jiwer
import pandas as pd
import Levenshtein

def calculate_wer_and_f1score(ground_truth, predicted):
    wer = jiwer.wer(ground_truth, predicted)
    
    total_words = len(ground_truth.split())
    pred_words = len(predicted.split())
    
    edit_distance = int(wer * total_words)
    overlap = total_words - edit_distance
    
    if overlap>0:
        precision_ocr = overlap/pred_words
        recall_ocr = overlap/total_words
        f1score_ocr= 2*precision_ocr*recall_ocr/(precision_ocr+recall_ocr)
    else:
        f1score_ocr = 0
    return wer, f1score_ocr, overlap

def calculate_cer_and_f1score(ground_truth, predicted):
    # Calculate the edit distance (Levenshtein Distance)
    edit_distance = Levenshtein.distance(ground_truth, predicted)
    
    # Calculate total characters in the ground truth
    total_chars = len(ground_truth)
    
    # Calculate CER
    cer = edit_distance / total_chars
    
    # Calculate overlapping characters
    overlap = total_chars - edit_distance
    
    if overlap>0:
        precision_ocr = overlap/len(predicted)
        recall_ocr = overlap/len(ground_truth)
        f1score_ocr= 2*precision_ocr*recall_ocr/(precision_ocr+recall_ocr)
    else:
        f1score_ocr = 0
    
    return cer, f1score_ocr, edit_distance

# Specify the path to the checkpoint file
pretrained_path = f'{workdir}/TrOCR-GloSAT-DRAfrica-without-augmentation/' #TrOCR-GloSAT-DRAfrica-without-augmentation/'
checkpoints = [pth for pth in os.listdir(pretrained_path) if '.pth' in pth]

macro_scores = []

for pth in checkpoints:
    checkpoint_path = f"{pretrained_path}/{pth}"
    # checkpoint_path = 'combined_dataset_checkpoint_epoch_0.pth'
    print(f'Loading checkpoint {checkpoint_path}')
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Remove 'module.' prefix from keys if loading on a single GPU
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = value

        # Load the new_state_dict into the model
        model.load_state_dict(new_state_dict)

        # Load the model's state dictionary from the checkpoint
        model.to(device)

        epoch = checkpoint['epoch']

        label = ['img_path', 'gt_text','pred_text']
        outputs = []

        ground_truths = []
        predictions = []
        
        ground_truths = []
        predictions = []
        output_s = []
        
        for batch_num, batch in enumerate(test_dataloader):
            model.eval()
            print(batch_num, len(test_dataloader))
            images = batch["pixel_values"].to(device)
            labels = batch["texts_ids"].squeeze(1).long().to(device)
        
            generated_ids = model.generate(images)
            generated_tokens = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for image_path, gt_text, pred_text in zip(batch['image_paths'], batch['texts'],generated_tokens):
                ground_truths.append(gt_text)
                predictions.append(pred_text)
                output_s.append((image_path, gt_text, pred_text))  
                
            del images, labels, generated_ids, generated_tokens

        f = open(f'{out_dir}/trOCR-epoch-{epoch}-drafrica-output.pkl', 'wb')
        pkl.dump(output_s,f)
        f.close()
        
        print(f'Saved output at {out_dir}/trOCR-epoch-{epoch}-drafrica-output.pkl')
            
        # Initialize the ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Calculate ROUGE scores for each pair of texts
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        # Initialize lists to hold WER and CER values
        wer_scores = []
        cer_scores = []
        f1_char_scores = []
        f1_token_scores = []
        
        matched = 0
        for gt, pred in zip(ground_truths, predictions):
            
            gt = gt.replace(' ','').replace('\t','').replace('·','.').lower()
            pred = pred.replace(' ','').replace('\t','').replace('·','.').lower()
            
            gt = re.sub(r'\.{2,}', '.', gt)
            pred = re.sub(r'\.{2,}', '.', pred)
            
            scores = scorer.score(gt, pred)
            rouge_1_scores.append(scores['rouge1'].fmeasure)
            rouge_2_scores.append(scores['rouge2'].fmeasure)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
            
            # Calculate CER
            cer, f1score_char, edit_distance = calculate_cer_and_f1score(gt, pred)
            cer_scores.append(cer)
            f1_char_scores.append(f1score_char)
                        
            if edit_distance > 1 or len(gt)<3:
            # if cer>0.01:
                # Calculate WER
                wer, f1score, overlap_words = calculate_wer_and_f1score(gt, pred)
            else:
                matched+=1
                wer = 0
                f1score = 1
                
            f1_token_scores.append(f1score)
            wer_scores.append(wer)
            print(f"Ground-truth: {gt}, Predicted Text: {pred}, CER: {cer:.4f}, WER: {wer:.4f}, Edit-distance:{edit_distance}")#

        # Create a DataFrame to display the results
        results_df = pd.DataFrame({
            'Ground Truth': ground_truths,
            'Prediction': predictions,
            'ROUGE-1': rouge_1_scores,
            'ROUGE-2': rouge_2_scores,
            'ROUGE-L': rouge_l_scores,
            'WER': wer_scores,
            'CER': cer_scores,
            'F1-scores-characters': f1_char_scores,
            'F1-scores-tokens': f1_token_scores
        })

        # print(results_df)
        results_df.to_csv(f'{out_dir}/trOCR-epoch-{epoch}-drafrica-scores.csv', index=False)

        # Calculate average scores
        average_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
        average_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
        average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
        average_wer_scores = sum(wer_scores) / len(wer_scores)
        average_cer_scores = sum(cer_scores) / len(cer_scores)
        average_f1_chars_scores = sum(f1_char_scores) / len(f1_char_scores)
        average_f1_tokens_scores = sum(f1_token_scores) / len(f1_token_scores)

        exact_matched = matched/len(ground_truths)    

        print(f"\nEpoch: {epoch} Average scores: ROUGE-L: {average_rouge_l:.4f} WER-score: {average_wer_scores:.4f} CER-score: {average_cer_scores:.4f}, F1_tokens_scores: {average_f1_tokens_scores:.4f} F1_chars_scores: {average_f1_chars_scores:.4f}")
        macro_scores.append((epoch,average_rouge_1, average_rouge_2, average_rouge_l, average_wer_scores, average_cer_scores, exact_matched, average_f1_chars_scores, average_f1_tokens_scores))
        
        del ground_truths, predictions, average_rouge_1, average_rouge_2, average_rouge_l, average_wer_scores, average_cer_scores, exact_matched, average_f1_chars_scores, average_f1_tokens_scores
    except:
        pass
#---------loop ends---------
result_label = ['Epoch', 'Rogue-1', 'Rogue-2', 'Rogue-L', 'WER', 'CER', 'EM', 'F1_chars_scores', 'F1_tokens_scores']
df = pd.DataFrame(macro_scores, columns=result_label)
df.to_csv(f'{out_dir}/trOCR-drafrica-macro-scores-new.csv', index=False)
print(df)