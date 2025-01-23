# UoS Data Rescue: A Framework for Digitizing Historical Tabular Records  

[![License](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview  
Digitizing historical tabular records is essential for preserving and analyzing valuable data across various domains. This repository contains the source code, dataset, and pre-trained models introduced in the paper:  

**Title:** "Tabular Context-aware Optical Character Recognition and Tabular Data Reconstruction for Historical Records"  

**Abstract:**  
> Digitizing historical tabular records is essential for preserving and analyzing valuable data across various fields, but it presents challenges due to complex layouts, mixed text types, and degraded document quality. This paper introduces a comprehensive framework to address these issues through three key contributions:  
> - **UoS_Data_Rescue Dataset:** A novel dataset of 1,113 historical logbooks with 594,000 annotated text cells, tackling challenges like handwritten entries, aging artifacts, and intricate layouts.  
> - **TrOCR-ctx:** A novel context-aware text extraction approach to reduce cascading errors during table digitization.  
> - **Enhanced End-to-End OCR Pipeline:** Integrates TrOCR-ctx with ByT5 for real-time post-OCR correction, improving multilingual support and achieving state-of-the-art performance.  

The framework offers a robust solution for large-scale digitization of tabular documents, extending applications beyond climate records to other domains requiring structured document preservation.  

## Features  
- **Dataset:** UoS_Data_Rescue, a rich collection of historical tabular data.  
- **Models:** Pre-trained TrOCR-ctx and ByT5 for OCR tasks.  
- **Pipeline:** End-to-end OCR processing with real-time post-OCR correction.  

## Setup Instructions  

### Prerequisites  
Ensure you have the following installed on your system:  
- Python 3.8+  
- Git  
- CUDA-enabled GPU (optional, for model training/inference)  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-repo-name/your-project-name.git  
   cd your-project-name  '''

2. Create a virtual environment and activate it:  
   ```bash  
   python -m venv env  
   source env/bin/activate   # On Windows: env\Scripts\activate  '''



3. Install dependencies:
    ```bash 
    pip install -r requirements.txt  
    '''

4. Dataset Setup

The dataset is hosted on Zenodo. Download the dataset and extract it to the data/ directory:

    [UoS_Data_Rescue Dataset](https://ceur-ws.org/Vol-3649/Paper1.pdf)


Evaluate the Model

Evaluate the model performance on a test set:

python scripts/evaluate.py --model_dir ./models --test_data ./processed_data/test  

4. Perform Inference

Digitize new tabular records:

python scripts/inference.py --model_dir ./models --input ./samples/input_image.jpg --output ./output/  


You can copy and paste this directly into your `README.md`. Let me know if there’s anything else!