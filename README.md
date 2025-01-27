# A Framework for Digitizing Historical Tabular Records  

[![License](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


## Overview  
Digitizing historical tabular records is essential for preserving and analyzing valuable data across various domains. This repository contains the source code, dataset, and pre-trained models introduced in the paper:  

**Title:** "Digitizing Historical Tabular Records with a Comprehensive OCR Framework"  
**Abstract:**  
> Digitizing historical tabular records is essential for preserving and analyzing valuable data across various fields, but it presents challenges due to complex layouts, mixed text types, and degraded document quality. This paper introduces a comprehensive framework to address these issues through three key contributions:  
> - **UoS_Data_Rescue Dataset:** A novel dataset of 1,113 historical logbooks with 594,000 annotated text cells, tackling challenges like handwritten entries, aging artifacts, and intricate layouts.  
> - **TrOCR-ctx:** A novel context-aware text extraction approach to reduce cascading errors during table digitization.  
> - **Enhanced End-to-End OCR Pipeline:** Integrates TrOCR-ctx with ByT5 for real-time post-OCR correction, improving multilingual support and achieving state-of-the-art performance.  

The framework offers a robust solution for large-scale digitization of tabular documents, extending applications beyond climate records to other domains requiring structured document preservation.  

## Pipeline for Digitizing Historical Tabular Data  
This framework involves three main modules to digitize tabular data effectively:  

1. **Table Structure Recognition (TSR):**  
   TSR is the process of identifying and reconstructing the layout of a table, including detecting table boundaries, cell boundaries, and the relationships between rows and columns. This step is crucial to preserve the structural integrity of the tabular data for accurate digitization.
   
   The TSR module in this framework is implemented using CascadeTabNet, a state-of-the-art model for table structure recognition. 
   > Details on the installation, configuration, and training of the TSR model using CascadeTabNet can be found [here](https://github.com/stuartemiddleton/glosat_table_dataset).  

2. **Text Extraction:**  
   This step extracts textual content from the cells identified by the TSR module.  
   - **Model:** TrOCR-ctx (Transformer-based OCR with contextual embedding).  
   - **Key Features:**  
     - Context-aware text extraction to reduce cascading errors.  
     - Handles challenges like handwritten entries, degraded text, and mixed languages.  

3. **Tabular Data Reconstruction:**  
   After text extraction, this module aligns the textual data with the recognized table structure to generate a final digital table.  
   - **Post-OCR Correction:** Uses ByT5 for real-time corrections, improving character and word error rates.  
   - **Output:** Fully digitized and reconstructed tables in a structured format like CSV or JSON.  

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
   git clone https://github.com/gyanendrol9/table_extraction.git  
   cd table_extraction  
   ```

2. Create a virtual environment and activate it:  
   ```bash  
    # create ocr_env env in conda
    conda create --name ocr_env python=3.8
    conda activate ocr_env   
    ```

3. Install dependencies:
    ```bash 
    pip install -r requirements.txt  
    ```

4. Dataset Setup

The dataset is hosted on Zenodo. Download the dataset and extract it to the data/ directory:
[UoS_Data_Rescue Dataset](https://ceur-ws.org/Vol-3649/Paper1.pdf)


5. Evaluate the Model
Evaluate the model performance on a test set:
    ```bash 
    python scripts/evaluate.py --model_dir ./models --test_data ./processed_data/test  
    ```

6. Perform Inference
Digitize new tabular records:
    ```bash
    python scripts/inference.py --model_dir ./models --input ./samples/input_image.jpg --output ./output/  
    ```



### Results:
- Word Error Rate (WER): 0.049
- Character Error Rate (CER): 0.035
- Improvement: Up to 41% in OCR tasks and 10.74% in table reconstruction tasks compared to existing methods.


### Acknowledgments:
This work is funded through the Natural Environment Research Council (grant NE/S015604/1) and WCSSP South Africa project, a collaborative initiative between the Met Office, South African, and UK partners, supported by the International Science Partnership Fund (ISPF) from the UK's Department for Science, Innovation and Technology (DSIT).  It is also supported by the Centre for Machine Intelligence (CMI) and Web Science Institute (WSI). The authors acknowledge the IRIDIS High-Performance Computing Facility at the University of Southampton.

