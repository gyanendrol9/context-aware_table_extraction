# A Framework for Digitizing Historical Tabular Records  

[![License](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


## Overview  
Digitizing historical tabular records is essential for preserving and analyzing valuable data across various domains. This repository contains the source code, dataset, and pre-trained models introduced in the paper:  

**Title:** "Tabular Context-aware Optical Character Recognition and Tabular Data Reconstruction for Historical Records"  [Paper link](https://link.springer.com/article/10.1007/s10032-025-00543-9)

**Abstract:**  
> Digitizing historical tabular records is essential for preserving and analyzing valuable data across various fields, but it presents challenges due to complex layouts, mixed text types, and degraded document quality. This paper introduces a comprehensive framework to address these issues through three key contributions:  
> - **UoS_Data_Rescue Dataset:** A novel dataset of 1,113 historical logbooks with 594,000 annotated text cells, tackling challenges like handwritten entries, aging artifacts, and intricate layouts.  
> - **TrOCR-ctx:** A novel context-aware text extraction approach based on [TrOCR](https://huggingface.co/docs/transformers/en/model_doc/trocr) to reduce cascading errors during table digitization.  
> - **Enhanced End-to-End OCR Pipeline:** Integrates TrOCR-ctx with ByT5 for real-time post-OCR correction, improving multilingual support and achieving state-of-the-art performance.  

The framework offers a robust solution for large-scale digitization of tabular documents, extending applications beyond climate records to other domains requiring structured document preservation.  

## Pipeline for Digitizing Historical Tabular Data  
This framework involves three main modules to digitize tabular data effectively:  

1. **Table Structure Recognition (TSR):**  
    TSR is the process of identifying and reconstructing the layout of a table, including detecting table boundaries, cell boundaries, and the relationships between rows and columns. This step is crucial to preserve the structural integrity of the tabular data for accurate digitization.

    - **Model Used:** [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet), a state-of-the-art model for table structure recognition.  
    - **Prerequisites:**  
        > - Python 3.7  
        > - Dependencies specific to CascadeTabNet (e.g., Torch, MMCV, etc.).  
            - Detailed installation, configuration, and training instructions for the TSR module can be found [here](https://github.com/stuartemiddleton/glosat_table_dataset).  
        > - YAML file for setting up the environment is provided [here](environment/open-mmlab.yml). Update the environment using the provided YAML file:
        ```bash  
        conda env update --name open-mmlab --file environment/open-mmlab.yml
        ```

2. **Text Extraction:**  
    This step extracts textual content from the cells identified by the TSR module.  
    - **Model Used:** TrOCR-ctx ([TrOCR](https://huggingface.co/docs/transformers/en/model_doc/trocr) with contextual embedding).  
    - **Key Features:**  
        > - Context-aware text extraction to reduce cascading errors.  
        > - Handles challenges like handwritten entries, degraded text, and mixed languages.  
    - **Prerequisites:**  
        > - Python 3.8+  
        > - Libraries such as `torch`, `transformers`, and `datasets`.  
        > - YAML file for setting up the environment is provided [here](environment/ocrenv.yml).          

3. **Tabular Data Reconstruction:**  
   After text extraction, this module aligns the textual data with the recognized table structure to generate a final digital table.  
   - **Post-OCR Correction:** Uses ByT5 for real-time corrections, improving character and word error rates.  
   - **Output:** Fully digitized and reconstructed tables in a structured format like CSV or JSON.  

## Features  
- **Dataset:** UoS_Data_Rescue, a rich collection of historical tabular data.  
- **Models:** Pre-trained TrOCR-ctx and [ByT5](https://huggingface.co/yelpfeast/byt5-base-english-ocr-correction) for OCR tasks.  
- **Pipeline:** End-to-end OCR processing with real-time post-OCR correction.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/gyanendrol9/table_extraction.git  
   cd table_extraction  
   ```

2. Create a virtual environment and activate it:  
   ```bash  
    # create ocrenv env in conda
    conda create --name ocrenv python=3.8
    ```

3. Install dependencies:
    ```bash 
    pip install -r requirements.txt  
    
    # You can also update the environment using the provided YAML file [Optional]
    conda env update --name ocrenv --file environment/ocrenv.yml # [Optional]
    ```

4. Dataset Setup:
    The dataset is hosted on Zenodo. Download the dataset and extract it to the data directory:
    [UoS_Data_Rescue Dataset](https://www.researchsquare.com/article/rs-5462018/v1)
    ```bash 
    mkdir UoS_Data_Rescue

    #Store the dataset and model checkpoints inside this folder
    ```

    - Dataset Structure
    For training the OCR model, the dataset must be structured in the following JSON format. You can format your custom dataset accordingly, ensuring each entry includes the following fields:  
        > **`img_path`**: A string representing the file path to the image.

        > **`text`**: A string representing the text associated with the image.

    - JSON Structure Example
        ```json
        [
            {
                "img_path": "path/to/image1.jpg",
                "text": "This is an example text associated with the image1.jpg."
            }
            {
                "img_path": "path/to/image2.jpg",
                "text": "This is an example text associated with the image2.jpg."
            }
        ]
        ```

5. Train the Model  
This framework involves three main components: 
    - **Table Structure Recognition (TSR)**, 
    - **Text Extraction (TrOCR-ctx)**, and 
    - **Tabular Data Reconstruction**. 
    
    Each step is trained separately to ensure high performance across the pipeline.

- Pipeline Integration  
    > Step 1: Train the TSR Model  
    Table Structure Recognition (TSR) identifies and reconstructs the table layout, including table boundaries, cell boundaries, and relationships between rows and columns.  
    - To train the TSR model, **complete all the steps provided** in the [glosat_table_dataset](https://github.com/stuartemiddleton/glosat_table_dataset) repository. 

    > Step 2: Train the Text Extraction Model (TrOCR-ctx)  
    TrOCR-ctx (Transformer-based OCR with context-aware embeddings) extracts text from cells identified by the TSR module. This step reduces cascading errors and improves text recognition accuracy, especially for handwritten or degraded text. The [TrOCR](https://huggingface.co/docs/transformers/en/model_doc/trocr) and [ByT5](https://huggingface.co/yelpfeast/byt5-base-english-ocr-correction) models are trained jointly, with losses from both models contributing to the overall training objective. 
    - **Training Instructions on Ubuntu 20.04 LTS:**  
        ```bash
        conda activate ocr_env   
        python train-trocr-ctx.py <dataset_dir> <output_checkpoint_dir>
        
        # Examples
        python train-trocr.py UoS_Data_Rescue UoS_Data_Rescue/TR-OCR-checkpoint
        python train-trocr-ctx.py UoS_Data_Rescue UoS_Data_Rescue/TR-OCR-ctx-checkpoint
        ```

    - **Training Instructions on IRIDIS 5/IRIDIS X:**  
        ```bash
        sbatch run_sbatch_ocr.sh train-trocr-ctx.py <dataset_dir> <output_checkpoint_dir>
        
        # Examples 
        sbatch run_sbatch_ocr.sh train-trocr-ctx.py UoS_Data_Rescue UoS_Data_Rescue/TR-OCR-ctx-checkpoint
        sbatch run_sbatch_ocr.sh train-trocr.py UoS_Data_Rescue UoS_Data_Rescue/TR-OCR-checkpoint
        ```
    - **Text extraction:**  
        ```bash
        python text_extraction-folder_v2.py <image_source_dir> <output_dir> <tr_ocr_checkpoint>

        # Example
        ls UoS_Data_Rescue/TR-OCR-ctx-checkpoint
        
        python text_extraction-folder_v2.py UoS_Data_Rescue/Images UoS_Data_Rescue/Tabular_Data_reconstruction UoS_Data_Rescue/TR-OCR-ctx-checkpoint/best_epoch.pth
        ```

    > Step 3: Heuristic Approach to Tabular Data Reconstruction  
    - The final step in the pipeline involves reconstructing the tabular data by aligning the text extracted by the TrOCR-ctx model with the table structure detected by the TSR module. This process ensures that the reconstructed data preserves the original table's layout and logical relationships.
    - A heuristic-based approach is used for this alignment, leveraging the coordinates of table cells and the extracted text. The reconstructed tabular data is then output in structured formats such as CSV or JSON.
    - To execute the reconstruction step, run the following command:
        ```bash 
        python reconstruction_v3-folder.py <image_source_dir> <output_dir>
        ```  

    Once trained, these components can be seamlessly integrated to provide an end-to-end solution for digitizing historical tabular data.

6. Evaluate the TrOCR model performance on the test set:
    ```bash 
    python testing-finetuned-TrOCR.py <dataset_dir> <output_dir> <tr_ocr_checkpoint_dir>

    # Example
    python testing-finetuned-TrOCR.py UoS_Data_Rescue UoS_Data_Rescue/TrOCR_evaluation UoS_Data_Rescue/TR-OCR-ctx-checkpoint
    ```

7. To digitize new tabular records, the pipeline combines the three modules into a single workflow. The pipeline takes a folder of input images and processes them to generate structured tabular data in a desire folder. Use the provided script to process the images and generate structured output.
    ```bash        
    # Change the python environment path in the bash file before running the following command.
    bash text-extraction-pipeline-folder-input.sh <image_source_dir> <output_dir> <tr_ocr_checkpoint>
    
    # Example
    bash text-extraction-pipeline-folder-input.sh UoS_Data_Rescue/Images UoS_Data_Rescue/Tabular_Data_reconstruction UoS_Data_Rescue/TR-OCR-ctx-checkpoint/best_epoch.pth

    ```

### Results:
- Word Error Rate (WER): 0.049
- Character Error Rate (CER): 0.035
- Improvement: Up to 41% in OCR tasks and 10.74% in table reconstruction tasks compared to existing methods.

### Acknowledgments:
This work is funded through the Natural Environment Research Council (grant [NE/S015604/1](https://gtr.ukri.org/projects?ref=NE%2FS015604%2F1)) and WCSSP South Africa project, a collaborative initiative between the Met Office, South African, and UK partners, supported by the International Science Partnership Fund (ISPF) from the UK's Department for Science, Innovation and Technology (DSIT).  It is also supported by the Centre for Machine Intelligence ([CMI](https://www.southampton.ac.uk/research/institutes-centres/centre-for-machine-intelligence)) and Web Science Institute ([WSI](https://www.southampton.ac.uk/research/projects/multimodal-information-extraction-of-embedded-text-in-online-images)). The authors acknowledge the [IRIDIS High-Performance Computing Facility](https://www.southampton.ac.uk/research/facilities/iridis-research-computing-facility) at the University of Southampton.

### Project links:
- [GloSAT](https://glosat.org/)
- [DR-Africa](https://www.southampton.ac.uk/~sem03/DR-Africa.html)

### Citation:
Singh, L.G., Middleton, S.E. Tabular context-aware optical character recognition and tabular data reconstruction for historical records. IJDAR (2025). https://doi.org/10.1007/s10032-025-00543-9