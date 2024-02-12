'''
#Cells  days    Rate
280000  25.93   (8sec/cells)
(20*2000*7) 
'''

import os

dataset_type = 'GlOSAT'
files = os.listdir(dataset_type)

batch = 1
for file in files:
    batch_folder = f"{dataset_type}/{file}/JSON"
    json_files = os.listdir(batch_folder)
    batch_folder = f"{dataset_type}/{file}/Image"
    img_files = os.listdir(batch_folder)

    print(f'{file}\t#JSON: {len(json_files)}\t#Image: {len(img_files)}')



