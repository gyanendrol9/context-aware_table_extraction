import os
source = '/data/glosat/active_learning-2/dla_models/model_table_struct_fine_train/VOC2007'
des_dir = '/data/glosat/active_learning-2/dla_models/model_table_struct_fine_aclr_0/VOC2007/'

#List of files to move
f = open('/data/glosat/active_learning-2/dla_models/model_table_struct_fine_train/VOC2007/ImageSets/main.txt','r')
files = f.readlines()
f.close()

if not os.path.exists(f'{des_dir}/Annotations'):
    os.mkdir(f'{des_dir}/Annotations')
    
if not os.path.exists(f'{des_dir}/ICDAR'):
    os.mkdir(f'{des_dir}/ICDAR')

if not os.path.exists(f'{des_dir}/Transkribus'):
    os.mkdir(f'{des_dir}/Transkribus')

if not os.path.exists(f'{des_dir}/JPEGImages'):
    os.mkdir(f'{des_dir}/JPEGImages')

for file in files:
    file = file.strip()
    # os.system(f'cp {source}/ICDAR/{file}.xml {des_dir}/ICDAR/{file}.xml')
    # os.system(f'cp {source}/Transkribus/{file}.xml {des_dir}/Transkribus/{file}.xml')
    os.system(f'cp {source}/Annotations/{file}.xml {des_dir}/Annotations/{file}.xml')
    os.system(f'cp {source}/JPEGImages/{file}.jpg {des_dir}/JPEGImages/{file}.jpg')
    



