import os
files = os.listdir()

dataset_type = 'GlOSAT'
searchfor = 'all_files_'

json_source = 'JSON_with_20_cells_new'
image_source = 'Image_with_20_cells_original'

if not os.path.exists(f"{dataset_type}"):
    os.mkdir(f"{dataset_type}")

f = open('exclude_files','r')
lines = f.readlines()
f.close()

exclude_files = [line.strip() for line in lines]

batch = 1
for file in files:
    if searchfor in file:
        batch_folder = f"{dataset_type}/{file}"

        if not os.path.exists(f"{batch_folder}"):
            os.mkdir(f"{batch_folder}")
            os.mkdir(f"{batch_folder}/JSON")
            os.mkdir(f"{batch_folder}/Image")
            os.mkdir(f"{batch_folder}/Segmented")

        f = open(file,'r')
        lines = f.readlines()
        f.close()

        considered_files = [line.strip() for line in lines if line.strip() not in exclude_files]

        for filename in considered_files:
            os.system(f'cp {json_source}/{filename}.json {batch_folder}/JSON/.')
            os.system(f'cp {image_source}/{filename}.jpg {batch_folder}/Image/.')
            os.system(f'cp Image_with_20_cells/{filename}.jpg {batch_folder}/Segmented/.')

        batch+=1
