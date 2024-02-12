import os

source = '/data/glosat/Appen/Annotation/cell_check/GlOSAT/all_files_aa_complete'
batch_folder = '/data/glosat/Appen/Annotation/cell_check/GlOSAT/all_files_aa_pilot'

if not os.path.exists(f"{batch_folder}"):
    os.mkdir(f"{batch_folder}")
    os.mkdir(f"{batch_folder}/JSON")
    os.mkdir(f"{batch_folder}/Image")
    os.mkdir(f"{batch_folder}/Segmented")

#List of files to move
f = open('/data/glosat/Appen/Annotation/cell_check/all_files_aa_pilot','r')
files = f.readlines()
f.close()

files = [line.strip() for line in files]

for filename in files:
    os.system(f'mv {source}/JSON/{filename}.json {batch_folder}/JSON/.')
    os.system(f'mv {source}/Image/{filename}.jpg {batch_folder}/Image/.')
    os.system(f'mv {source}/Segmented/{filename}.jpg {batch_folder}/Segmented/.')

# to separate remaining files
# json_files = os.listdir(f'{source}/JSON/')