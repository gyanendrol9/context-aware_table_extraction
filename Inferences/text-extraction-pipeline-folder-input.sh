#Code location
python_det_env=/home/gyanendro/anaconda3/envs/open-mmlab/bin/python 
detection_code_loc=/home/gyanendro/Desktop/active_learning-2/TSR-inference-aclr.py

python_ocr_env=/home/gyanendro/anaconda3/envs/ocrenv/bin/python
ocr_code_loc=/home/gyanendro/Desktop/mm-ocr-update/Tabular-data-extraction/text_extraction-folder_v2.py

reconstruction_code_loc=/home/gyanendro/Desktop/mm-ocr-update/Tabular-data-extraction/reconstruction_v3-folder.py

#Input output parameter
img_folder_path=$1
aclr=12
outdirectory=$2

mkdir $outdirectory

echo "************************** Table structure recognition begin ************************** "
echo ''
files=$(ls $img_folder_path/*.jpg)
echo $files
for file in $files; do
    echo "TSR for image "$file
    echo ""
    $python_det_env $detection_code_loc $file $aclr $outdirectory
done

echo ''
echo "************************** Table structure recognition ended ************************** "
echo ''

echo "************************** OCR Text extraction begin ************************** "
$python_ocr_env $ocr_code_loc $img_folder_path $outdirectory
echo "************************** OCR Text extraction ended ************************** "
echo ''


echo "************************** Table reconstruction begin ************************** "
$python_det_env $reconstruction_code_loc $img_folder_path $outdirectory
echo "************************** Table reconstruction ended ************************** "
echo ''

