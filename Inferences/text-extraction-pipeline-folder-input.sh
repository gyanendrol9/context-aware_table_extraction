#Code location
python_det_env=~/anaconda3/envs/open-mmlab/bin/python 
detection_code_loc=Inferences/TSR-inference.py
aclr=12

python_ocr_env=~/anaconda3/envs/ocrenv/bin/python
ocr_code_loc=Inferences/text_extraction-folder_v2.py

reconstruction_code_loc=Inferences/reconstruction_v3-folder.py

#Input output parameter
img_folder_path=$1
outdirectory=$2
$tr_ocr_checkpoint = $3

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
$python_ocr_env $ocr_code_loc $img_folder_path $outdirectory $tr_ocr_checkpoint
echo "************************** OCR Text extraction ended ************************** "
echo ''


echo "************************** Table reconstruction begin ************************** "
$python_det_env $reconstruction_code_loc $img_folder_path $outdirectory
echo "************************** Table reconstruction ended ************************** "
echo ''

