# C3-Segmentation

The codebase for auto-segmenting the C3 skeletal muscle mass and for calculating muscle cross-sectional area (CSA), and Skeletal Mass Index (SMI) from 3D CT images.

Consists of two deep-learning models: Slice Selection Model and Segmentation Model. The Slice Selection Model is adapted from DenseNet and the Segmentation model is adapted from UNet.


### Trained Model Weights 

Model Weights are available for download at the following link.

https://drive.google.com/drive/folders/1A3NlgyvlhXL6pgR0weXT4c-XygGl6r-M?usp=drive_link

### Steps before you execute the Scripts listed 

1. Download the model weights from the google drive link provided above, unzip the files, and save them in the 'model' subfolder. Please note that the CV folder within the zip file contains five-fold cross-validation sub-models for segmentation.

2. Make sure you set the env properities (pass as command line arguments to main.py script)
    <proj_dir>  - Project Directories Path where this code is copied
    <dataset> - instuition name of curated dataset e.g. BWH , default value is "test"

3. The input files are stored in the following folder

   Raw scans - /data/<dataset>/img/

4. Following are default directories/paths that get created to stroe the pre-processed files, slice selection output as segmentation output. If you prefer, you can also set them using env variables <pre_process_dir>, <slice_csv_path> and <output_dir>. Passing them to main.py as command line arguments is also accepted.   
   
   Pre-processed Scans - /data/<dataset>/preprocessed
   Slice Selection Output CSV - /data/<dataset>/<data-set>_C3_top_slice_pre.csv
   Segmentation Model Output - /data/<dataset>/output_segmentations 
     

#####Script Execution
    
1. Main script: 
    
    Main script executes pre-processing, slice selction and segmentation sequentially
    STEP argument can be used to execute each of the above steps individually. The arguments takes values ALL, SLICE, PREPROCESS, SEGMENT
  
Other scripts provided here OPTIONAL, used to generate the needed CSA information, generates statistics, visualizations and final clinical file needed combining patient meta information with image segmenation data 

2. 'get_CSA.py'

  	Set the proj_dir, raw_img_dir, raw_seg_dir, Slice Prediction CSV, and output_dir directories before executing the code. 
	This script generates the Cross-Sectional Area (CSA) of the C3 Skeletal Muscle Mass from the auto-segmented masks.
   	- Input#1: Auto segmentation files
 	- Input#2: Manual segmentation files of test data
 	- Input#3: C3_Top_Slice_Prediction.csv
 	- Output: CSA in a CSV file


3. 'visualize.py'

	Set the proj_dir, raw_img_dir, raw_seg_dir, Slice Prediction CSV, and output_dir directories before executing the code. 
	This script generates the masks with contours for easy visualization. 
   	- Input#1: Auto segmentation files
 	- Input#3: C3_Top_Slice_Prediction.csv
 	- Output: Segmentation files with contours for visualizing the masks

4. 'clinical.py'

	Set the proj_dir, raw_img_dir, raw_seg_dir, Slice Prediction CSV, Clinical Meta-data files and output_dir directories before executing the code. 
	This script generates the C3 Skeletal Muscle Index (SMI) and translates that to the corresponding L3 Skeletal Muscle Index (SMI).
   	- Input#1: Auto segmentation files
 	- Input#2: Cross-Sectional Area CSV file
 	- Input#3: C3_Top_Slice_Prediction.csv
 	- Input#4: Patient Clinical meta information in a CSV file
	- Output: SMI data in a CSV file
