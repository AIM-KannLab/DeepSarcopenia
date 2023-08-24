# DeepSarcopenia: Automated Deep Learning Platform for Sarcopenia Assessment in Head and Neck Cancer
[![Build Status](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/actions/workflows/build.yml/badge.svg)](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/actions/workflows/build.yml)
[![Python Version](https://img.shields.io/pypi/pyversions/foundation-cancer-image-biomarker.svg)](https://pypi.org/project/foundation-cancer-image-biomarker/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] 
[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

Keywords: Sarcopenia, Head and Neck Cancer, Deep Learning, Convolutional Neural Network, CT

Keras implementation with codes and pretrained models for the article "[Development and Validation of an Automated Image-Based Deep Learning Platform for Sarcopenia Assessment in Head and Neck Cancer](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2808141)" published in JAMA Network Open 2023;6(8),e2328280. 

<p align="center"><img src="https://github.com/AIM-KannLab/DeepSarcopenia/blob/main/figures/figure.png" align="middle" width="800" title="figure" /></p>
Estimated segmentations with varying Dice similarity coefficient (DSC) values (greater than, equal to, or less than the median DSC) were overlaid on axial CT images to show their similarities to ground truth segmentations for validation and internal test sets. Corresponding difference (Δh) between the estimated C3 section slice and ground truth slice and DSC values are annotated for each case in the validation and internal test sets. Model-estimated C3 slices on sagittal CT images and segmentation on axial CT images are also shown for the external test set. Acceptability scores from expert clinicians’ review are annotated for corresponding cases in the external test set.

## Implemented models

 - DenseNet-163
 - 3D U-Net

## Results
U-Net Segmentation Model Performance
<p align="center"><img src="https://github.com/AIM-KannLab/DeepSarcopenia/blob/main/figures/table.png" align="middle" width="800" title="figure" /></p>

## Requirements

* Python 3.8
* TensorFlow 2.4

## Set-up

This code was developed and tested using Python 3.8.5.

For the code to run as intended, all the packages under requirements.txt should be installed. In order not to break previous installations and ensure full compatibility, it's highly recommended to create a virtual environment to run the DeepContrast pipeline in. Here follows an example of set-up using python virtualenv:

* install python's virtualenv
```
sudo pip install virtualenv
```
* parse the path to the python3 interpreter
```
export PY2PATH=$(which python3)
```
* create a virtualenv with such python3 interpreter named "venv"
(common name, already found in .gitignore)
```
virtualenv -p $PY2PATH venv 
```
* activate the virtualenv
```
source venv/bin/activate
```
At this point, (venv) should be displayed at the start of each bash line. Furthermore, the command which python3 should return a path similar to /path/to/folder/venv/bin/python3. Once the virtual environment is activated:

* once the virtualenv is activated, install the dependencies
```
pip install -r requirements.txt
```
At this stage, everything should be ready for the data to be processed by the DeepContrast pipeline. Additional details can be found in the markdown file under src.

The virtual environment can be deactivated by running:
```
deactivate
```

## Model Inference

The codebase for auto-segmenting the C3 skeletal muscle mass and for calculating muscle cross-sectional area (CSA), and Skeletal Mass Index (SMI) from 3D CT images.

Consists of two deep-learning models: Slice Selection Model and Segmentation Model. The Slice Selection Model is adapted from DenseNet and the Segmentation model is adapted from UNet.


### Trained Model Weights 

Model Weights are available in the "model/test" folder.


### Before You Start

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
 

### Steps to Run DeepSarcopenia Model and Calculate C3 SMI

    
#### Step 1: Main script
    
Main script executes pre-processing, slice selction and segmentation sequentially

STEP argument can be used to execute each of the above steps individually. The arguments takes values ALL, SLICE, PREPROCESS, SEGMENT
  
Other scripts provided here OPTIONAL, used to generate the needed CSA information, generates statistics, visualizations and final clinical file needed combining patient meta information with image segmenation data 

The Main script step can be run by executing:
```
main.py
```

#### Step 2: Calculate C3 Cross Sectional Area

This script generates the Cross-Sectional Area (CSA) of the C3 Skeletal Muscle Mass from the auto-segmented masks.

Set the proj_dir, raw_img_dir, raw_seg_dir, Slice Prediction CSV, and output_dir directories before executing the code. 
	
   1. Input#1: Auto segmentation files
   2. Input#2: Manual segmentation files of test data
   3. Input#3: C3_Top_Slice_Prediction.csv
   4. Output: CSA in a CSV file

The Calculate C3 Cross Sectional Area step can be run by executing:
```
get_CSA.py
```

#### Step 3: Visualize C3 Segmentation

This script generates the masks with contours for easy visualization.

Set the proj_dir, raw_img_dir, raw_seg_dir, Slice Prediction CSV, and output_dir directories before executing the code. 

   1. Input#1: Auto segmentation files
   2. Input#3: C3_Top_Slice_Prediction.csv
   3. Output: Segmentation files with contours for visualizing the masks

The Visualize C3 Segmentation function can be run by executing:
```
visualize.py
```

#### Step 4: Calculate C3 Skeletal Muscle Index (SMI)

This script generates the C3 Skeletal Muscle Index (SMI) and translates that to the corresponding L3 Skeletal Muscle Index (SMI).

Set the proj_dir, raw_img_dir, raw_seg_dir, Slice Prediction CSV, Clinical Meta-data files and output_dir directories before executing the code. 

   1. Input#1: Auto segmentation files
   2. Input#2: Cross-Sectional Area CSV file
   3. Input#3: C3_Top_Slice_Prediction.csv
   4. Input#4: Patient Clinical meta information in a CSV file
   5. Output: SMI data in a CSV file

The Calculate C3 Skeletal Muscle Index (SMI) step can be run by executing:
```
clinical.py
```

## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@article{Ye2023DeepSarcopenia,
  title = {Development and Validation of an Automated Image-Based Deep Learning Platform for Sarcopenia Assessment in Head and Neck Cancer}, 
  author = {Z Ye, A Saraf, Y Ravipati, F Hoebers, P Catalano, Y Zha, A Zapaishchykova, J Likitlersuang, C Guthier, R Tishler, J Schoenfeld, D Margalit, R Haddad, R Mak, M Naser, K Wahid, J Sahlsten, J Jaskari, K Kaski, A Mäkitie, C Fuller, H Aerts, and B Kann},
  journal = {JAMA Network Open 6 (8), e2328280},
  year = {2023}, 
}
```

## Acknowledgements

Code development, testing, refactoring and documentation: YR and ZY.

## Disclaimer
The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. 

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
