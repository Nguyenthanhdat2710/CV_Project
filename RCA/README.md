# RCA

This is a cloned version of the original repository https://github.com/maeve07/RCA with addition of the evaluation code on PASCAL VOC 2012 dataset and some modifications and fixes so that the code could run on the recent version of python and pip packages.
This repository also contains a saved model weight which was trained for 30 epochs using default parameters which was specified in the RCA research paper as well as its global attention maps for generating pseudo labels. The model can be downloaded from https://drive.google.com/file/d/1FPZyqEWpxkJlxVuDFiIgfSizhBSIC4bo/view and the attention maps can be downloaded at https://drive.google.com/file/d/1N7T6HhrEh_cN2wouFScDQNJjfJr0UyRh/view. For their usages, please refer to the following sections. This repo also contains the pre-generated pseudo labels using our trained models. These labels can be downloaded https://drive.google.com/file/d/1bVZuAdE3acF5O7b5WiZK4ieD1wORG_oH/view.

## Requirements
Install all required packages using pip:
```
pip install Cmake
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Download PASCAL VOC 2012 dataset
Download the dataset from link: https://drive.google.com/file/d/1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X/view and unzip dataset in the repo root directory.
After this, inside the RCA source directory, there will be 'VOCdevkit' dataset directory.

## Train RCA model
The first command is to train the model from scratch with default parameters and number of epochs.

By default, for every 3 epoch and for the last epoch, the code will save the current model and optimizer weight as a checkpoint in runs/pascal/model directory.

The second command can be used to resume training from a checkpoint given the checkpoint path.
```
python train.py
python train.py --resume ./pascal_voc_epoch_30.pth --epoch 50
```

## Evalute model
After training for every epoch, the code will save global attention maps under directory runs/pascal/feat.
Or for the attention maps of our trained model, it can be download from the link above. To use it, unzip it under runs/pascal.
Then, we need to resize the attention maps to theirs original size:
```
python res.py
```
To generate pseudo labels for the weakly supervised segmentation task, RCA need to use saliency map for the PASCAL VOC dataset, which can be download from here: https://drive.google.com/file/d/1ENS6jR6EUIDtxWsYwgwZ9YELSp0tFQ0k/view.
Extract this into the path: VOCdevkit/
Finally, we can generate the pseudo labels:
```
python gen_labels.py
```
The pseudo labels will be saved under path data/proxy_label.

To calculate the mIOU between our pseudo labels against the original PASCAL VOC labels, we can run the command:
```
python evaluate.py
```
