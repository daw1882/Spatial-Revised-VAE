# Spatial-Revised-VAE
This is a repository containing the model architecture from "Spatial Revising Variational Autoencoder-Based Feature Extraction Method for Hyperspectral Images" as well as scripts for training, generating encoded datasets, and running task heads. For any of the below scripts, use the `-h` or `--help` flag for more information on how to run.

## Installation
`pip -r requirements.txt`

## Training
Training of the model requires a directory containing hyperspectral image bands or a .mat file containing the spectral image data. Examples of configuration files used for training are in the `configs` folder.

For loading from a directory:

Usage: `python training.py --model_dir model_dir --config path_to_config --image path_to_image_data --data_type dir`

For loading from a .mat file: (specifically loading paviaU)

Usage: `python training.py --model_dir model_dir --config path_to_config --image path/PaviaU.mat --data_key paviaU --data_type mat`


## Encoded Dataset Creation

For creating a classification dataset from a .mat file:

Usage: `python datasets/latent_dim_dataset.py --model path_to_model_chkpt --config path_to_config --image path/PaviaU.mat --labels path/PaviaU_gt.mat --data_key paviaU --data_key_gt paviaU_gt --data_type mat --output_dir path/paviaU.csv --output_type classification`

For creating a clustering dataset from a directory:

Usage: `python datasets/latent_dim_dataset.py --model path_to_model_chkpt --config path_to_config --image path_to_img_dir --data_type dir --output_dir path/data.csv --output_type clustering`

## Reconstruction

To visualize the reconstruction from the VAE:

Usage: `--model path_to_model_chkpt --config path_to_config --image path/PaviaU.mat --data_key paviaU --data_type mat --image_width 599 --image_height 329`


## Classification

Perform classification task on an encoded dataset:

Usage: `python tasks/classification.py --csv path_to_csv_data --output path/paviaU.png --image_width 599 --image_height 329`


## Clustering

Perform clustering task on an encoded dataset:

Usage: `--csv path/data.csv --output path/output.png --image_width 111 --image_height 111`
