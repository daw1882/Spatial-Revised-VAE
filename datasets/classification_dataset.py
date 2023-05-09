"""
Generate a classification dataset with encoded pixel vectors from a spectral
image.

Author: Dade Wood
CSCI 736
"""

import argparse
import csv
import json

import numpy as np
import pandas as pd
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectral_dataset import SpectralImage, SpectralVAEDataset
from spectral_spatial_vae.spectral_vae import SpatialRevisedVAE


def make_classification_dataset(model: SpatialRevisedVAE, dataloader, labels):
    """
    Create a dataset for classification of spectral pixel vectors.

    Parameters
    ----------
    model
        The VAE model for extracting features.
    dataloader
        The pytorch dataloader containing spectral pixel information.
    labels
        Labels for each pixel vector.

    Returns
    -------
    A DataFrame containing the encoded pixel vectors and labels.
    """
    with open('output.csv', 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        with tqdm(dataloader, unit="batch") as loader:
            for batch in loader:
                encoded = model.encoder(batch)
                writer.writerows(encoded.tolist())
    df = pd.read_csv('output.csv', header=None)
    df['labels'] = labels
    df.to_csv('output.csv', header=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='Path to the model to load.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--config',
        help='Configuration file containing model parameters.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image',
        help='Path to hyperspectral image directory.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--labels',
        help='Path to hyperspectral image directory.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--output_dir',
        help='Path to output the dataset to.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--data_key',
        help='Key for accessing the data from a .mat file.',
        type=str,
    )
    parser.add_argument(
        '--data_type',
        help='Form that the hyperspectral image is stored in.',
        type=str,
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    model_config = config["model"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = SpectralImage(args.image, args.data_type, args.data_key)
    dataset = SpectralVAEDataset(img, model_config["window_size"],
                                 device=device)
    loader = DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=0,
    )
    gt = np.transpose(
        scipy.io.loadmat(args.labels)[args.data_key + "_gt"]
    )
    r, c = gt.shape
    gt = gt[0:r-model_config["window_size"], 0:c-model_config[
        "window_size"]].flatten()

    vae = SpatialRevisedVAE(
        s=model_config["window_size"],
        ld=model_config["latent_dims"],
        spectral_bands=img.spectral_bands,
        spec_layers=model_config["ae_layers"],  # Encoder
        ss_layers=model_config["ss_layers"],  # LSTM
        ls_layers=model_config["ls_layers"],  # CNN
        device=device,
    ).to(device)
    vae.load_state_dict(torch.load(args.model))
    vae.eval()
    print("Model loaded!")

    make_classification_dataset(vae, loader, gt)
