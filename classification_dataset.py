import scipy
import torch
from torch.utils.data import DataLoader
from spectral_dataset import SpectralImage, SpectralVAEDataset
import argparse
from spectral_vae import SpatialRevisedVAE
import json
from torchsummary import summary
from tqdm import tqdm
from sklearn import svm
import csv
import numpy as np
import pandas as pd


def make_classification_dataset(model: SpatialRevisedVAE, dataloader, labels):
    # Note, this is really for fancy display of the progress,
    # the tqdm is not necessary but the dataloader is for
    # performing inference in batches
    with open('output.csv', 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        with tqdm(dataloader, unit="batch") as loader:
            for batch in loader:
                # you'd call model.decoder(input) for decoding part
                encoded = model.encoder(batch)
                writer.writerows(encoded.tolist())
    df = pd.read_csv('output.csv', header=None)
    print(df.head())
    print(df.shape)
    print(labels.shape)
    df['labels'] = labels
    df.to_csv('output.csv', header=False)


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
    dataloader = DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=0,
    )
    labels = np.transpose(
        scipy.io.loadmat(args.labels)[args.data_key + "_gt"]
    )
    r, c = labels.shape
    labels = labels[0:r-model_config["window_size"], 0:c-model_config["window_size"]].flatten()
    print(len(dataset), labels.size)

    model = SpatialRevisedVAE(
        s=model_config["window_size"],
        ld=model_config["latent_dims"],
        spectral_bands=img.spectral_bands,
        layers=model_config["ae_layers"],  # Encoder
        ss_layers=model_config["ss_layers"],  # LSTM
        ls_layers=model_config["ls_layers"],  # CNN
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    print("Model loaded!")
    # print("Model Summary:")
    # summary(model, (img.spectral_bands, model_config["window_size"], model_config["window_size"]))

    make_classification_dataset(model, dataloader, labels)


