import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.spectral_dataset import SpectralImage, SpectralVAEDataset
import argparse
from spectral_spatial_vae.spectral_vae import SpatialRevisedVAE
import json
from torchsummary import summary
from tqdm import tqdm
import cv2
from spectral_spatial_vae import utils


def encode(model: SpatialRevisedVAE, dataloader, width, height, s):
    # Note, this is really for fancy display of the progress,
    # the tqdm is not necessary but the dataloader is for
    # performing inference in batches
    original = None
    reconstructed = None
    with tqdm(dataloader, unit="batch") as loader:
        for batch in loader:
            # print(batch, batch.size())
            # you'd call model.decoder(input) for decoding part
            output_e = model.encoder(batch)
            output = model.decoder(output_e)
            output = output.cpu().detach().numpy()
            if reconstructed is None:
                reconstructed = output
                input_vector = utils.extract_spectral_data(batch, s).cpu().detach().numpy()
                original = input_vector
            else:
                reconstructed = np.concatenate((reconstructed, output), axis=0)
                input_vector = utils.extract_spectral_data(batch, s).cpu().detach().numpy()
                original = np.concatenate((original, input_vector), axis=0)
    for channel in range(model.spectral_bands):
        orig_show = original[:, channel]
        cv2.imwrite(f"images/{channel}_original.tif", orig_show.reshape(
            height, width))
        img = reconstructed[:, channel]
        # img_show = cv2.normalize(img.reshape(111, 111), None, 0, 1.0,
        #                          cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_show = img.reshape(height, width)
        cv2.imwrite(f"images/{channel}_decode_result.tif", img_show)


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
    # parser.add_argument(
    #     '--output_dir',
    #     help='Path to output the dataset to.',
    #     required=True,
    #     type=str,
    # )
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
    parser.add_argument(
        '--image_width',
        help='Width of the output image.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--image_height',
        help='Height of the output image.',
        required=True,
        type=int,
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    model_config = config["model"]
    training_config = config["training"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = SpectralImage(args.image, args.data_type, args.data_key)
    dataset = SpectralVAEDataset(img, model_config["window_size"],
                                 device=device)
    dataloader = DataLoader(
        dataset, batch_size=training_config["batch_size"], shuffle=False,
        num_workers=0,
    )

    model = SpatialRevisedVAE(
        s=model_config["window_size"],
        ld=model_config["latent_dims"],
        spectral_bands=img.spectral_bands,
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    print("Model loaded!")
    time.sleep(0.5)
    # print("Model Summary:")
    # summary(model, (model_config["window_size"], model_config["window_size"],
    #                 img.spectral_bands))

    encode(model, dataloader, args.image_width, args.image_height,
           model_config["window_size"])
