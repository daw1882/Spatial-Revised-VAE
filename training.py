import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_vae import SpatialRevisedVAE
from spectral_dataset import SpectralVAEDataset, SpectralImage
from loss import reconstruction_loss
import utils
from tqdm import tqdm
from time import sleep
from datetime import datetime
import os
from torchsummary import summary
import math
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        help='Directory to save model directories in.',
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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    model_config = config["model"]
    train_config = config["training"]

    # Training Parameters
    model_dir = args.model_dir
    # image_dir = "/mnt/d/PycharmProjects/nn_data/test"
    image_dir = args.image
    epochs = train_config["epochs"]
    update_iters = 10
    batch_size = train_config["batch_size"]
    window_size = model_config["window_size"]
    latent_dimensions = model_config["latent_dims"]

    # Optimizer Parameters
    learn_rate = train_config["learn_rate"]

    if not os.path.exists(model_dir):
        print(f"Model directory does not exist, creating directory at {model_dir}.")
        os.makedirs(model_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}.")

    spec_img = SpectralImage(image_dir)
    dataset = SpectralVAEDataset(spec_img, window_size, device)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    print("Dataloader created!")

    model = SpatialRevisedVAE(
        s=window_size,
        ld=latent_dimensions,
        spectral_bands=spec_img.spectral_bands,
        layers=model_config["ae_layers"],       # Encoder
        ss_layers=model_config["ss_layers"],    # LSTM
        ls_layers=model_config["ls_layers"],    # CNN
        device=device,
    ).to(device)

    print("Model Summary:")
    summary(model, (window_size, window_size, spec_img.spectral_bands))

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    print("Beginning Training!")
    sleep(0.5)
    best_loss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for epoch in range(epochs):
        with tqdm(dataloader, miniters=update_iters, unit="batch") as t_epoch:
            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            losses = []
            # Train loop for single epoch
            for i, batch in enumerate(t_epoch):
                t_epoch.set_description(f"Epoch {epoch}")

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(batch)

                input_vector = utils.extract_spectral_data(batch, model.spectral_bands)

                # Compute the loss and its gradients
                reconstruction_term = reconstruction_loss(input_vector, outputs)
                homology_term = model.encoder.homology
                kl_term = model.encoder.kl
                # print("reconstruction loss", reconstruction_term)
                # print("homology loss", homology_term)
                # print("kl loss", kl_term)
                if i == 100:
                    loss = reconstruction_term + kl_term + homology_term
                else:
                    loss = reconstruction_term + homology_term
                if math.isnan(loss.item()):
                    raise ValueError("Loss went to nan.")
                losses.append(loss.item())
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                if i % update_iters == 0:
                    t_epoch.set_postfix({
                        "loss": "{:.4f}".format(loss.item()),
                        "reconstruction": "{:.4f}".format(reconstruction_term.item()),
                        "kl": "{:.4f}".format(kl_term.item()),
                        "homology": "{:.4f}".format(homology_term.item()),
                    })

        avg_loss = np.average(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = '{}/model_{}_{}.pt'.format(model_dir, timestamp, epoch)
            torch.save(model.state_dict(), model_path)
