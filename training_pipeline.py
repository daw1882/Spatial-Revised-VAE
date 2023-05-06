import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_directml

from spectral_vae import SpatialRevisedVAE
from spectral_dataset import SpectralVAEDataset, SpectralImage
from loss import VAE_loss
import utils
from tqdm import tqdm
from time import sleep
from datetime import datetime
import os


if __name__ == '__main__':
    CUDA_GPU = True
    # Training Parameters
    model_dir = "./models"
    # image_dir = "/mnt/d/PycharmProjects/nn_data/test"
    image_dir = "D:/PycharmProjects/nn_data/test"
    epochs = 5
    update_iters = 10
    batch_size = 512
    window_size = 11
    latent_dimensions = 40

    # Optimizer Parameters
    learn_rate = 0.001

    if not os.path.exists(model_dir):
        print(f"Model directory does not exist, creating directory at {model_dir}.")
        os.makedirs(model_dir)

    if CUDA_GPU:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch_directml.device()
    print(f"Training on device: {device}.")

    spec_img = SpectralImage(image_dir)
    dataset = SpectralVAEDataset(spec_img, window_size, device)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True
    )
    print("Dataloader created!")

    model = SpatialRevisedVAE(
        s=window_size,
        ld=latent_dimensions,
        spectral_bands=spec_img.spectral_bands,
        layers=3,       # Encoder
        ss_layers=3,    # LSTM
        ls_layers=3,    # CNN
        device=device,
    ).to(device)

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
            for i, inputs in enumerate(t_epoch):
                t_epoch.set_description(f"Epoch {epoch}")

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs, xss, xls = model(inputs)

                input_vector = utils.extract_spectral_data(inputs, model.spectral_bands)

                # Compute the loss and its gradients
                reconstruction_term, kl_term, homology_term = VAE_loss(input_vector, outputs, model.mu, model.std, xls, xss)
                loss = reconstruction_term + kl_term + homology_term
                losses.append(loss.item())
                # print("tests:")
                # reconstruction_term.backward()
                # kl_term.backward()
                # homology_term.backward()

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
            model_path = '{}/model_{}_{}'.format(model_dir, timestamp, epoch)
            torch.save(model.state_dict(), model_path)
