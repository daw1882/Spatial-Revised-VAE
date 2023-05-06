import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_vae import SpatialRevisedVAE
from spectral_dataset import SpectralVAEDataset, SpectralImage
from loss import VAE_loss
import utils
from tqdm import tqdm
from time import sleep
from datetime import datetime
import os


if __name__ == '__main__':
    model_dir = "./models"
    if not os.path.exists(model_dir):
        print(f"Model directory does not exist, creating directory at {model_dir}.")
        os.makedirs(model_dir)

    im = SpectralImage("C:\\Users\\dade_\\NN_DATA\\testing") # Change to be specific file path.
    dataset = SpectralVAEDataset(im, 11)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)
    print("Dataloader created!")

    model = SpatialRevisedVAE(11, 16, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    EPOCHS = 1
    update_iters = 10

    print("Beginning Training!")
    sleep(0.5)

    best_loss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for epoch in range(EPOCHS):
        with tqdm(dataloader, miniters=update_iters, unit="batch") as t_epoch:
            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            losses = []
            # Train loop for single epoch
            for i, inputs in enumerate(t_epoch):
                t_epoch.set_description(f"Epoch {epoch}")
                # Every data instance is an input + label pair

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs, xss, xls = model(inputs)

                input_vector = utils.extract_spectral_data(inputs, model.spectral_bands)

                # Compute the loss and its gradients
                reconstruction_term, kl_term, homology_term = VAE_loss(input_vector, outputs, model.mu, model.var, xls, xss)
                loss = reconstruction_term + kl_term + homology_term
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
                if i == 10:
                    break

        avg_loss = np.average(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = '{}/model_{}_{}'.format(model_dir, timestamp, epoch)
            torch.save(model.state_dict(), model_path)
