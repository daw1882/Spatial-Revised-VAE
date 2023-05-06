import torch
from torch.utils.data import DataLoader

from spectral_vae import SpatialRevisedVAE, SpectralSpatialEncoder, SpectralSpatialDecoder
from spectral_dataset import SpectralVAEDataset, SpectralImage
from loss import VAE_loss
import utils


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, inputs in enumerate(dataloader):
        # Every data instance is an input + label pair

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, xss, xls = model(inputs)

        input_vector = utils.extract_spectral_data(inputs, model.spectral_bands)
        # print("training input tensor", torch.any(input_vector < 0))
        # output_vector = utils.extract_spectral_data(outputs, model.spectral_bands)
        # Compute the loss and its gradients
        loss = VAE_loss(input_vector, outputs, model.mu, model.var, xls, xss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # exit(0)

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 1000  # loss per batch
            # print('\tbatch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


if __name__ == '__main__':
    im = SpectralImage("../test") # Change to be specific file path.
    dataset = SpectralVAEDataset(im, 11)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)
    print("Dataloader created!")

    model = SpatialRevisedVAE(11, 16, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 1

    print("Beginning Training!")
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()
        print(avg_loss)


