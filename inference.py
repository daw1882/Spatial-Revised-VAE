import torch
from torch.utils.data import DataLoader
from spectral_dataset import SpectralImage, SpectralVAEDataset
import argparse
from spectral_vae import SpatialRevisedVAE
import json
from torchsummary import summary
from tqdm import tqdm


def encode(model: SpatialRevisedVAE, dataloader):
    # Note, this is really for fancy display of the progress,
    # the tqdm is not necessary but the dataloader is for
    # performing inference in batches
    with tqdm(dataloader, unit="batch") as loader:
        for batch in loader:
            # you'd call model.decoder(input) for decoding part
            output = model.encoder(batch)
            print(output, output.size())


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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    model_config = config["model"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = SpectralImage(args.image)
    dataset = SpectralVAEDataset(img, model_config["window_size"],
                                 device=device)
    dataloader = DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=0,
    )
    print(len(dataset))

    model = SpatialRevisedVAE(
        s=model_config["window_size"],
        ld=model_config["latent_dims"],
        spectral_bands=img.spectral_bands,
        device=device,
    )
    model.load_state_dict(torch.load(args.model))
    model.eval()
    print("Model loaded!")
    print("Model Summary:")
    summary(model, (model_config["window_size"], model_config["window_size"],
                    img.spectral_bands))

    encode(model, dataloader)


