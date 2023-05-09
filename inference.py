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


def encode(model: SpatialRevisedVAE, dataloader):
    # Note, this is really for fancy display of the progress,
    # the tqdm is not necessary but the dataloader is for
    # performing inference in batches
    with tqdm(dataloader, unit="batch") as loader:
        for batch in loader:
            print(batch, batch.size())
            # you'd call model.decoder(input) for decoding part
            output_e = model.encoder(batch)
            output = model.decoder(output_e)
            output = output.cpu().detach().numpy()
            print(output.shape)
            for channel in range(51):
                # input_vector = utils.extract_spectral_data(batch, 11).cpu().detach().numpy()
                # input_vector = input_vector[:, channel]
                # cv2.imwrite(f"images/{channel}_original.tif", input_vector.reshape(111, 111))
                img = output[:, channel]
                # img_show = cv2.normalize(img.reshape(111, 111), None, 0, 1.0,
                #                          cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                img_show = img.reshape(111, 111)
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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    model_config = config["model"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = SpectralImage(args.image)
    dataset = SpectralVAEDataset(img, model_config["window_size"],
                                 device=device)
    dataloader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0,
    )
    print(len(dataset))

    model = SpatialRevisedVAE(
        s=model_config["window_size"],
        ld=model_config["latent_dims"],
        spectral_bands=img.spectral_bands,
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    print("Model loaded!")
    # print("Model Summary:")
    # summary(model, (model_config["window_size"], model_config["window_size"],
    #                 img.spectral_bands))

    encode(model, dataloader)
