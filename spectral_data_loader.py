import torch
from torch.utils.data import Dataset, DataLoader
from spectral_image import SpectralImage


class SpectralVAEDataset(Dataset):

    def __init__(self, spectral_im: SpectralImage, window_size):
        self.spectral_im = spectral_im
        self.window_size = window_size

    def __len__(self):
        return self.spectral_im.get_length(self.window_size)

    def __getitem__(self, idx):
        sample = self.spectral_im.get_window(idx, self.window_size)
        return torch.from_numpy(sample)


if __name__ == '__main__':
    print("---------------------")
    print("DATASET TEST")
    im = SpectralImage("C:\\Users\\dade_\\NN_DATA\\testing")
    dataset = SpectralVAEDataset(im, 11)

    print(len(dataset))
    for i in range(len(dataset)):
        x = dataset[i]
        print(i, x.size())

        if i == 3:
            break

    print("---------------------")
    print("DATALOADER TEST")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())

        if i_batch == 3:
            break
