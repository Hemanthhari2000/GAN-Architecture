from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from utils.gan_parameters import GanParameters

global ganParameters
ganParamerters = GanParameters()


def load_data_from_dataset(data_dir, transform=None):

    if transform == None:
        transform = T.Compose([
            T.Resize(ganParamerters.get_image_size()),
            T.CenterCrop(ganParamerters.get_image_size()),
            T.ToTensor(),
            T.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    return ImageFolder(data_dir, transform=transform)


def load_data_to_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=ganParamerters.get_batch_size(),
        shuffle=True,
        num_workers=ganParamerters.get_workers(),
        pin_memory=True
    )
