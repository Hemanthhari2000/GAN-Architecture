# from gan_parameters import GanParameters

# global ganParameters
# ganParameters = GanParameters()


# def printWorkers():
#     print(ganParameters.get_workers())


# def printDataset():
#     print(ganParameters.get_dataroot())


# printWorkers()
# printDataset()


from utils.show_data import showImages
from utils.load_data import *
from utils.gan_parameters import GanParameters
from utils.device_info import *


if __name__ == '__main__':
    ganParameters = GanParameters()

    train_ds = load_data_from_dataset(data_dir=ganParameters.get_dataroot())
    train_dl = load_data_to_dataloader(train_ds)

    for img, _ in train_dl:
        print(img.shape)
        showImages(img, title='first_batch_images')
        break

    print(get_default_device())
