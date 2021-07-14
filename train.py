from models.generator import Generator
from models.discriminator import Discriminator
from utils.train_models import *
from utils.show_data import *
from utils.load_data import *
from utils.gan_parameters import *
from utils.device_info import *
from utils.save_weights import save_losses_and_scores, save_model_weights_only


import os
import numpy as np


if __name__ == '__main__':
    ganParameters = GanParameters()

    train_ds = load_data_from_dataset(data_dir=ganParameters.get_dataroot())
    train_dl = load_data_to_dataloader(train_ds)

    # img, _ = next(iter(train_dl))
    # showImages(img, title='first_batch_images')

    device = get_default_device()
    print(f'{device} device is used')

    train_dl = DeviceDataLoader(train_dl, device)

    discriminator = Discriminator(
        ganParamerters.get_nc(),
        ganParamerters.get_ndf()
    )
    generator = Generator(
        ganParamerters.get_nz(),
        ganParamerters.get_ngf(),
        ganParamerters.get_nc()
    )

    discriminator = to_device(discriminator, device)
    generator = to_device(generator, device)

    # showImages(
    #     generator(
    #         torch.rand(
    #             ganParamerters.get_batch_size(),
    #             ganParamerters.get_nz(),
    #             1,
    #             1,
    #             device=device
    #         )
    #     ),
    #     title='first_random_images'
    # )

    # save_samples(
    #     0,
    #     torch.rand(
    #         ganParamerters.get_batch_size(),
    #         ganParamerters.get_nz(),
    #         1,
    #         1,
    #         device=device
    #     ),
    #     generator=generator,
    # )

    ganParamerters.set_num_epochs(25)
    history = fit(
        discriminator,
        generator,
        train_dl,
        device,
        epochs=ganParamerters.get_num_epochs(),
        lr=ganParamerters.get_lr()
    )

    # img = img.to(device)
    # rp = discriminator(img)
    # print(rp.shape)
    # print(img.size())
    # rt = torch.ones(img.size(0), 1, 1, 1, device=device)
    # rl = F.binary_cross_entropy(rp, rt)
    # print(rl)

    save_model_weights_only(generator=generator, discriminator=discriminator)
    save_losses_and_scores(history=history)
