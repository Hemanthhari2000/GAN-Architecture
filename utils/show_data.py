from torchvision.utils import make_grid
from torchvision.utils import save_image


import matplotlib.pyplot as plt
import os

global testing_image_path
testing_image_path = 'generated/testing_images'

global generated_image_path
generated_image_path = 'generated/generated_images'


def denorm(img_tensor):
    return img_tensor * 0.5 + 0.5


def showImages(images, nmax=64, title=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid(denorm(images.cpu().detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()
    if title:
        fig.savefig(os.path.join(testing_image_path,
                    f'{title}.png'), transparent=True)


def showSingleBatch(dl, nmax=64, title=None):
    for img, _ in dl:
        showImages(img, nmax=nmax, title=title)
        break


def save_samples(index, latent_tensors, generator, show=True, nmax=64):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images[:nmax]), os.path.join(
        generated_image_path, fake_fname), nrow=8)
    print("Saving", fake_fname)
    if show:
        _, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach()
                  [:nmax], nrow=8).permute(1, 2, 0))
        plt.show()
