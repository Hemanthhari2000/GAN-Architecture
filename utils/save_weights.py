import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.NOTSET)


global weights_path
weights_path = 'generated\weights'


def get_path(only_path=False):

    dirs = os.listdir(weights_path)

    if only_path:
        return os.path.join(weights_path, str(dirs[-1]))

    elif len(dirs) > 0:
        val = int(dirs[-1].split('_')[1]) + 1
        new_dir = f'iteration_{val}'
        path = os.path.join(weights_path, new_dir)
    else:
        path = os.path.join(weights_path, 'iteration_1')

    os.makedirs(path)
    return path


def save_model_weights_only(generator, discriminator):
    path = get_path()

    torch.save(
        generator.state_dict(),
        os.path.join(path, 'generator.pth')
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(path, 'discriminator.pth')
    )
    logging.info('Model Weights Saved')


def save_plot(val1, val2, path, ylabel, legend, title, filename):
    plt.plot(val1, '-')
    plt.plot(val2, '-')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)
    plt.savefig(os.path.join(path, f'{filename}.png'))


def save_losses_and_scores(history, with_plot=True):
    losses_g, losses_d, real_scores, fake_scores = history
    path = get_path(only_path=True)
    losses_g = np.array(losses_g)
    losses_d = np.array(losses_d)
    real_scores = np.array(real_scores)
    fake_scores = np.array(fake_scores)

    if with_plot:
        save_plot(
            losses_g,
            losses_d,
            path,
            ylabel='loss',
            legend=['Discriminator', 'Generator'],
            title='Losses',
            filename='losses'
        )
        save_plot(
            real_scores,
            fake_scores,
            path,
            ylabel='score',
            legend=['Real', 'Fake'],
            title='Scores',
            filename='scores'
        )

    np.save(os.path.join(path, 'losses_g.npy'), losses_g)
    np.save(os.path.join(path, 'losses_d.npy'), losses_d)
    np.save(os.path.join(path, 'real_scores.npy'), real_scores)
    np.save(os.path.join(path, 'fake_scores.npy'), fake_scores)

    logging.info('Metrics And Plots Saved Successfully')
