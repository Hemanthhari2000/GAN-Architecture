import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.gan_parameters import GanParameters
from utils.show_data import save_samples


global ganParameters
ganParamerters = GanParameters()


def train_discriminator(discriminator, generator, real_images, opt_d, device):
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, 1, 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(ganParamerters.get_batch_size(),
                         ganParamerters.get_nz(), 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, 1, 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(discriminator, generator, opt_g, device):
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(ganParamerters.get_batch_size(),
                         ganParamerters.get_nz(), 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(ganParamerters.get_batch_size(),
                         1, 1, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


def fit(discriminator, generator, train_dl, device, epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Random Noise (latent vector)
    fixed_latent = torch.randn(
        ganParamerters.get_batch_size(),
        ganParamerters.get_nz(),
        1,
        1,
        device=device
    )

    #Losses and Scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):

            # Train discriminator

            loss_d, real_score, fake_score = train_discriminator(
                real_images=real_images,
                opt_d=opt_d,
                discriminator=discriminator,
                generator=generator,
                device=device
            )

            loss_g = train_generator(
                opt_g=opt_g,
                discriminator=discriminator,
                generator=generator,
                device=device
            )

        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

        # Save Generated images
        save_samples(index=epoch+start_idx, generator=generator, latent_tensors=fixed_latent, show=False)
    return losses_g, losses_d, real_scores, fake_scores
