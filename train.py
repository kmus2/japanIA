import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import os
import numpy as np

# Import models from model.py
from model import Generator, Discriminator

# --- Configuration ---
# You can run this script in a Google Colab notebook.
# 1. Upload `model.py` to your Colab environment.
# 2. Make sure to select a GPU runtime (Runtime -> Change runtime type -> T4 GPU).
# 3. Mount your Google Drive to save the trained model:
#    from google.colab import drive
#    drive.mount('/content/drive')
# 4. Set the `output_dir` to a path in your Google Drive.

# Hyperparameters
n_epochs = 10 # Reduced for faster training.
batch_size = 64
lr = 0.0002
b1 = 0.5 # Adam optimizer beta1
b2 = 0.999 # Adam optimizer beta2
latent_dim = 100 # Dimensionality of the latent space
img_size = 28
channels = 1
num_classes = 10
sample_interval = 400 # Interval between saving image samples
output_dir = "output" # Change this to your Google Drive path in Colab, e.g., "/content/drive/MyDrive/gan_output"

# --- Setup ---
img_shape = (channels, img_size, img_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

# --- Models ---
generator = Generator(latent_dim=latent_dim, num_classes=num_classes, img_shape=img_shape).to(device)
discriminator = Discriminator(num_classes=num_classes, img_shape=img_shape).to(device)

# Loss function
adversarial_loss = nn.MSELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# --- Dataset ---
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    torchvision.datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# --- Training Loop ---
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = torch.randn(n_row ** 2, latent_dim, device=device)
    labels = torch.LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])).to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join(output_dir, f"images/{batches_done}.png"), nrow=n_row, normalize=True)

print("Starting Training...")

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.full((imgs.size(0), 1), 1.0, device=device, requires_grad=False)
        fake = torch.full((imgs.size(0), 1), 0.0, device=device, requires_grad=False)

        # Configure input
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(imgs.shape[0], latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (imgs.shape[0],), device=device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

        if batches_done % sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)


# --- Save Model ---
model_save_path = os.path.join(output_dir, "generator.pth")
torch.save(generator.state_dict(), model_save_path)
print(f"Generator model saved to {model_save_path}")
print("Training finished.") 