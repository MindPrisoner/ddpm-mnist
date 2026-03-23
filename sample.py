import os
import torch
from torchvision.utils import save_image

from models.unet import SimpleUNet
from utils.diffusion_utils import Diffusion


@torch.no_grad()
def sample(model, diffusion, n, device):
    model.eval()
    x = torch.randn((n, 1, 28, 28)).to(device)

    for i in reversed(range(1, diffusion.noise_steps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]

        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
        ) + torch.sqrt(beta) * noise

    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("outputs/ddpm_unet_mnist.pth", map_location=device))

    diffusion = Diffusion(device=device)

    samples = sample(model, diffusion, n=16, device=device)
    save_image(samples, "outputs/generated_samples.png", nrow=4)
    print("Generated samples saved to outputs/generated_samples.png")


if __name__ == "__main__":
    main()
