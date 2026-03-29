"""
DCGAN Fashion-MNIST Generator Microservice
Deployed on HuggingFace Spaces with Gradio.
"""

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image

# Generator Architecture (must match training code)
nz = 100
ngf = 64
nc = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


# Load trained model
device = torch.device("cpu")
model = Generator().to(device)
model.load_state_dict(torch.load("generator_fashion_mnist.pth", map_location=device))
model.eval()


def generate_fashion(num_images, seed):
    if seed >= 0:
        torch.manual_seed(int(seed))

    num_images = max(1, min(16, int(num_images)))

    with torch.no_grad():
        noise = torch.randn(num_images, nz, 1, 1, device=device)
        generated = model(noise).cpu()

    images = generated.squeeze(1).numpy()
    images = ((images + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    grid_h = rows * 32 + (rows - 1) * 4
    grid_w = cols * 32 + (cols - 1) * 4
    grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 255

    for idx in range(num_images):
        r, c = divmod(idx, cols)
        y = r * 36
        x = c * 36
        grid[y : y + 32, x : x + 32] = images[idx]

    pil_img = Image.fromarray(grid, mode="L")
    pil_img = pil_img.resize((grid_w * 4, grid_h * 4), Image.Resampling.NEAREST)
    return pil_img


demo = gr.Interface(
    fn=generate_fashion,
    inputs=[
        gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Number of Images"),
        gr.Number(value=-1, label="Random Seed", info="Use -1 for random, any other number for reproducible output.", precision=0),
    ],
    outputs=gr.Image(type="pil", label="Generated Fashion Images"),
    title="DCGAN Fashion-MNIST Generator",
    description="Generates synthetic 32x32 grayscale clothing images from random noise using a DCGAN trained on Fashion-MNIST.",
    examples=[[4, 42], [1, 123], [16, -1], [9, 7]],
    allow_flagging="never",
)

demo.launch()
