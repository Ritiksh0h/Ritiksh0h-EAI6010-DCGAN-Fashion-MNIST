# DCGAN Fashion-MNIST Generator

Adapted from [PyTorch's official DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) to generate synthetic clothing images using the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

## Live Demo

👗 **Try it now:** [https://huggingface.co/spaces/ritikshah2002/dcgan-fashion-mnist](https://huggingface.co/spaces/ritikshah2002/dcgan-fashion-mnist)

- **Input:** Number of images (1–16) and an optional random seed
- **Output:** A grid of generated 32×32 grayscale clothing images

## Changes from Original Tutorial

| Parameter | Original (CelebA) | Adapted (Fashion-MNIST) |
|-----------|-------------------|------------------------|
| Channels | 3 (RGB) | 1 (Grayscale) |
| Image Size | 64×64 | 32×32 |
| Generator Layers | 5 | 4 |
| Discriminator Layers | 5 | 4 |
| Epochs | 5 | 50 |

## Files

- `script.py` — Standalone training script
- `Module_4_assignment.ipynb` — Colab training notebook
- `generator_fashion_mnist.pth` — Trained Generator weights
- `discriminator_fashion_mnist.pth` — Trained Discriminator weights
- `real_images.png` — Sample of real training images

## Results

- **Simplified FID Score:** 29.66
- **Training:** 50 epochs, batch size 128, lr=0.0002, Adam (β1=0.5)

## HuggingFace Space Deployment

The trained Generator is deployed as a Gradio microservice on HuggingFace Spaces.

| Detail | Value |
|--------|-------|
| **URL** | [huggingface.co/spaces/ritikshah2002/dcgan-fashion-mnist](https://huggingface.co/spaces/ritikshah2002/dcgan-fashion-mnist) |
| **SDK** | Gradio |
| **Hardware** | CPU |
| **Model Size** | ~1.2 MB |

## References

- Radford et al. (2016). Unsupervised Representation Learning with DCGANs
- Xiao et al. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking ML Algorithms
- Goodfellow et al. (2016). Generative Adversarial Nets