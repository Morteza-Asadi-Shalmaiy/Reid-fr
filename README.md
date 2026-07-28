# GAN Cat Faces

A DCGAN (Deep Convolutional GAN) trained from scratch on the [Cats Faces 64x64 for Generative Models](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) dataset, generating novel 64×64 cat face images from random noise.

<!-- TODO: replace with your actual GIF once training is done -->
![Training progress](assets/training_progress.gif)

<!--
Badges — fill in <YOUR_USERNAME>/<YOUR_REPO> once this is pushed to GitHub,
then this becomes a one-click "run this in Colab" button.
-->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/<YOUR_REPO>/blob/main/gan_cats.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DCGAN-orange)

## Overview

This project implements a DCGAN — a generator/discriminator pair trained adversarially — to synthesize cat face images. It includes:

- A from-scratch DCGAN architecture (transposed-conv generator, strided-conv discriminator)
- Checkpointing every N epochs with full resume support (model + optimizer states + loss history + fixed noise vector), so training can be paused and continued across multiple sessions
- A fixed latent vector tracked across epochs, so the sample grid shown at each epoch reflects the *same* points in latent space evolving over training — not unrelated random draws
- Loss curve tracking and plotting
- A standalone inference cell to generate new images from any saved checkpoint

## Results

<!-- TODO: after training, drop your best sample grid + loss curve into assets/ and update these -->

**Sample outputs (epoch `<N>`):**

![Final samples](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/gan-cats-faces/refs/heads/main/assets/epoch30.png)

**Training loss:**

![Loss curve](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/gan-cats-faces/refs/heads/main/assets/plot0-20.png)
![Loss curve](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/gan-cats-faces/refs/heads/main/assets/plot20-39.png)
Trained for `<N>` epochs on a `<GPU type, e.g. Colab T4>` GPU, `<total training time>`.

## Architecture

| | Generator | Discriminator |
|---|---|---|
| Input | 100-dim latent vector | 64×64×3 image |
| Layers | 5× `ConvTranspose2d` + BatchNorm + ReLU, final `Tanh` | 5× `Conv2d` + BatchNorm + LeakyReLU(0.2), final `Sigmoid` |
| Output | 64×64×3 image | Real/fake probability |

Standard DCGAN weight initialization (`N(0, 0.02)`) is applied to all conv and batchnorm layers. Optimized with Adam (`lr=0.0002`, `beta1=0.5`) for both networks, trained with binary cross-entropy loss.

## Repo structure

```
gan-cats-faces/
├── gan_cats.ipynb          # Main notebook: data loading, model, training, resume, inference
├── make_gif.py             # Stitches saved epoch samples into assets/training_progress.gif
├── requirements.txt
└── assets/                 # Curated results shown in this README
    ├── training_progress.gif
    ├── final_samples.png
    └── loss_curve.png
```

`checkpoints/`, `samples/`, and raw `.pth` weight files are produced when you run the notebook but are git-ignored — only the curated highlights in `assets/` are committed.

## Running it

1. Open `gan_cats.ipynb` in Google Colab (badge above) or locally with a GPU.
2. Run all cells top to bottom. This downloads the dataset via `kagglehub`, trains the DCGAN, and saves sample grids to `samples/` and checkpoints to `checkpoints/` every 2 epochs.
3. **To resume training later** (e.g. you trained 10 epochs, now want 20 total): re-run the "Resume Training" cell with `num_epochs` set to your new total and `resume_training = True`. See the notebook's Resume Training section for details.
4. **To generate images from a saved checkpoint** without retraining, use the `generate_images()` cell near the end of the notebook.
5. Once you have a full set of `samples/fake_cats_epoch_*.png`, build the GIF:
   ```bash
   pip install -r requirements.txt
   python make_gif.py --samples-dir samples --output assets/training_progress.gif
   ```

## Credits

- Dataset: [Cats Faces 64x64 for Generative Models](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) by spandan2 on Kaggle.
- Architecture follows the standard DCGAN design from [Radford et al., 2015](https://arxiv.org/abs/1511.06434).

## License

<!-- TODO: pick a license, e.g. MIT -->
MIT
