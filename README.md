# Multimodal Diffusion Model

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This research repository contains the implementation of a Multimodal Diffusion Model. It leverages text, image, and style embeddings through a custom fusion architecture to generate conditionally styled images. 

This project explores integrating multimodal information inside the UNet generation process using components like CLIP for text and imagery, and a specialized style encoder.

## Features

- **Text & Image Encoders**: Uses standard CLIP models (`openai/clip-vit-base-patch32` and Stable Diffusion v1-4 encoders).
- **Style Encoder**: A custom CNN-based style extractor generating a style representation via Gram Matrix analysis.
- **Multimodal Fusion Module**: Fuses text, reference images, and style embeddings using Multihead Attention mechanism and dynamic modality weights.
- **Support for Multi-Datasets**: Built-in loaders and preprocessing pipelines for `COCO` and `Flowers102` datasets.
- **Custom Loss Function**: Combines diffusion MSE loss with directional loss (for text-image alignment) and style-matching loss.

## Project Structure

- `Genai_Research.ipynb`: The main notebook containing the full implementation of model architectures, training pipelines, dataset handling, and evaluation.
- `Research.pdf`: Complete research documentation detailing the design, results, and theoretical foundation of the model.
- `requirements.txt`: Python package dependencies.

## Installation

Ensure you have Python 3.10+ installed.

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Multimodal-Diffusion.git
   cd Multimodal-Diffusion
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

The code involves downloading large datasets (`COCO` and `Flowers102`). When you run the `Genai_Research.ipynb` notebook:
- **Flowers102** (~345 MB): Automatically downloads via `torchvision.datasets`.
- **COCO 2017** (~18 GB): Downloaded and extracted within the notebook using scripts for training and annotations. Ensure you have over 20 GB of free disk space.

## Architecture

![Architecture](https://img.shields.io/badge/Architecture-Custom-purple)

The core is governed by the `MultimodalDiffusionModel` class, which brings together:
1. `CLIPTextModel` (for prompts).
2. `CLIPVisionModel` (for reference images).
3. `StyleEncoder` (for capturing style references).
4. `MultimodalFusionModule` (Attention-based feature fusion).
5. `AutoencoderKL` and `UNet2DConditionModel` (Core latent diffusion components).

## Usage

Start the Jupyter server to interact with the code:
```bash
jupyter notebook
```
Open `Genai_Research.ipynb` and run the cells sequentially to initialize components, download the datasets, and trigger the training mechanism.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code for your research, please refer to the provided `Research.pdf` or cite it accordingly.
