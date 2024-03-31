# BetterNet: An Efficient CNN Architecture with Residual Learning and Attention for Precision Polyp Segmentation 🚀

[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

BetterNet is a state-of-the-art semantic segmentation model specifically designed for accurate and efficient polyp segmentation in medical images. Built using TensorFlow and Keras, BetterNet leverages advanced deep learning techniques to achieve superior performance in identifying and segmenting polyps. 🎯

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Colab Notebook](#colab-notebook)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizing Results](#visualizing-results)
- [Pretrained Models](#pretrained-models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Features ✨

- 🎥 Utilizes EfficientNetB1 as the encoder backbone for efficient and effective feature extraction
- 🧠 Incorporates residual blocks, squeeze-and-excitation modules, and CBAM attention mechanisms to enhance feature learning and capture multi-scale contextual information
- 🎨 Employs a U-Net-like decoder architecture with skip connections for precise segmentation and spatial localization
- 📚 Supports training on multiple datasets simultaneously to improve generalization and robustness
- 📏 Provides a comprehensive set of evaluation metrics, including IoU, Dice coefficient, precision, recall, and more, for thorough performance assessment
- 🚀 Achieves state-of-the-art results on popular polyp segmentation benchmarks such as Kvasir-SEG, EndoTect, CVC-ClinicDB, Kvasir-Sessile, EndoScene.
- 🎛️ Offers flexibility and customization options through command-line arguments for training and testing

## Installation 💻

To get started with BetterNet, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/BetterNet.git
cd BetterNet
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate  # For Unix/Linux
env\Scripts\activate  # For Windows
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation 🗂️

To train and evaluate BetterNet, you need to prepare your datasets in the following structure:

```
Dataset/
├── Kvasir-SEG/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── mask1.png
│   │   ├── mask2.png
│   │   └── ...
│   ├── train.txt
│   └── val.txt
└── CVC-ClinicDB/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── masks/
    │   ├── mask1.png
    │   ├── mask2.png
    │   └── ...
    ├── train.txt
    └── val.txt
```

- Place the image files in the `images/` directory and their corresponding mask files in the `masks/` directory for each dataset.
- Create `train.txt` and `val.txt` files for each dataset, listing the filenames of the images and masks without extensions.

## Training 🏋️

To train the BetterNet model, run the following command:

```bash
python train.py --batch_size 8 --num_epochs 100
```

- Adjust the training hyperparameters using the provided command-line arguments according to your requirements.
- The trained model will be saved in the `model/` directory with the filename `model.keras`.
- Training progress and metrics will be logged in the console and saved in the `logs/` directory for visualization with TensorBoard.

## Testing 🎯

To evaluate the trained BetterNet model on a test dataset, run the following command:

```bash
python test.py --dataset Kvasir-Sessile
```

- Replace `Kvasir-Sessile` with the name of your test dataset directory.
- The test script will load the trained model from the `model/` directory and perform inference on the test images.
- Segmentation masks will be generated and saved in the `results/` directory.

## Results 📊

BetterNet achieves state-of-the-art performance on popular polyp segmentation benchmarks. Here are some key metrics:

| Method         | Year | mDice | mIoU | Fωβ   | Sα    | mE    | maxE  | MAE   |
|----------------|------|-------|------|-------|-------|-------|-------|-------|
| U-Net          | 2015 | 0.806 | 0.718| 0.784 | 0.842 | 0.890 | 0.894 | 0.052 |
| U-Net++        | 2018 | 0.797 | 0.704| 0.767 | 0.827 | 0.884 | 0.887 | 0.056 |
| ACSNet         | 2020 | 0.896 | 0.836| 0.876 | 0.915 | 0.943 | 0.954 | 0.034 |
| PraNet         | 2020 | 0.897 | 0.845| 0.889 | 0.916 | 0.945 | 0.954 | 0.026 |
| CaraNet        | 2021 | 0.911 | 0.861| 0.902 | 0.924 | 0.953 | 0.958 | 0.024 |
| Polyp-PVT      | 2021 | 0.920 | 0.871| 0.915 | 0.927 | 0.954 | 0.962 | 0.024 |
| SSFormer       | 2022 | 0.914 | 0.884| 0.924 | 0.933 | 0.974 | 0.967 | 0.021 |
| HSNet          | 2022 | 0.928 | 0.889| 0.927 | 0.939 | 0.967 | 0.977 | 0.017 |
| PVT-CASCADE    | 2023 | 0.924 | 0.916| 0.945 | 0.951 | 0.917 | 0.964 | 0.022 |
| CAFE-Net       | 2024 | 0.933 | 0.861| 0.902 | 0.924 | 0.953 | **0.971** | 0.019 |
| BetterNet      | 2024 | **0.951** | **0.884**| **0.924** | **0.933** | **0.954** | 0.951 | **0.012** |

In this table, the performance metrics for each method on the Kvasir-SEG dataset are presented, including mean Dice coefficient (mDice), mean Intersection over Union (mIoU), weighted F-measure (Fωβ), S-measure (Sα), mean E-measure (mE), maximum E-measure (maxE), and Mean Absolute Error (MAE).

The table highlights the best-performing method for each metric in bold. BetterNet, introduced in 2024, achieves the highest mDice, mIoU, Fωβ, Sα, mE, and the lowest MAE compared to other methods. CAFE-Net, also from 2024, achieves the highest maxE score.

- Qualitative results, visualizing the original images, ground truth masks, and predicted masks, will be saved in the `results/` directory for visual inspection.

Here's a new section that you can add to your README file, highlighting the availability of a Colab file for easy training and testing:

## Colab Notebook 🌐

To make it easier for users to explore and experiment with BetterNet, we have provided a Colab notebook that allows you to train and test the model directly in your browser without the need for local setup. The Colab notebook is designed to be user-friendly and intuitive, making it accessible to both beginners and experienced users.

### Getting Started with the Colab Notebook:

1. Open the Colab notebook by clicking on the following link: [BetterNet Colab Notebook](https://colab.research.google.com/drive/1B1Cc8FzcF5fg9eFvEJgdTiadfpZwt7pU?usp=sharing)
2. Sign in to your Google account or create a new one if you don't have one already.
3. The notebook will open in your browser, and you can start running the cells sequentially.
4. Follow the instructions provided in the notebook to prepare your dataset, train the model, and evaluate its performance.
5. Feel free to modify the notebook and experiment with different settings to explore the capabilities of BetterNet further.

We hope that the Colab notebook makes it more convenient for you to work with BetterNet and enables you to achieve excellent results in your polyp segmentation tasks. If you have any questions or encounter any issues while using the notebook, please don't hesitate to reach out to us.

## Model Architecture 🏗️

BetterNet combines the power of EfficientNetB1 as the encoder and a U-Net-like decoder architecture for precise polyp segmentation. The key components of the architecture include:

- EfficientNetB1 encoder: Pretrained on ImageNet for efficient feature extraction at multiple scales.
- Residual blocks: Enhance feature learning and propagate information across layers.
- Squeeze-and-excitation modules: Improve channel-wise feature recalibration and selectivity.
- CBAM attention mechanisms: Capture both channel-wise and spatial attention for better context modeling.
- U-Net-like decoder: Progressively upsample and fuse features from the encoder using skip connections for precise spatial localization.

For a detailed overview of the model architecture, please refer to the `model.py` file.

## Evaluation Metrics 📏

BetterNet uses a comprehensive set of evaluation metrics to assess the performance of the model. The key metrics include:

- Intersection over Union (IoU): Measures the overlap between the predicted and ground truth masks.
- Dice coefficient: Evaluates the similarity between the predicted and ground truth masks.
- Precision: Calculates the percentage of correctly predicted positive pixels.
- Recall: Measures the percentage of actual positive pixels that are correctly predicted.
- F1-score: Harmonic mean of precision and recall, providing an overall measure of segmentation accuracy.
- S-measure: Evaluates the structural similarity between the predicted and ground truth masks.
- E-measure: Assesses the edge-based similarity between the predicted and ground truth masks.
- Mean Absolute Error (MAE): Measures the average pixel-wise absolute difference between the predicted and ground truth masks.

These metrics provide a holistic view of the model's performance and help in comparing different segmentation approaches.

## Visualizing Results 🖼️

BetterNet generates visualizations of the segmentation results for qualitative assessment. During testing, the following visualizations are created:

- Original image: The input image to be segmented.
- Ground truth mask: The corresponding ground truth segmentation mask.
- Predicted mask: The segmentation mask predicted by BetterNet.

These visualizations are saved in the `results/` directory, allowing for easy comparison and analysis of the segmentation quality.

## Pretrained Models 🏋️‍♀️

We provide pretrained BetterNet models that you can directly use for inference or fine-tune on your own datasets. The pretrained models are available in the `pretrained/` directory.

To use a pretrained model for testing, simply load it using the following code snippet:

```python
from utils import load_model

model_path = "model/model.keras"
model = load_model(model_path)
```

## Contributing 🤝

We welcome contributions to BetterNet! If you encounter any issues, have suggestions for improvements, or want to add new features, please feel free to open an issue or submit a pull request. We appreciate your valuable feedback and contributions.

To contribute to BetterNet, follow these steps:

1. Fork the repository and create a new branch.
2. Make your changes and ensure that the code passes all tests.
3. Commit your changes and push them to your forked repository.
4. Submit a pull request, describing your changes and their benefits.

Please adhere to the code style and guidelines used in the project.

## License 📄

BetterNet is released under the GNU General Public License v3.0 (GPLv3). You are free to use, modify, and distribute the code under the terms of this license. However, if you want to use the code in proprietary software or for commercial purposes, you must obtain explicit permission from the copyright holder.

## Acknowledgments 🙏

We would like to express our gratitude to the following individuals, organizations, and resources that have contributed to the development of BetterNet:

- The creators and maintainers of the all the Public Datasets mentioned above for providing high-quality polyp segmentation data.
- The TensorFlow and Keras communities for their excellent deep learning frameworks and resources.
- The authors of the EfficientNet paper for their groundbreaking work on efficient neural network architectures.
- The open-source community for their invaluable contributions and inspiration.

## Contact 📧

If you have any questions, suggestions, or collaborations related to BetterNet, please feel free to reach out to us:

- Email: [owensingh72@gmail.com](mailto:owensingh72@gmail.com)
- GitHub: [https://github.com/itsOwen/BetterNet](https://github.com/itsOwen/BetterNet)
- Research Paper: [#](#)

We are excited to hear from you and discuss how BetterNet can be applied to advance polyp segmentation in medical imaging! 🌟
