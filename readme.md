# DualCascadeTSF-MobileNetV2: A Lightweight Violence Behavior Recognition Model

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)
- [Contact](#contact)

## Introduction
This repository contains the implementation of DualCascadeTSF-MobileNetV2, a lightweight deep learning model designed for violence behavior recognition. The model combines the strengths of the Dual Cascade Temporal Shift and Fusion (DualCascadeTSF) module with the lightweight architecture of MobileNetV2 to achieve high accuracy while maintaining low computational cost. This makes it ideal for deployment on edge devices, such as mobile phones or embedded systems.

The model addresses the challenges of extracting long-term temporal information in video streams while avoiding excessive computational overhead, making it suitable for real-time applications like surveillance and security monitoring.

## Key Features
- **Temporal Feature Extraction**: Utilizes the DualCascadeTSF module to enhance the extraction of long-term temporal correlations in video data.
- **Lightweight Design**: Combines the DualCascadeTSF module with MobileNetV2, significantly reducing the number of parameters and computational requirements.
- **High Accuracy**: Achieves state-of-the-art performance on violence detection datasets (Crowd Violence, RWF-2000, Hockey Fights).
- **Efficient Deployment**: Optimized for resource-constrained environments, such as edge devices.

## Model Architecture
The DualCascadeTSF-MobileNetV2 model integrates the following components:

### DualCascadeTSF Module:
- Introduces feature fusion after each temporal shift operation to strengthen the correlation between feature maps in the temporal dimension.
- Prevents information sparsity caused by excessive shifting operations.

### MobileNetV2:
- Uses depthwise separable convolutions and inverted residual blocks to reduce computational complexity while maintaining performance.
- Provides a lightweight backbone for efficient inference.

### Overall Structure:
- The DualCascadeTSF module is placed before each depthwise convolution in the bottleneck layers of MobileNetV2 to balance performance and efficiency.
- For more details, refer to the paper.

## Datasets
The model was evaluated on three publicly available violence-related datasets:

### Crowd Violence:
- Focuses on scenes of crowd violence.
- Contains 246 video clips with durations ranging from 1 to 6.5 seconds.

### RWF-2000:
- A medium-sized dataset containing 2000 surveillance video clips collected from YouTube.
- Includes both violent and non-violent behaviors.

### Hockey Fights:
- Contains 1000 video clips from ice hockey games.
- Records fighting behaviors and normal physical contacts during games.

## Results
The DualCascadeTSF-MobileNetV2 model achieves the following results on the benchmark datasets:

| Dataset        | Accuracy (%) | Parameters (MB) | Memory (MB) | Training Time (min) |
|----------------|--------------|-----------------|-------------|---------------------|
| Crowd Violence | 98.98        | 16.99           | 347.13      | 35.92               |
| RWF-2000       | 88.5         | 16.99           | 347.13      | 35.92               |
| Hockey Fights  | 98.0         | 16.99           | 347.13      | 35.92               |

## Contributions
We welcome contributions to improve the model's performance, optimize its structure, or extend its applications. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a clear description of your modifications.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or feedback, please contact:

Yong Li: lilili819@163.com