# GD-TSF-MobileNetV2: Group-Shared Differential Temporal Shift Fusion for Lightweight Violence Recognition

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
Violence recognition is an important task in intelligent video surveillance and public safety. Violent behaviors usually involve short-term bursts and strong local motion variations, so effectively capturing dynamic changes between consecutive frames with low computational cost remains a key challenge.

This repository contains the implementation of GD-TSF-MobileNetV2, a lightweight deep learning model that addresses this challenge. The model introduces GD-TSF (Group-shared Differential Temporal Shift Fusion), a novel temporal modeling module built on a lightweight temporal shift framework. GD-TSF incorporates first-order feature differences between consecutive frames as a high-frequency dynamic enhancement, using forward and backward differential branches to capture temporal dynamics from past and future frames, while preserving identity mapping for stable spatial semantics. A group-shared softmax fusion mechanism adaptively combines differential, temporal, and spatial information. The GD-TSF module is integrated into MobileNetV2 to form GD-TSF-MobileNetV2, achieving improved accuracy with low parameter and computational overhead, making it ideal for deployment in resource-constrained scenarios such as edge devices and real-time surveillance systems.

## Key Features
- **First-Order Differential Enhancement**: Introduces first-order feature differences between consecutive frames to capture high-frequency dynamic changes, effectively modeling short-term bursts and strong local motion variations in violent behaviors.
- **Forward and Backward Differential Branches**: Captures temporal dynamics from both past and future frames, providing bidirectional motion perception.
- **Identity Mapping**: Preserves stable spatial semantics alongside temporal modeling, preventing information loss during temporal shift operations.
- **Group-Shared Softmax Fusion**: Adaptively combines differential, temporal, and spatial information through a lightweight, learnable fusion mechanism.
- **Lightweight Design**: Built on the MobileNetV2 backbone with low parameter and computational overhead, achieving improvements of 1.02, 0.50, and 1.00 percentage points on Crowd Violence, Hockey Fights, and RWF-2000 respectively over MobileNet-TSM.
- **Efficient Deployment**: Optimized for resource-constrained environments, such as edge devices and real-time surveillance systems.

## Model Architecture
The GD-TSF-MobileNetV2 model integrates the following components:

### GD-TSF (Group-shared Differential Temporal Shift Fusion) Module:
- **Temporal Shift Backbone**: Built on a lightweight temporal shift framework that efficiently exchanges information across adjacent frames without expensive 3D convolutions.
- **Forward and Backward Differential Branches**: Computes first-order feature differences between consecutive frames along the temporal dimension, capturing high-frequency dynamic cues from both past and future directions.
- **Identity Mapping Branch**: Retains the original spatial features to preserve stable semantic information and prevent degradation during temporal modeling.
- **Group-Shared Softmax Fusion**: A lightweight attention mechanism that adaptively weights and fuses the differential, temporal, and spatial features. The "group-shared" design shares fusion weights across feature groups, reducing parameters while maintaining fusion quality.

### MobileNetV2 Backbone:
- Uses depthwise separable convolutions and inverted residual blocks to reduce computational complexity while maintaining representational capacity.
- Provides a lightweight backbone suitable for resource-constrained deployment.

### Overall Structure:
- The GD-TSF module is integrated into the bottleneck layers of MobileNetV2, replacing standard temporal shift operations to enhance the model's ability to perceive violent motion dynamics.
- For more details, refer to the paper.

## Datasets
The model was evaluated on three publicly available violence recognition benchmarks:

### Crowd Violence:
- Focuses on scenes of crowd violence in public spaces.
- Contains 246 video clips with durations ranging from 1 to 6.5 seconds, captured under varying illumination and scene conditions.

### Hockey Fights:
- Contains 1000 video clips from ice hockey games.
- Includes both fighting behaviors and normal physical contacts during games, posing a challenging fine-grained action recognition task.

### RWF-2000:
- A medium-scale dataset containing 2000 surveillance video clips collected from YouTube.
- Covers diverse real-world scenarios with both violent and non-violent behaviors, making it representative of practical surveillance applications.

## Results
The DualCascadeTSF-MobileNetV2 model achieves the following results on the benchmark datasets:

| Dataset        | Accuracy (%) | Parameters (MB) | Memory (MB) | Training Time (min) |
|----------------|--------------|-----------------|-------------|---------------------|
| Crowd Violence | 98.98        | 16.99           | 347.13      | 42.59               |
| RWF-2000       | 88.75         | 16.99           | 347.13      | 42.59               |
| Hockey Fights  | 98.0         | 16.99           | 347.13      | 42.59               |

## Contributions
We welcome contributions to improve the model's performance, optimize its structure, or extend its applications. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a clear description of your modifications.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or feedback, please contact:

Yong Li: liyong@alumni.nudt.edu.cn