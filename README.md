
**Status:** Archive (code is provided as-is, no updates expected)


**Fusion-Based Framework for Emotion Recognition Using Wearable Sensors**, by
Omer Tariq; Yookyung Oh; Dongsoo Han

[Omer Tariq et al.](https://ieeexplore.ieee.org/document/10784695) 
Wearable sensor technology emerges significantly in sensing physiological data for real-time health monitoring, showing promise for early disease detection by continuously monitoring vital signs like body temperature, heart rate, and electrodermal activity. However, current emotion recognition methods are limited by high measurement noise in wearable health sensors, negatively impacting performance. Additionally, most classification models are too complex to be deployed on IoT devices for real-time performance. This paper presents EmoHeal, a lightweight deep learning-based fusion architecture to process wrist sensor-based physiological sensor data for IoT devices. EmoHeal integrates data augmentation, attention mechanisms, Temporal Convolutional Networks (TCNs), and Gated Recurrent Units (GRUs) for robust emotion classification. Each sensor's data is processed through a separate TCN block that extracts temporal features using dilated convolutions. These blocks are enhanced by channel and spatial attention mechanisms to emphasize significant features. The extracted features are then fused and fed into a bidirectional GRU, which captures long-term dependencies in the wrist sensor data. EmoHeal demonstrated excellent performance on the K-EmoCon dataset, achieving a mean accuracy of ≈72% arousal for an edge-deployable emotion recognition model, surpassing state-of-the-art models. This research significantly benefits real-time mobile health sensing in Internet of Things (IoT) applications.

![image](https://github.com/user-attachments/assets/cbed8252-f438-4e72-adf2-f4d3a7c05385)


## Setup

To run this code you need the following:

- a machine with GPU
- Python3
- Numpy, TensorFlow and other packages:
```
```

Here is the PyTorch implementation of the EmoHEAL model. EmoHEAL is a lightweight deep learning architecture that uses wrist sensor data and achieves 72.35% accuracy on the K-EmoCon dataset, making it suitable for IoT applications in smartwatches.

# Architectures
1. TCN+CA-SA+GRU Fusion
2. TCN+GRU Fusion
3. TCN+xLSTM
4. TCN+MHA
5. TCN+Transformer Encoder
6. ResNet + GRU Fusion

# Dataset
You can request the dataset from [K-Emocon](https://zenodo.org/records/3931963). Place the dataset in the `dataset` folder.

## Citation

If you find this code useful please cite us in your work:

```
@INPROCEEDINGS{10784695,
  author={Tariq, Omer and Oh, Yookyung and Han, Dongsoo},
  booktitle={2024 IEEE SENSORS}, 
  title={EmoHEAL: A Fusion-Based Framework for Emotion Recognition Using Wearable Sensors}, 
  year={2024},
  volume={},
  number={},
  pages={1-4},
  keywords={Performance evaluation;Wrist;Temperature measurement;Emotion recognition;Feature extraction;Real-time systems;Sensors;Internet of Things;Biomedical monitoring;Wearable sensors;Sensor Fusion;Emotion Recognition;AIoT;Temporal Convolution;Affective Computing;Deep Learning},
  doi={10.1109/SENSORS60989.2024.10784695}}

```
# EmoHEAL
