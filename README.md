
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

## Training the model

Use the `train.py` script to train the model. To train the default model on 
CIFAR-10 simply use:

```
python3 train.py
```

You might want to at least change the `--data_dir` and `--save_dir` which
point to paths on your system to download the data to (if not available), and
where to save the checkpoints.

**I want to train on fewer GPUs**. To train on fewer GPUs we recommend using `CUDA_VISIBLE_DEVICES` 
to narrow the visibility of GPUs to only a few and then run the script. Don't forget to modulate
the flag `--nr_gpu` accordingly.

**I want to train on my own dataset**. Have a look at the `DataLoader` classes
in the `data/` folder. You have to write an analogous data iterator object for
your own dataset and the code should work well from there.

## Pretrained model checkpoint

You can download our pretrained (TensorFlow) model that achieves 2.92 bpd on CIFAR-10 [here](http://alpha.openai.com/pxpp.zip) (656MB).

## Citation

If you find this code useful please cite us in your work:

```
@inproceedings{Salimans2017PixeCNN,
  title={PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications},
  author={Tim Salimans and Andrej Karpathy and Xi Chen and Diederik P. Kingma},
  booktitle={ICLR},
  year={2017}
}
```
# pixel-cnn-rotations
