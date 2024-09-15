### Harmful Brain Activity Classification using EEG Data

This project focuses on detecting harmful brain activity from EEG signals using deep learning models. The dataset comes from the HMS Harmful Brain Activity Classification competition, which contains EEG signals and associated expert labels for various types of harmful brain activities such as seizures, LPD (Lateralized Periodic Discharges), GPD (Generalized Periodic Discharges), LRDA (Lateralized Rhythmic Delta Activity), GRDA (Generalized Rhythmic Delta Activity), and others.

#### Methodology

The approach involves several key steps for processing EEG data and training a deep learning model for classification. The following is an overview of the process:

#### 1. **Data Loading**
EEG data is stored in `.parquet` format, and the accompanying metadata is loaded from a `.csv` file. The code uses the pandas library to handle the metadata and store it in a dataframe.

```python
train = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
```

#### 2. **EEG Signal Preprocessing**
Each EEG file is loaded, and the EEG signals are extracted using a subset of important electrode channels:
- `Fp1`, `T3`, `C3`, `O1`, `Fp2`, `C4`, `T4`, `O2`

The EEG data is clipped to a middle section of 50 seconds to ensure consistent input size. The signal is then converted to a numpy array, and any missing data (NaN values) is handled by replacing them with the channel's mean.

#### 3. **Spectrogram Generation**
A **Short-Time Fourier Transform (STFT)** is applied to convert the EEG signals into the frequency domain, creating spectrograms that represent the signal's frequency components over time.

For each EEG channel, the spectrogram is computed using:

```python
f, t, Sxx = spectrogram(data, fs=200, nperseg=200, noverlap=int(200*0.8), scaling='density', return_onesided=True)
Sxx_db = 10 * np.log10(Sxx)
```

The result is a spectrogram in decibels (dB), which is normalized for each EEG channel to ensure consistency.

#### 4. **Data Augmentation and Normalization**
Before feeding the spectrograms into the model, the data is normalized between 0 and 1 to remove any scale variance:

```python
sample[:, :, 0] = (sample[:, :, 0] - sample[:, :, 0].min()) / (sample[:, :, 0].max() - sample[:, :, 0].min() + epsilon)
```

#### 5. **Model Architecture**
The model architecture is based on **MobileNetV3Small**, a pre-trained deep learning model optimized for lightweight and efficient training. The input spectrogram data is first passed through a 2D convolutional layer to reduce the 8 EEG channels into a 3-channel image-like representation.

```python
inp = tf.keras.Input(shape=(224, 224, 8))
x = tf.keras.layers.Conv2D(3, kernel_size=(3,3), activation='gelu', padding='same')(inp)
```

The MobileNetV3Small model is used as the backbone, followed by a series of dense layers to predict the harmful brain activity labels. The final layer uses a softmax activation function to output probabilities for the six target classes.

#### 6. **Data Generator for Model Training**
A custom `DataGenerator` class is implemented to load and preprocess the data in batches during training. This generator uses the spectrogram transformations to generate input data in a form suitable for the model.

```python
class DataGenerator(tf.keras.utils.Sequence):
    def __getitem__(self, index):
        X, y = self.__data_generation(indexes)
        return X, y
```

#### 7. **Training and Evaluation**
The model is compiled using the Adam optimizer and categorical cross-entropy loss. The data is split into training and validation sets, and the model is trained using the EEG data with real-time data augmentation.

#### Example of Data Visualization
The spectrograms for each EEG channel can be visualized as follows:

```python
plt.pcolormesh(X[0,:,:,0], cmap='jet')
plt.title('EEG Channel 1 Spectrogram')
plt.colorbar(label='Magnitude (dB)')
plt.show()
```

#### Conclusion
This method efficiently transforms raw EEG signals into a format suitable for deep learning models by leveraging frequency-domain representations (spectrograms) and a powerful CNN-based architecture (MobileNetV3). The model is designed to detect harmful brain activities, such as seizures, from EEG data with high accuracy.

#### Files
- `train.csv`: Metadata for the EEG signals.
- `train_eegs/*.parquet`: Raw EEG data files.
- `EEG processing and model training scripts`: Code for processing EEG data, generating spectrograms, and training the deep learning model.

#### Dependencies
- `tensorflow`
- `scipy`
- `numpy`
- `pandas`
- `matplotlib`

This project serves as an example of how EEG signals can be processed and classified using deep learning techniques for medical applications.
