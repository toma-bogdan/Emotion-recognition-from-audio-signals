import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa.display
import librosa.feature
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


def load_data():
    paths = []
    labels = []
    for dirname, _, filenames in os.walk('./TESS Toronto emotional speech set data'):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        if len(paths) == 2800:
            break

    dataframe = pd.DataFrame()
    dataframe['path'] = paths
    dataframe['label'] = labels
    print(dataframe['label'].value_counts())

    return dataframe


def extract_mfccs(dataframe):
    mfccs = []
    X = [librosa.load(path, sr=44000)[0] for path in dataframe['path']]
    for i in tqdm(X):
        mfcc = librosa.feature.mfcc(y=i, sr=44000, n_mfcc=20)
        mfcc = mfcc.T
        mfccs.append(mfcc)

    return mfccs


def extract_emotions(data, sampling_rate, model, emotion="unknown"):
    time = np.arange(0, len(data)) / sampling_rate

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Sound wave of {emotion}')
    plt.show()

    D = np.abs(librosa.stft(data))

    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(DB, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f db')
    plt.show()

    plt.magnitude_spectrum(data, scale='dB')
    plt.show()

    new_mfcc = librosa.feature.mfcc(y=data, sr=44000, n_mfcc=20)
    new_mfcc = new_mfcc.T

    # Show MFCC coefficients
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(new_mfcc, sr=sampling_rate, x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCC for {emotion}')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.show()

    # Pad or truncate sequences to the fixed length
    if len(new_mfcc) < max_length:
        pad_value = np.median(new_mfcc)
        padded_new_mfcc = np.pad(new_mfcc, [(0, max_length - len(new_mfcc)), (0, 0)], mode='constant',
                                 constant_values=pad_value)
    else:
        padded_new_mfcc = new_mfcc[:max_length, :]

    # Convert to NumPy array
    new_data_input = np.array([padded_new_mfcc])
    predictions = model.predict(new_data_input)
    predicted_label_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_label_index]

    print(f"The predicted emotion for the new audio file is: {predicted_emotion}")
    # Plotting the probabilities for each emotion class
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ['blue' if i != predicted_label_index else 'red' for i in range(len(class_labels))]
    bars = ax.barh(class_labels, predictions[0], color=bar_colors)

    # Adding labels and title
    ax.set_xlabel('Probability')
    ax.set_ylabel('Emotion')
    ax.set_title('Emotion Prediction Analysis')

    # Adding text for accuracy and similarity
    similarity_text = 'Similarity to other emotions:\n'
    for label, prob in zip(class_labels, predictions[0]):
        if label != predicted_emotion:
            similarity_text += f'{label}: {prob:.2%}\n'

    # Adding text for accuracy and similarity
    ax.text(0.5, -0.5, similarity_text, ha='center', va='center', transform=ax.transAxes, color='green')

    plt.tight_layout()
    plt.show()


def preprocess_mfccs(mfccs, max_length):
    mfccs_homogeneous = []
    for mfcc in mfccs:
        if len(mfcc) < max_length:
            pad_value = np.median(mfcc)
            padded_mfcc = np.pad(mfcc, [(0, max_length - len(mfcc)), (0, 0)], mode='constant',
                                 constant_values=pad_value)
        else:
            padded_mfcc = mfcc[:max_length, :]

        mfccs_homogeneous.append(padded_mfcc)

    return np.array(mfccs_homogeneous)


if __name__ == '__main__':

    df = load_data()

    mfccs = extract_mfccs(df)

    model = Sequential()
    model.add(layers.Conv1D(256, 5, padding='same',
                            input_shape=(236, 20)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=8))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(7))
    model.add(layers.Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use appropriate loss for your task
                  metrics=['accuracy'])  # Use appropriate metrics
    class_labels = df['label'].unique()
    num_classes = len(class_labels)

    # Map class labels to integers
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    df['label_index'] = df['label'].map(class_to_index)

    # One-hot encode the labels
    y_labels = to_categorical(df['label_index'], num_classes=num_classes)

    max_length = 236  # Set the length of the audio files

    # Pad or truncate sequences to the fixed length
    mfccs_homogeneous = preprocess_mfccs(mfccs, max_length)
    model.fit(np.array(mfccs_homogeneous), np.array(y_labels), epochs=10, batch_size=32)

    # Select a random audio file from data frame
    ind = np.random.randint(0, len(df))
    data, sampling_rate = librosa.load(df['path'][ind], sr=44100)
    emotion = df['label'][ind]

    extract_emotions(data, sampling_rate, model, emotion)

    # Do the same from a random audio file
    data2, sampling_rate2 = librosa.load("./03-01-02-01-01-02-24.wav", sr=44100)
    extract_emotions(data2, sampling_rate2, model)
