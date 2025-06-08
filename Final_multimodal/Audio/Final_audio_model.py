# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import re
import torch
import pandas as pd
import numpy as np
from pydub import AudioSegment
from torch import Tensor
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from typing import Dict
import soundfile as sf
import torchaudio
import librosa
import ffmpeg
from sklearn.preprocessing import StandardScaler
import pickle

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def modified_mel_filterbank(sr, n_fft, n_mels, fmin, fmax, high_res_start):
    mel_filter_low = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=high_res_start)
    n_filters_low = mel_filter_low.shape[0]

    mel_filter_high = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filters_low, fmin=high_res_start, fmax=fmax)

    mel_filter_combined = np.vstack((mel_filter_low, mel_filter_high))

    return mel_filter_combined


def lfcc(audio_path, n_lfcc=80, n_fft=512, hop_length=160, sr=16000, fmax=8000, high_res_start=4000):
    y = audio_path
    sr = 16000

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    mel_filter = modified_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_lfcc, fmin=0, fmax=fmax, high_res_start=high_res_start)

    mel_S = np.dot(mel_filter, S)

    log_mel_S = np.log(mel_S + 1e-10)

    lfcc_features = librosa.feature.mfcc(S=log_mel_S, sr=sr, n_mfcc=n_lfcc, dct_type=2)

    return lfcc_features


def spectogram(signal):
    spectogram_signal = []

    for i in range(signal.shape[0]):
        Specto = lfcc(signal[i], n_lfcc=80, n_fft=512, hop_length=160, sr=16000, fmax=8000, high_res_start=4000)
        spectogram_signal.append(Specto)

    return spectogram_signal


# Load the scaler
scaler = StandardScaler()

# Load the saved scaler parameters from the file
#with open('E:/NLP/NLP_voice_model/minmax_scaler.pkl', 'rb') as file:
#    scaler = pickle.load(file)

batch_size = 16


# Train section
def Reshape(Eqalized_fake_train, Eqalized_Real_train):
    Eqalized_fake_train = np.array(Eqalized_fake_train)
    Eqalized_fake_train = scaler.transform(Eqalized_fake_train)

    Eqalized_Real_train = np.array(Eqalized_Real_train)
    Eqalized_Real_train = scaler.transform(Eqalized_Real_train)

    Spectogram_fake_train = spectogram(Eqalized_fake_train)
    Spectogram_real_train = spectogram(Eqalized_Real_train)

    Spectogram_fake_train = np.array(Spectogram_fake_train)
    Spectogram_real_train = np.array(Spectogram_real_train)

    Spectogram_real_train = np.reshape(Spectogram_real_train, [np.shape(Spectogram_real_train)[0], 1, np.shape(Spectogram_real_train)[1], np.shape(Spectogram_real_train)[2]])
    Spectogram_fake_train = np.reshape(Spectogram_fake_train, [np.shape(Spectogram_fake_train)[0], 1, np.shape(Spectogram_fake_train)[1], np.shape(Spectogram_fake_train)[2]])

    del Eqalized_fake_train
    del Eqalized_Real_train

    input_signal_train = np.concatenate((Spectogram_fake_train, Spectogram_real_train))

    Y1_train = np.zeros(np.shape(Spectogram_fake_train)[0])
    Y2_train = np.zeros(np.shape(Spectogram_real_train)[0]) + 1
    Y_train = np.concatenate((Y1_train, Y2_train))
    Y_train = Y_train.reshape((-1, 1))

    input_signal_train = Tensor(input_signal_train)
    Y_train = Tensor(Y_train)

    # Train
    from torch.utils.data import TensorDataset, DataLoader

    dataset_train = TensorDataset(input_signal_train, Y_train)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    return train_loader

import torch.nn as nn



import torch.nn as nn


class Residual_block2D(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class SpecRNet(nn.Module):
    def __init__(self, d_args, **kwargs):
        super().__init__()

        self.device = kwargs.get("device", "cuda")

        self.first_bn = nn.BatchNorm2d(num_features=d_args["filts"][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(
            Residual_block2D(nb_filts=d_args["filts"][1], first=True)
        )
        self.block2 = nn.Sequential(Residual_block2D(nb_filts=d_args["filts"][2]))
        d_args["filts"][2][0] = d_args["filts"][2][1]
        self.block4 = nn.Sequential(Residual_block2D(nb_filts=d_args["filts"][2]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=d_args["filts"][1][-1], l_out_features=d_args["filts"][1][-1]
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1], l_out_features=d_args["filts"][2][-1]
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1], l_out_features=d_args["filts"][2][-1]
        )

        self.bn_before_gru = nn.BatchNorm2d(num_features=d_args["filts"][2][-1])
        self.gru = nn.GRU(
            input_size=d_args["filts"][2][-1],
            hidden_size=120,
            num_layers=d_args["nb_gru_layer"],
            batch_first=True,
            bidirectional=True,
        )

        self.fc1_gru = nn.Linear(
            in_features=120 * 2, out_features=d_args["nb_fc_node"] * 2
        )

        self.fc2_gru = nn.Linear(
            in_features=d_args["nb_fc_node"] * 2,
            out_features=2,
            bias=True,
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)
        y0 = y0.unsqueeze(-1)
        x = x0 * y0 + y0

        x = nn.MaxPool2d(2)(x)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)
        y2 = y2.unsqueeze(-1)
        x = x2 * y2 + y2

        x = nn.MaxPool2d(2)(x)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)
        y4 = y4.unsqueeze(-1)
        x = x4 * y4 + y4

        x = nn.MaxPool2d(2)(x)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.squeeze(-2)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)

        return x

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        l_fc.append(nn.Linear(in_features=in_features, out_features=l_out_features))
        return nn.Sequential(*l_fc)

from typing import Dict


def config(input_channels: int) -> Dict:
    return {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }

import time
from typing import Tuple

import tqdm
import torch
import numpy as np


def benchmarks(model) -> int:
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params



def get_inference_durations_on_rand_frontend(
    model,
    samples_count: int = 1000,
    input_shape: Tuple = (1, 1, 300, 300),
    device: str = "cpu",
) -> np.ndarray:
    durations = []
    for _ in tqdm.tqdm(range(samples_count)):
        random_input = torch.rand(input_shape, device=device)
        start = time.time()
        model(random_input)
        end = time.time()
        durations.append(end - start)
    return np.array(durations)



if __name__ == "__main__":
    import benchmarks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specrnet_config = config(input_channels=1)

    model = SpecRNet(specrnet_config, device=device)
    model = model.to(device)

    # Calculate number of parameters
    parameters_count = benchmarks.count_parameters(model)
    print(f"SpecRNet is composed of: {parameters_count} parameters.")

    batch_size = 16
    input_shape = (batch_size, 1, 80, 404)

    # Check inference times
    durations = benchmarks.get_inference_durations_on_rand_frontend(
        model=model,
        samples_count=100,
        input_shape=input_shape,
        device=device,
    )
    print(
        f"Time benchmark for batch size: {batch_size}\n",
        f"min: {durations.min()}, max: {durations.max()}, std: {durations.std()}, avg: {durations.mean()}",
    )

# Training
num_epochs = 40
batch_size = 16
initial_learning_rate = 0.000001
d_args = config(1)

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
mymodel = SpecRNet(d_args)
mymodel.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=initial_learning_rate, weight_decay=0.1)

# Training variables
train_losses = []
val_losses = []
val_accuracies = []
best_loss = float('inf')
l1_lambda = 0.001

# Best accuracy tracking
best_accuracy_diff = float('inf')
best_accuracy_avg = 0.0
best_diff = 0.4



import re
import numpy as np
import soundfile as sf
import librosa
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pickle

def read_flac_files(path, k):
    flac_files = [f for f in os.listdir(path) if f.endswith('.sph') or f.endswith('.mp3') or f.endswith('.wav')]

    # Function to extract the numerical part of the filename for sorting
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    # Sort files based on the extracted number
    flac_files.sort(key=extract_number)

    num_files = min(1, np.inf)
    n = -1
    sr = []
    matrix = []

    for i in range(num_files):
        file_path = os.path.join(path, flac_files[k])
        n += 1
        print(n)
        data, s = sf.read(file_path)
        sr.append(s)

        if data.ndim > 1 and data.shape[1] > 1:
            data = data[:, 0]
        matrix.append(data)

    del flac_files, num_files
    return matrix, sr

def training_resize(signal):
    matrix_real_train = []
    matrix_fake_train = []

    alpha = len(signal)
    alpha = alpha / 2
    signal = np.concatenate(signal)

    signal_audio_train = []

    for i in range(int(len(signal) / 64600)):
        V1_signal = signal[64600 * i:64600 * (i + 1)]
        signal_audio_train.append(V1_signal)

    del matrix_real_train
    return signal_audio_train

def resample_signals(signal_list, sr):
    resampled_signals = []
    N = []
    
    for i in range(len(signal_list)):
        signal = signal_list[i]
        s = sr[i]
        resampled_signal = librosa.resample(signal, orig_sr=s, target_sr=16000)
        resampled_signals.append(resampled_signal)
        num = int(len(resampled_signal) / 64600)
        N.append(num)
    
    N = np.array(N)
    return resampled_signals, N

def pre_process_speech(path, k):
    fake_audio_train, sr = read_flac_files(path, k)
    resampled_signals, N = resample_signals(fake_audio_train, sr)
    data_preprocessed = training_resize(resampled_signals)
    return data_preprocessed, N

def evaluate_model(k, mymodel, device, path, model_path='Best5_model.pth'):
    # Training parameters
    num_epochs = 40
    batch_size = 16
    initial_learning_rate = 0.000001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mymodel.parameters(), lr=initial_learning_rate, weight_decay=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mymodel.to(device)

    # Load the pre-trained model
    mymodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    mymodel.eval()

    # Prepare test data
    base_dir = os.getcwd()


    min_max = os.path.join(base_dir, "Audio", "weights", "minmax_scaler.pkl")
    with open(min_max, 'rb') as file:
        scaler = pickle.load(file)

    Data_test, N = pre_process_speech(path, k)
    Eqalized_test = np.array(copy.copy(Data_test))
    Eqalized_test = scaler.transform(Eqalized_test)

    Spectogram_test = spectogram(Eqalized_test)
    Spectogram_test = np.array(Spectogram_test)
    Spectogram_test = np.reshape(Spectogram_test, [np.shape(Spectogram_test)[0], 1, np.shape(Spectogram_test)[1], np.shape(Spectogram_test)[2]])

    Y_test = np.zeros(np.shape(Spectogram_test)[0])
    Y_test = Y_test.reshape((-1, 1))
    input_signal_test = torch.Tensor(Spectogram_test)
    Y_test = torch.Tensor(Y_test)

    dataset_test = TensorDataset(input_signal_test, Y_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    test_losses = []
    test_accuracies = []
    test_loss = 0
    test_correct = 0
    test_total = 0
    result = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device).squeeze()
            outputs = mymodel(inputs)

            _, predicted = torch.max(outputs.data, 1)
            result.append(F.softmax(outputs, dim=1))

    return result, N


import re
import copy
import numpy as np

def process_results(k, path, model_path):
    result, N = evaluate_model(k, mymodel, device, path, model_path)
    Results = []
    
    for out in result:
        out = out.cpu()
        out = np.array(out)
        Results.append(out)
    
    Z = np.concatenate(Results)
    P = []
    S = []
    s = 0

    for i in range(len(N)):
        if i == 0:
            Q = copy.copy(Z)
            prob = np.mean(Q[0:int(N[i])], axis=0)
            score = Q[0:int(N[i])]
            score[score >= 0.5] = 1
            score[score < 0.5] = 0
            s1 = np.sum(score[:, 0])
            s2 = np.sum(score[:, 1])
            score = np.zeros(2)
            S_fake = copy.copy(s1)
            S_real = copy.copy(s2)

            if s1 >= s2:
                s1 = 1
                s2 = 0
            else:
                s1 = 0
                s2 = 1
            
            score[0] = s1
            score[1] = s2
            s += N[i]
        else:
            c = int(s)
            Q = copy.copy(Z)
            prob = np.mean(Q[c:c + N[i]], axis=0)
            score = Q[c:c + N[i]]
            score[score >= 0.5] = 1
            score[score < 0.5] = 0
            s1 = np.sum(score[:, 0])
            s2 = np.sum(score[:, 1])
            S_fake = copy.copy(s1)
            S_real = copy.copy(s2)
            score = np.zeros(2)

            if s1 >= s2:
                s1 = 1
                s2 = 0
            else:
                s1 = 0
                s2 = 1

            score[0] = s1
            score[1] = s2
            s += N[i]
        
        P.append(prob)
        S.append(score)

    P = np.array(P)
    S = np.array(S)

    return P, S, S_real, S_fake


import re


import re

def process_all_files(path):
    # Determine if the path is a file or directory
    if os.path.isfile(path):
        # Only one file to process
        flac_files = [os.path.basename(path)]
        folder_path = os.path.dirname(path)
    else:
        # Process all supported audio files in the directory
        folder_path = path
        flac_files = [f for f in os.listdir(folder_path)
                      if f.endswith('.sph') or f.endswith('.mp3') or f.endswith('.ogg')]

        # Sort files numerically if they have digits
        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else float('inf')

        flac_files.sort(key=extract_number)

    # Initialize result lists
    S_totall = []
    P_totall = []

    # Process each file
    for i, file_name in enumerate(flac_files):
        base_dir = os.getcwd()
        model_path = os.path.join(base_dir, "Audio", "weights", "best4_model.pth")
#        model_path = r"E:\NLP\Final_multi_modal\Audio\best4_model.pth"
        full_file_path = os.path.join(folder_path, file_name)

        P, S, S_real, S_fake = process_results(i, folder_path, model_path)
        S_totall.append(S[0])
        P_totall.append(P)

    return P_totall, S_totall


# Example usage
#path = r"C:\Users\Milad Esmaeil-Zadeh\Downloads\audio-1745951291431.mp3"
#P_totall, S_totall = process_all_files(path)













