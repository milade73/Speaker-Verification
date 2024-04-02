import librosa
import numpy as np
import pickle

def extract_features_for_signals(signal_list):
    all_features = []
    n=0
    for signal in signal_list:
        y, sr = signal,16000
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc=mfcc.reshape(1,13*127)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma=chroma.reshape(1,12*127)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast=contrast.reshape(1,7*127)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz=tonnetz.reshape(1,6*127)
        rms = librosa.feature.rms(y=y)
        n=n+1
        

        
        
        features = np.concatenate((mfcc, zcr, centroid, bandwidth, rolloff, chroma, contrast, tonnetz, rms),axis=1)
        all_features.append(features)
        print(n)
    return np.array(all_features)


with open('D:/NLP/NLP_dataset/Eqalized_fake_train2.pkl', 'rb') as f:
    y= pickle.load(f)[:710]




Q=extract_features_for_signals(y)


with open('D:/NLP/NLP_dataset/fake_valid_features.pkl','wb') as f:
    pickle.dump(Q,f)





with open('D:/NLP/NLP_dataset/Eqalized_real_train2.pkl', 'rb') as f:
    y= pickle.load(f)[:710]




    
Q_real=extract_features_for_signals(y)    


with open('D:/NLP/NLP_dataset/real_valid_features.pkl','wb') as f:
    pickle.dump(Q_real,f)
