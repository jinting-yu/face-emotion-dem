import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_mfcc(file_path, n_mfcc=10):
    y, sr = librosa.load(file_path, sr=16000, duration=1)  # 只取 1 秒
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

data_dir = "./data/SpeechCommands/speech_commands_v0.02"
X, y = [], []
labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))][:5]  # 只取 5 类
label_to_idx = {label: i for i, label in enumerate(labels)}

max_samples_per_class = 100
for label in labels:
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue
    files = [f for f in os.listdir(label_dir) if f.endswith(".wav")][:max_samples_per_class]
    for file in files:
        file_path = os.path.join(label_dir, file)
        X.append(extract_mfcc(file_path))
        y.append(label_to_idx[label])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"处理完成！共 {len(X)} 条样本，训练集 {len(X_train)}，测试集 {len(X_test)}")