import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- 1. 配置与工具函数 ----------------------
# 数据集路径（和你本地一致）
DATA_DIR = "./data/SpeechCommands/speech_commands_v0.02"
# 只取前5个类别，提速
SELECT_LABELS = 5
# 每个类别最多取100条样本
MAX_SAMPLES_PER_CLASS = 100
# MFCC特征维度
N_MFCC = 10

def extract_mfcc(file_path, n_mfcc=N_MFCC):
    """提取音频MFCC特征（优化提速）"""
    y, sr = librosa.load(file_path, sr=16000, duration=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# ---------------------- 2. 批量提取特征（原3.py） ----------------------
print("🔄 开始批量提取MFCC特征...")
X, y = [], []
labels = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))][:SELECT_LABELS]
label_to_idx = {label: i for i, label in enumerate(labels)}

for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    files = [f for f in os.listdir(label_dir) if f.endswith(".wav")][:MAX_SAMPLES_PER_CLASS]
    for file in files:
        file_path = os.path.join(label_dir, file)
        X.append(extract_mfcc(file_path))
        y.append(label_to_idx[label])
        if len(X) % 200 == 0:
            print(f"已处理 {len(X)} 条音频...")

# 数据集划分与标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"✅ 特征提取完成！总样本数: {len(X)}, 训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ---------------------- 3. CNN模型定义与训练（原4.py） ----------------------
class SpeechCNN(nn.Module):
    def __init__(self, input_dim=N_MFCC, num_classes=len(labels)):
        super(SpeechCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * ((input_dim - 4) // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SpeechCNN(input_dim=N_MFCC, num_classes=len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 训练循环
print("\n🚀 开始训练模型...")
loss_history = []
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    print(f"Epoch {epoch+1:2d}, Loss: {loss.item():.4f}")

# ---------------------- 4. 模型评估 & 弹窗绘图（原5.py，弹窗不保存） ----------------------
print("\n📊 开始模型评估...")
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).argmax(dim=1).numpy()

# 输出指标
acc = accuracy_score(y_test, y_pred)
print(f"🎯 测试集准确率: {acc:.4f}")
print("\n📋 分类报告:")
print(classification_report(y_test, y_pred, target_names=labels))

# 1. Loss曲线弹窗
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.show()  # 弹窗弹出，不保存文件

# 2. 混淆矩阵弹窗
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()  # 弹窗弹出

# 3. 音频波形弹窗
if len(X_test) > 0:
    test_file = os.path.join(DATA_DIR, labels[0], os.listdir(os.path.join(DATA_DIR, labels[0]))[0])
    y_wav, sr_wav = librosa.load(test_file, sr=16000)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y_wav, sr=sr_wav)
    plt.title(f"Audio Waveform (Label: {labels[0]})")
    plt.tight_layout()
    plt.show()  # 弹窗弹出

print("\n🎉 全流程运行完成！所有图表已弹窗展示")