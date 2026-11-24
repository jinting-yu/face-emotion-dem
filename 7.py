import os
# 屏蔽tf警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ========== 1. 配置参数 ==========
csv_path = "fer2013/fer2013.csv"  # 你的csv文件路径
img_size = 48
batch_size = 64
epochs = 30
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ========== 2. 加载并预处理数据 ==========
df = pd.read_csv(csv_path)

x = []
y = []
for idx, row in df.iterrows():
    pixels = list(map(int, row['pixels'].split(" ")))
    x.append(np.array(pixels).reshape(img_size, img_size))
    y.append(row['emotion'])

# 归一化 + 扩展维度
x = np.array(x).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(np.array(y), 7)

# 划分训练/测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ========== 3. 搭建CNN模型 ==========
model = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ========== 4. 训练模型 ==========
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)

# 保存模型
model.save("emotion_model.h5")
print("✅ 模型已保存为 emotion_model.h5")

# ========== 5. 绘制训练/验证折线图（和你截图一致） ==========
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 避免中文乱码
plt.figure(figsize=(16, 8))

# 左图：准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_accuracy'], label='验证准确率', color='#ff7f0e', linewidth=2)
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.3, 0.85)  # 和你截图的y轴范围一致
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# 右图：损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', color='#ff7f0e', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.ylim(0.5, 1.7)  # 和你截图的y轴范围一致
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 折线图已保存为 loss_curve.png")

# ========== 6. 绘制混淆矩阵（可选，用于报告） ==========
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 混淆矩阵已保存为 confusion_matrix.png")