import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from tensorflow.keras.utils import to_categorical

# 读取FER2013数据集
df = pd.read_csv("fer2013/fer2013.csv")
x = []
y = []
for idx,row in df.iterrows():
    pixels = list(map(int, row['pixels'].split(" ")))
    x.append(np.array(pixels).reshape(48,48))
    y.append(row['emotion'])
# 数据归一化与维度扩充
x = np.array(x).reshape(-1,48,48,1)/255.0
y = to_categorical(np.array(y),7)
# 划分训练、测试集
split_idx = int(len(x)*0.8)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 搭建CNN
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(48,48,1)))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=50,validation_data=(x_test,y_test))
# 保存模型
model.save("emotion_model.h5")
print("模型训练完成并保存")