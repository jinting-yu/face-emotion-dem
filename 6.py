import os
# 屏蔽TF警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 改用英文标签，彻底解决中文问号乱码
emotion_labels_en = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
img_size = 48
# 模型同目录直接读取
model_path = r"emotion_model.h5"

# 加载模型和人脸检测器
model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 打开摄像头（先试0，再试1）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("无法打开摄像头，请检查设备！")
    exit()

# 降低分辨率，提升识别速度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✅ 摄像头启动成功，按 q 键退出程序")

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 转灰度图用于人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 调整人脸检测参数，适配强光环境
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # 截取人脸ROI
        face_roi = gray[y:y + h, x:x + w]
        # 缩放到模型输入尺寸
        face_resize = cv2.resize(face_roi, (img_size, img_size))
        # 归一化并扩展维度适配模型输入
        face_input = np.expand_dims(face_resize, axis=(0, -1)) / 255.0

        # 模型预测
        pred = model.predict(face_input, verbose=0)[0]
        max_idx = np.argmax(pred)
        emo_text = emotion_labels_en[max_idx]
        conf = round(pred[max_idx] * 100, 2)

        # 人脸框加粗放大显示
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # 文字字号加大、线条加粗
        show_text = f"{emo_text} {conf}%"
        cv2.putText(frame, show_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 创建窗口并设置初始宽高 1000*700，窗口更大
    cv2.namedWindow("Camera Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Detect", 1000, 700)
    cv2.imshow("Camera Detect", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("✅ 程序已正常退出！")