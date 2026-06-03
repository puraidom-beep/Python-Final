import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt

# 1. 基礎路徑與硬體設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
LBF_MODEL_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "best_masked_face_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 終極 DMS 系統啟動！正在調用核心設備: {device}")

# 載入 OpenCV 檢測器
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL_PATH)

# 載入親手訓練的 99.7% 口罩預測 AI 大腦
model = models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2類別: WithMask, WithoutMask
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# 影像預處理標準
predict_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 47 + 1))


def calculate_ear(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C) if C != 0 else 0


def main():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
    img_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(valid_extensions)]

    if not img_files:
        print("❌ 錯誤：images 資料夾內沒有圖片！")
        return

    # 隨機挑選一張圖片進行雙層 DMS 監控展示
    import random
    target_img_name = random.choice(img_files)
    img_path = os.path.join(IMAGES_DIR, target_img_name)

    image = cv2.imread(img_path)
    if image is None: return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    output_img = image.copy()
    print(f"🎬 系統正在實時掃描影像: {target_img_name}")

    if len(faces) == 0:
        print("❌ 畫面上未捕捉到面部特徵。")
        return

    # 叫特徵點模型擬合 68 個關鍵點
    success, landmarks = facemark.fit(gray, faces)

    for i, (x, y, w, h) in enumerate(faces):
        # --- 第一層：AI 大腦預測口罩狀態 ---
        face_roi = image[max(0, y):min(image.shape[0], y + h), max(0, x):min(image.shape[1], x + w)]
        if face_roi.size == 0: continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        input_tensor = predict_transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item() * 100

        class_labels = {0: "WithMask", 1: "WithoutMask"}
        ai_mask_result = class_labels[predicted.item()]

        # --- 第二層：幾何公式計算眼部 EAR 狀態 ---
        ear_text = "EAR: N/A"
        is_drowsy = False

        if success and landmarks is not None:
            pts = landmarks[i][0]
            mean_ear = (calculate_ear(pts[LEFT_EYE_IDX]) + calculate_ear(pts[RIGHT_EYE_IDX])) / 2.0
            ear_text = f"EAR: {mean_ear:.2f}"

            # 疲勞判定門檻
            if mean_ear < 0.22:
                is_drowsy = True

            # 在眼部點上紅色特徵點
            for pt in pts[LEFT_EYE_IDX]: cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
            for pt in pts[RIGHT_EYE_IDX]: cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        # --- 綜合視覺化呈現 ---
        # 根據是否疲勞與口罩狀態，改變方框顏色
        box_color = (0, 0, 255) if is_drowsy else (0, 255, 0)  # 疲勞亮紅框，安全亮綠框
        cv2.rectangle(output_img, (x, y), (x + w, y + h), box_color, 2)

        # 標籤文字
        status_str = "DROWSY!!" if is_drowsy else "ACTIVE"
        info_text = f"{ai_mask_result}({confidence:.1f}%) | {ear_text} [{status_str}]"

        # 繪製背景底框讓文字更清晰
        cv2.rectangle(output_img, (x, y - 25), (x + w, y), box_color, -1)
        cv2.putText(output_img, info_text, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # 顯示全功能 DMS 整合成果
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Dual-Layer Driver Monitoring System (DMS) Result")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()