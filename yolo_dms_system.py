import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from ultralytics import YOLO  # 🚀 調用官方 YOLO 核心
import matplotlib.pyplot as plt

# 1. 基礎路徑與硬體設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LBF_MODEL_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")
MASK_MODEL_PATH = os.path.join(BASE_DIR, "best_masked_face_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 超進化 YOLO-DMS 系統啟動！核心運算設備: {device}")

# 2. 載入官方 YOLO 模型 (如果專案裡沒有，它會自動極速下載，完全不會報 404！)
print("📡 正在載入/自動下載 官方輕量化 YOLOv8 權重...")
yolo_detector = YOLO("yolov8n.pt").to(device)

# 載入幾何工具
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL_PATH)

# 載入你親手訓練達 99.7% 的口罩預測大腦
mask_model = models.mobilenet_v2()
num_ftrs = mask_model.classifier[1].in_features
mask_model.classifier[1] = nn.Linear(num_ftrs, 2)
mask_model.load_state_dict(torch.load(MASK_MODEL_PATH, map_location=device))
mask_model = mask_model.to(device).eval()

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
        print("❌ images 資料夾內沒有圖片！")
        return

    # 隨機挑選一張圖片進行展示
    import random
    target_img_name = random.choice(img_files)
    img_path = os.path.join(IMAGES_DIR, target_img_name)

    image = cv2.imread(img_path)
    if image is None: return
    output_img = image.copy()

    print(f"🎬 YOLO 正在強力掃描影像: {target_img_name}")

    # 使用 YOLO 偵測，我們指定 classes=[0] 代表「只抓人類 (Person)」
    results = yolo_detector(image, classes=[0], verbose=False)[0]

    faces = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, xyxy)

        # 💡 高級物理特徵技巧：因為抓的是整個人體，我們利用比例幾何學（Anatomy ratio），
        # 截取人體方框垂直上半部的 25% 區塊，這在人體結構中完美對應「頭部與面部」！
        h_total = ymax - ymin
        head_ymax = ymin + int(h_total * 0.25)
        w = xmax - xmin
        h = head_ymax - ymin

        if w > 10 and h > 10:
            faces.append([xmin, ymin, w, h])

    if len(faces) == 0:
        print("❌ 畫面上未捕捉到任何人臉。")
        return

    print(f"🎉 YOLO 成功捕捉到 {len(faces)} 個監控對象！")
    faces_np = np.array(faces)

    # 轉為灰階提供給 LBF 點點模型
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, landmarks = facemark.fit(gray, faces_np)

    for i, (x, y, w, h) in enumerate(faces):
        # --- 第一層：MobileNetV2 大腦預測口罩狀態 ---
        face_roi = image[max(0, y):min(image.shape[0], y + h), max(0, x):min(image.shape[1], x + w)]
        if face_roi.size == 0: continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        input_tensor = predict_transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = mask_model(input_tensor)
            _, predicted = outputs.max(1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item() * 100

        class_labels = {0: "WithMask", 1: "WithoutMask"}
        ai_mask_result = class_labels[predicted.item()]

        # --- 第二層：幾何公式計算眼部 EAR ---
        ear_text = "EAR: N/A"
        is_drowsy = False

        if success and landmarks is not None and i < len(landmarks):
            pts = landmarks[i][0]
            mean_ear = (calculate_ear(pts[LEFT_EYE_IDX]) + calculate_ear(pts[RIGHT_EYE_IDX])) / 2.0
            ear_text = f"EAR: {mean_ear:.2f}"

            if mean_ear < 0.22:
                is_drowsy = True

            # 點上紅色關鍵點
            for pt in pts[LEFT_EYE_IDX]: cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
            for pt in pts[RIGHT_EYE_IDX]: cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        # --- 科技感十足的視覺化呈現 ---
        box_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), box_color, 2)

        status_str = "DROWSY!!" if is_drowsy else "ACTIVE"
        info_text = f"YOLO_DMS_{i + 1} | {ai_mask_result}({confidence:.1f}%) | {ear_text} [{status_str}]"

        cv2.rectangle(output_img, (x, y - 25), (x + w, y), box_color, -1)
        cv2.putText(output_img, info_text, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 顯示全功能 YOLO-DMS 整合成果
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Next-Gen YOLO + Deep Learning Driver Monitoring System")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()