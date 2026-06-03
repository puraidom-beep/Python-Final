import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. 基礎路徑與硬體設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASK_MODEL_PATH = os.path.join(BASE_DIR, "best_masked_face_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 【次世代 2026】YOLOv11-Pose 終極監控系統啟動！運算設備: {device}")

# 2. 載入模型大腦
yolo_pose = YOLO("yolo11n-pose.pt").to(device)

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


def main():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
    img_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(valid_extensions)]

    if not img_files:
        print("❌ images 資料夾內沒有圖片！")
        return

    # 隨機挑選一張圖片進行盲測展示
    import random
    target_img_name = random.choice(img_files)
    img_path = os.path.join(IMAGES_DIR, target_img_name)

    image = cv2.imread(img_path)
    if image is None: return
    output_img = image.copy()

    print(f"🎬 YOLOv11 正在進行超感度掃描: {target_img_name}")

    results = yolo_pose(image, verbose=False)[0]

    if len(results.boxes) == 0:
        print("❌ 畫面上未捕捉到任何監控目標。")
        return

    print(f"🎉 YOLOv11 成功捕捉到 {len(results.boxes)} 個監控對象！")

    for i, (box, kpt_obj) in enumerate(zip(results.boxes, results.keypoints)):
        # 1. 取得人體外框
        xyxy = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, xyxy)

        # 2. 取得精準的關鍵點矩陣 (確保只取前 3 個維度: x, y, conf)
        kpts = kpt_obj.data[0].cpu().numpy()
        if len(kpts) < 5: continue

        # 提取黃金五官點
        nose = kpts[0]
        left_eye = kpts[1]
        right_eye = kpts[2]

        # 🔥 核心修正：利用人體框的寬高比例「精準切頭」，再也不會往下噴成大長條！
        person_w = xmax - xmin
        person_h = ymax - ymin

        # 頭部通常佔人體總寬度的 50%，總高度的 20%
        face_w = int(person_w * 0.5)
        face_h = int(person_h * 0.22)

        # 以眼睛或人體頂部為基準點進行安全切圖
        center_x = int((left_eye[0] + right_eye[0]) / 2) if (left_eye[2] > 0.3 and right_eye[2] > 0.3) else int(
            xmin + person_w / 2)

        face_xmin = max(0, center_x - int(face_w / 2))
        face_xmax = min(image.shape[1], center_x + int(face_w / 2))
        face_ymin = max(0, ymin)
        face_ymax = min(image.shape[0], ymin + face_h)

        if (face_xmax - face_xmin) < 10 or (face_ymax - face_ymin) < 10:
            continue

        # --- 第一層：MobileNetV2 大腦預測口罩狀態 ---
        face_roi = image[face_ymin:face_ymax, face_xmin:face_xmax]
        ai_mask_result = "Unknown"
        confidence = 0.0

        if face_roi.size > 0:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            input_tensor = predict_transform(face_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = mask_model(input_tensor)
                _, predicted = outputs.max(1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item() * 100

            class_labels = {0: "WithMask", 1: "WithoutMask"}
            ai_mask_result = class_labels[predicted.item()]

        # --- 第二層：YOLOv11 原生眼部信心度疲勞判定 ---
        is_drowsy = False
        # 只要有一隻眼睛的追蹤信心度過低，就代表閉眼、渙散或打瞌睡
        if left_eye[2] < 0.55 or right_eye[2] < 0.55:
            is_drowsy = True

        # 在眼部、鼻尖畫上科技感實心小圓點
        if left_eye[2] > 0.4: cv2.circle(output_img, (int(left_eye[0]), int(left_eye[1])), 3, (0, 0, 255), -1)
        if right_eye[2] > 0.4: cv2.circle(output_img, (int(right_eye[0]), int(right_eye[1])), 3, (0, 0, 255), -1)
        if nose[2] > 0.4: cv2.circle(output_img, (int(nose[0]), int(nose[1])), 3, (0, 255, 255), -1)

        # --- 綜合視覺化 HUD 呈現 ---
        box_color = (0, 0, 255) if is_drowsy else (0, 255, 0)

        # 繪製精準的人臉外框
        cv2.rectangle(output_img, (face_xmin, face_ymin), (face_xmax, face_ymax), box_color, 2)

        status_str = "DROWSY!!" if is_drowsy else "ACTIVE"
        info_text = f"ID:{i + 1} | {ai_mask_result}({confidence:.0f}%) | Conf:{min(left_eye[2], right_eye[2]):.2f} [{status_str}]"

        # 文字標籤背景
        cv2.rectangle(output_img, (face_xmin, face_ymin - 20), (face_xmax, face_ymin), box_color, -1)
        cv2.putText(output_img, info_text, (face_xmin + 3, face_ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    # 顯示全功能 YOLOv11-Pose DMS 整合成果
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Next-Gen YOLOv11 + Deep Learning Driver Monitoring System")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()