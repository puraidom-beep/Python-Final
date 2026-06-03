import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import models, transforms
from ultralytics import YOLO

# 1. 基礎パスとハードウェア設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_MODEL_PATH = os.path.join(BASE_DIR, "best_masked_face_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 【次世代 2026】YOLOv11-Pose 究極DMSシステム起動！ 演算デバイス: {device}")

# 2. AI大脳のロード
print("🧠 YOLOv11-Poseモデルを初期化中...")
yolo_pose = YOLO("yolo11n-pose.pt").to(device)

print("🧠 マスク判定モデルを初期化中...")
mask_model = models.mobilenet_v2()
num_ftrs = mask_model.classifier[1].in_features
mask_model.classifier[1] = nn.Linear(num_ftrs, 2)
mask_model.load_state_dict(torch.load(MASK_MODEL_PATH, map_location=device))
mask_model = mask_model.to(device).eval()

# 画像前処理の標準化
predict_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')

    # 🌟 探索対象のディレクトリ（Testとimagesの両方のサブフォルダをすべて探します）
    SEARCH_DIRS = [
        os.path.join(BASE_DIR, "Test"),
        os.path.join(BASE_DIR, "images")
    ]

    img_paths = []

    # フォルダの中のサブフォルダ（WithMask/WithoutMaskなど）まで徹底的に画像を探す
    for search_dir in SEARCH_DIRS:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith(valid_extensions):
                        img_paths.append(os.path.join(root, f))

    if not img_paths:
        print("❌ 指定されたディレクトリ内に画像が見つかりません！")
        return

    print(f"📦 合計 {len(img_paths)} 枚のテスト画像を検出しました！")

    # リストをシャッフル
    random.shuffle(img_paths)

    # 🌟 テスト実行ループ
    for img_path in img_paths:
        image = cv2.imread(img_path)
        if image is None: continue
        output_img = image.copy()

        target_img_name = os.path.basename(img_path)
        print(f"🎬 YOLOv11 スキャン中: {target_img_name}")

        results = yolo_pose(image, verbose=False)[0]

        if len(results.boxes) > 0:
            for i, (box, kpt_obj) in enumerate(zip(results.boxes, results.keypoints)):
                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)

                kpts = kpt_obj.data[0].cpu().numpy()
                if len(kpts) < 5: continue

                nose = kpts[0]
                left_eye = kpts[1]
                right_eye = kpts[2]

                person_w = xmax - xmin
                person_h = ymax - ymin
                face_w = int(person_w * 0.5)
                face_h = int(person_h * 0.22)

                center_x = int((left_eye[0] + right_eye[0]) / 2) if (left_eye[2] > 0.3 and right_eye[2] > 0.3) else int(
                    xmin + person_w / 2)

                face_xmin = max(0, center_x - int(face_w / 2))
                face_xmax = min(image.shape[1], center_x + int(face_w / 2))
                face_ymin = max(0, ymin)
                face_ymax = min(image.shape[0], ymin + face_h)

                if (face_xmax - face_xmin) < 10 or (face_ymax - face_ymin) < 10:
                    continue

                # --- 第1層：MobileNetV2によるマスク判定 ---
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

                # --- 第2層：YOLOv11ネイティブの眼部信頼度による疲労判定 ---
                is_drowsy = False
                if left_eye[2] < 0.55 or right_eye[2] < 0.55:
                    is_drowsy = True

                if left_eye[2] > 0.4: cv2.circle(output_img, (int(left_eye[0]), int(left_eye[1])), 3, (0, 0, 255), -1)
                if right_eye[2] > 0.4: cv2.circle(output_img, (int(right_eye[0]), int(right_eye[1])), 3, (0, 0, 255),
                                                  -1)
                if nose[2] > 0.4: cv2.circle(output_img, (int(nose[0]), int(nose[1])), 3, (0, 255, 255), -1)

                box_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
                cv2.rectangle(output_img, (face_xmin, face_ymin), (face_xmax, face_ymax), box_color, 2)

                status_str = "DROWSY!!" if is_drowsy else "ACTIVE"
                info_text = f"ID:{i + 1} | {ai_mask_result}({confidence:.0f}%) | Conf:{min(left_eye[2], right_eye[2]):.2f} [{status_str}]"

                cv2.rectangle(output_img, (face_xmin, face_ymin - 20), (face_xmax, face_ymin), box_color, -1)
                cv2.putText(output_img, info_text, (face_xmin + 3, face_ymin - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # 🌟 あなた専用の手動コントロール魔法 🌟
        cv2.imshow("Next-Gen YOLOv11 DMS", output_img)

        # ウィンドウの「バツ（X）ボタン」が押されるか、キーが押されるまでずっと待機します
        while cv2.getWindowProperty("Next-Gen YOLOv11 DMS", cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(50)

            # 'q' キーか 'Esc' キーを押すと、プログラム全体を安全に終了します
            if key == ord('q') or key == 27:
                print("⏹️ テストを完全に終了します。")
                cv2.destroyAllWindows()
                return

                # エンターやスペースなど、他のキーを押しても「次の画像」へ進めます
            elif key != -1:
                break

    # 全ての画像を見終わった後にウィンドウを閉じる
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()