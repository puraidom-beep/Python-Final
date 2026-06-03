import os
import cv2
import numpy as np

# 1. 設定基礎路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
LBF_MODEL_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")

# 2. 載入檢測器
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL_PATH)

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 47 + 1))


def calculate_ear(eye_pts):
    """計算 EAR 公式"""
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C) if C != 0 else 0


def main():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
    img_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(valid_extensions)]

    if not img_files:
        print(f"❌ 錯誤：'{IMAGES_DIR}' 資料夾內沒有任何圖片！")
        return

    print(f"🔍 進入連續偵測模式：共計 {len(img_files)} 張圖片")
    print("💡 操作提示：在圖片視窗上按下【任意鍵】可切換到下一張；按【ESC】鍵可直接退出程式。")

    # 🔥 核心修正：使用迴圈巡覽所有圖片
    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 演算法自主尋找人臉
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        output_img = image.copy()

        if len(faces) > 0:
            success, landmarks = facemark.fit(gray, faces)
            if success and landmarks is not None:
                for i, landmark in enumerate(landmarks):
                    pts = landmark[0]
                    mean_ear = (calculate_ear(pts[LEFT_EYE_IDX]) + calculate_ear(pts[RIGHT_EYE_IDX])) / 2.0
                    status = "Closed" if mean_ear < 0.22 else "Open"

                    # 畫出人臉框與紅點
                    x, y, w, h = faces[i]
                    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    for pt in pts[LEFT_EYE_IDX]: cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                    for pt in pts[RIGHT_EYE_IDX]: cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

                    cv2.putText(output_img, f"Face_{i + 1} EAR:{mean_ear:.2f} [{status}]", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 在畫面上印出當前進度
        cv2.putText(output_img, f"Progress: {idx + 1}/{len(img_files)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 顯示 OpenCV 視窗（連續看圖必須用 cv2.imshow，不能用 matplotlib 的 plt.show）
        cv2.imshow("Autonomous Face Detection Loop", output_img)

        # 等待使用者按鍵：等待時間為 0 代表無限等待直到按鍵
        key = cv2.waitKey(0)

        # 如果使用者按下 ESC 鍵 (ASCII 碼為 27)，就退出整個迴圈
        if key == 27:
            print("👋 使用者已手動中止程式。")
            break

    cv2.destroyAllWindows()
    print("🏁 所有人臉偵測流程結束！")


if __name__ == "__main__":
    main()