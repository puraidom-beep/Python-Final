import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 基礎路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
LBF_MODEL_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")

# 2. 載入檢測器
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL_PATH)

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 47 + 1))


def parse_xml(xml_path):
    """解析 XML 獲取人臉框"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append({"label": label, "box": (xmin, ymin, xmax, ymax)})
    return filename, boxes


def calculate_ear(eye_pts):
    """計算 EAR 公式"""
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C) if C != 0 else 0


def main():
    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    if not xml_files:
        print("❌ 找不到 XML 檔案")
        return

    # 用來存儲所有臉部統計數據的清單
    all_results = []

    print(f"📊 開始批次處理整個資料集，共計 {len(xml_files)} 個 XML 檔案...\n")

    # 巡覽所有 XML 檔案進行批次分析
    for xml_file in xml_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        img_filename, face_data = parse_xml(xml_path)
        img_path = os.path.join(IMAGES_DIR, img_filename)

        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用 XML 標註框作為基準進行特徵點擬合，這樣即使戴口罩也能精準抓到眼睛
        faces = np.array(
            [[d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]] for d in face_data])

        success, landmarks = facemark.fit(gray, faces)
        if not success or landmarks is None:
            continue

        for i, landmark in enumerate(landmarks):
            pts = landmark[0]
            left_ear = calculate_ear(pts[LEFT_EYE_IDX])
            right_ear = calculate_ear(pts[RIGHT_EYE_IDX])
            mean_ear = (left_ear + right_ear) / 2.0

            # 蒐集數據：口罩狀態、EAR數值
            mask_label = face_data[i]["label"]
            status = "Closed/Drowsy" if mean_ear < 0.22 else "Open"

            all_results.append({
                "File": img_filename,
                "Mask_Status": mask_label,
                "EAR": mean_ear,
                "Eye_Status": status
            })

    # 3. 使用 Pandas 進行資料科學統計
    df = pd.DataFrame(all_results)

    print("=" * 50)
    print("📈 專案數據分析統計報告 (Pandas Export)")
    print("=" * 50)
    print(f"總計成功偵測人臉數: {len(df)} 張\n")

    # 統計不同口罩佩戴類別下的平均 EAR 數值
    print("💡 各類別下的眼睛平均縱橫比 (Mean EAR by Category):")
    summary = df.groupby("Mask_Status")["EAR"].mean().reset_index()
    print(summary.to_string(index=False))
    print("-" * 50)

    # 統計閉眼/疲勞人數佔比
    status_counts = df["Eye_Status"].value_counts()
    print("💡 全資料集駕駛/行人狀態分佈 (Eye Status Distribution):")
    for status, count in status_counts.items():
        percentage = (count / len(df)) * 100
        print(f" - {status}: {count}人 ({percentage:.1f}%)")
    print("=" * 50)

    # 額外好康：幫你畫出一張單張圖作為報告用成果（拿第一個檔案畫圖）
    # (此處保留原本的單張圖 matplotlib 繪圖程式碼，會自動彈出)
    if not df.empty:
        print("\n🎨 正在繪製視覺化成果圖...")
        # 重新讀取第一個檔案畫圖展示
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_files[0])
        img_filename, face_data = parse_xml(xml_path)
        img_path = os.path.join(IMAGES_DIR, img_filename)
        show_img = cv2.imread(img_path)
        gray_show = cv2.cvtColor(show_img, cv2.COLOR_BGR2GRAY)
        faces_show = np.array(
            [[d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]] for d in face_data])
        _, landmarks_show = facemark.fit(gray_show, faces_show)

        for i, lm in enumerate(landmarks_show):
            pts = lm[0]
            l_ear = calculate_ear(pts[LEFT_EYE_IDX])
            r_ear = calculate_ear(pts[RIGHT_EYE_IDX])
            m_ear = (l_ear + r_ear) / 2.0
            st = "Drowsy" if m_ear < 0.22 else "Open"
            x, y, w, h = faces_show[i]
            cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for pt in pts[LEFT_EYE_IDX]: cv2.circle(show_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
            for pt in pts[RIGHT_EYE_IDX]: cv2.circle(show_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
            cv2.putText(show_img, f"{face_data[i]['label']} EAR:{m_ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 0), 1)

        plt.figure(figsize=(7, 5))
        plt.imshow(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Final Demonstration Result")
        plt.show()


if __name__ == "__main__":
    main()