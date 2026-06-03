import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# 1. 定義資料夾路徑（對應 image_55e02f.png 的結構）
ANNOTATIONS_DIR = "annotations"
IMAGES_DIR = "images"


def parse_xml(xml_path):
    """解析 Pascal VOC XML 檔案，提取圖片檔名與所有人臉框"""
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


def crop_upper_face(image, box):
    """
    根據人臉比例，切出眼睛與眉毛區域 (Upper Face Features)
    通常眼睛和眉毛位於整張臉的垂直上半部（約 ymin 到 ymin + h*0.55 的區間）
    """
    xmin, ymin, xmax, ymax = box
    img_h, img_w, _ = image.shape

    # 防止邊界框超出圖片範圍
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)

    face_h = ymax - ymin

    # 幾何規則切取：取臉部的上半部 10% 到 55% 的區域，這通常是眉毛與眼睛的所在範圍
    upper_ymin = ymin + int(face_h * 0.10)
    upper_ymax = ymin + int(face_h * 0.55)

    # 切割影像 (ROI - Region of Interest)
    upper_face_roi = image[upper_ymin:upper_ymax, xmin:xmax]
    return upper_face_roi


def main():
    # 取得 annotations 資料夾內所有的 xml 檔案
    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]

    if not xml_files:
        print(f"請確保 '{ANNOTATIONS_DIR}' 資料夾內有 XML 標註檔案（如 image_55e862.png 所示）")
        return

    # 我們拿第一個 XML 檔案來做測試展示
    target_xml = xml_files[0]
    xml_path = os.path.join(ANNOTATIONS_DIR, target_xml)

    # 解析 XML
    img_filename, face_data = parse_xml(xml_path)
    img_path = os.path.join(IMAGES_DIR, img_filename)

    # 讀取對應的圖片
    image = cv2.imread(img_path)
    if image is None:
        print(f"找不到對應的圖片：{img_path}，請確認 images 資料夾內有該圖片。")
        return

    # OpenCV 預設是 BGR 順序，轉換成 RGB 方便 matplotlib 顯示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"成功處理檔案: {target_xml} -> 找到 {len(face_data)} 個人臉目標")

    # 巡覽圖中找到的每張臉，切出眼部並顯示
    for i, data in enumerate(face_data):
        label = data["label"]
        box = data["box"]

        # 切出眼部眉毛特徵
        upper_face = crop_upper_face(image_rgb, box)

        # 如果切出來的區域太小（可能標註有雜訊），就跳過
        if upper_face.size == 0:
            continue

        # 繪製結果
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        # 在原圖上畫出完整人臉框
        cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        plt.imshow(image_rgb)
        plt.title(f"Original: {label}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(upper_face)
        plt.title("Cropped Upper Face")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()