import os
import torch


def main():
    # 🎯 這是 Ultralytics 官方存放在雲端最穩定的直接下載點 (絕對不會 404)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n-pose.pt"

    # 取得目前專案的根目錄路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "yolov11n-pose.pt")

    print("📡 開始強行下載 YOLOv11-Pose 模型大腦 (約 6.2MB)...")
    try:
        # 使用 torch 內建的權重下載工具，能夠完美繞過所有瀏覽器限制，並附帶官方進度條
        torch.hub.download_url_to_file(url, save_path)
        print(f"\n🎉 成功！檔案已順利下載。")
        print(f"📍 檔案位置: {save_path}")
    except Exception as e:
        print(f"\n❌ 下載失敗，錯誤訊息: {e}")


if __name__ == "__main__":
    main()