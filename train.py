import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from tqdm import tqdm

# 1. 基礎路徑設定（完全對應 image_4a9267.png 的黃金結構）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR = os.path.join(BASE_DIR, "Validation")
TEST_DIR = os.path.join(BASE_DIR, "Test")

# 環境檢查（GPU CUDA 加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用硬體加速設備: {device}")


def main():
    # 2. 定義影像預處理與資料增強 (Data Augmentation)
    # 針對訓練集進行數據增強，防止過擬合
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 驗證集與測試集只需要縮放與標準化即可
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 使用 PyTorch 內建的 ImageFolder 自動讀取分類資料夾
    print("📦 正在讀取新資料庫...")
    try:
        train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_test_transform)
    except Exception as e:
        print(f"❌ 讀取失敗：請確保 Train, Validation, Test 資料夾內包含子資料夾（如 open/closed）。\n錯誤訊息: {e}")
        return

    # 輸出類別名稱（例如 ['closed', 'open']）
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"🎉 成功識別類別: {class_names} (共 {num_classes} 類)")
    print(
        f"📊 資料分佈 -> 訓練集: {len(train_dataset)}張 | 驗證集: {len(val_dataset)}張 | 測試集: {len(test_dataset)}張")

    # 建立 DataLoader（既然用 GPU，批次直接開 64 壓榨算力）
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 4. 構建 MobileNetV2 卷積神經網路
    print("🧠 正在建構 MobileNetV2 遷移學習模型...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # 根據新資料庫的類別數量修改最後的分類層
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 定義損失函數與最佳化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. 開始標準訓練與驗證迴圈
    epochs = 15  # 有了三向切分資料集，我們可以跑 15 輪，並且每輪都在 Validation 上驗證
    best_val_acc = 0.0

    print(f"⚙️ 開始訓練，預計執行 {epochs} 個輪次...")
    for epoch in range(epochs):
        # --- 訓練階段 ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- 驗證階段（用來即時監控有沒有過擬合） ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100.0 * val_correct / val_total

        print(f"【Epoch {epoch + 1:02d}/{epochs}】"
              f" Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% ||"
              f" Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # 如果這輪的驗證集表現最好，就存檔這個最佳權重
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model_save_path = os.path.join(BASE_DIR, "best_masked_face_model.pth")
            torch.save(model.state_dict(), model_save_path)

    print("\n" + "=" * 50)
    print(f"🏆 訓練結束！最佳驗證集準確率為: {best_val_acc:.2f}%")

    # 6. 最終盲測：在完全不參與訓練的 Test 集上做最終大考
    print("📝 正在加載最佳模型，對 Test 集進行最終大考驗...")
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_masked_face_model.pth")))
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    final_test_acc = 100.0 * test_correct / test_total
    print(f"🎯 【最終大考結果】測試集（Test Set）最終準確度: {final_test_acc:.2f}%")
    print("=" * 50)
    print(f"💾 最佳 AI 大腦已儲存至: {os.path.join(BASE_DIR, 'best_masked_face_model.pth')}")


if __name__ == "__main__":
    main()