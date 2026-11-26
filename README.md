# Lab 3 胸部 X 光影像四分類

使用胸部 X 光影像區分正常、細菌性肺炎、病毒性肺炎及 COVID-19。競賽以 macro-F1 評分。

## 實驗原始碼

| Python 檔案 | 實驗 |
| --- | --- |
| [V1](src/v1_baseline.py) | ResNet50、EfficientNet-B0、ConvNeXt-Tiny 與類別不平衡處理 |
| [V2](src/v2.py) | ConvNeXt 的解析度、訓練輪數與集成 |
| [V3](src/v3_cv.py) | 五折交叉驗證與測試時增強 |
| [V4](src/v4.py) | 加入 focal loss 的 ConvNeXt 集成 |
| [V5](src/v5_xrv.py) | 加入胸部 X 光預訓練的 DenseNet121 |
| [V6](src/v6_gate.py) | 細菌性與病毒性肺炎的二分類模型 |
| [V7](src/v7_lung.py) | 肺部 ROI 前處理 |
| [V8](src/v8_final.py) | 全圖與肺部 ROI 模型集成 |
| [V9](src/v9_seeds.py) | 三種訓練種子的五折 ConvNeXt 集成 |

## 在 Google Colab 執行

1. 使用 GPU 執行階段。加入 Kaggle 的 `nycu-aimi-2026-lab3-cxr` 競賽，在 Colab Secrets 加入 `KAGGLE_API_TOKEN` 並授權筆記本讀取。腳本會下載資料到 `/content/data`。
2. 在 Colab 執行下列安裝指令。其餘套件使用 Colab 環境提供的 PyTorch、torchvision、NumPy、pandas、Pillow、scikit-learn 和 Matplotlib。

   ```python
   %pip install -q -U kaggle timm torchxrayvision opencv-python-headless
   ```

3. 將 `src` 上傳到 Colab 的 `/content/src`。V9 會沿用 V3 seed 42 的五個 `cv_cvA_cnx_wce_224_fold*.npz` 檔案。請依序執行以下腳本，兩者使用相同的 `DRIVE_SUBDIR`，各折輸出預設會存到 Google Drive 的 `MyDrive/Lab3_v2/outputs`。

   ```python
   %run /content/src/v3_cv.py
   %run /content/src/v9_seeds.py
   ```

## V9 結果

V9 使用 224 像素輸入、CLAHE、加權交叉熵與測試時增強，平均訓練種子 42、1、2 的五折預測。依實驗報告，OOF macro-F1 為 **0.85667**，Kaggle macro-F1 為 **0.87312**。

- [各種子與集成的 OOF 分數](results/v9/cv_results_seeds.csv)
- [OOF 混淆矩陣](results/v9/cv_confusion_matrix_seeds.png)
- [執行結果摘要](results/v9/run_config_v9.json)
