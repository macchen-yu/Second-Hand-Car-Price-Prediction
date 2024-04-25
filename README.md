

# 二手車價格預測

## 簡介

這個項目的目的是建立一個模型來預測二手車的價格。數據來源為`car data.csv`檔案。主要步驟包括數據前處理、模型訓練和預測結果。

## 檔案說明

- `./py/car data.csv`: 原始數據
- `./py/data_processing.ipynb`: 數據前處理的Jupyter Notebook
- `./py/Predict.csv`: 經過前處理後的數據
- `./py/train.py`: 訓練模型的Python腳本
- `./py/view_pt_variables.py`: 查看模型參數的Python腳本
- `./py/best_model.pt`: 訓練後得到的模型權重檔案
- `main.cpp`: 使用C++實現預測的主程式

## 數據前處理

數據前處理的步驟包括:

1. 將`YEARS`(年份)換成`AGE`(車齡)，公式為: `AGE = 2023 - YEARS`
2. 將`KMS_DRIVEN`(里程數)除以10000進行縮小
3. 透過熱力圖分析找到輸入變數之間的關聯性，選擇`Age`、`Present_Price`、`Kms_Driven`、`Seller_Type`和`Fuel_Type`作為模型參數
4. 將文字資訊替換為數字:
   - `FUEL_TYPE`: `PETROL`改為2，`DIESEL`改為1
   - `SELLER_TYPE`: `DEALER`改為1，`INDIVIDUAL`改為0
   - `TRANSMISSION`: `MANUAL`改為1，`AUTOMATIC`改為0

前處理後的結果儲存在`Predict.csv`檔案中。

## 訓練模型

訓練模型的Python腳本為`train.py`。重要代碼包括:

1. 使用五個參數(`Age`、`Present_Price`、`Kms_Driven`、`Seller_Type`和`Fuel_Type`)送進模型訓練
2. `fit`函數使用SGD優化器調整模型參數，並儲存loss最低的模型權重檔案`best_model.pt`
3. 設定學習率為0.01，訓練3000個批次

## 預測結果

### Python實現

使用Python腳本`view_pt_variables.py`可以查看訓練後的模型參數。

### C++實現

C++主程式`main.cpp`實現了預測功能。主要步驟包括:

1. 從`Predict.csv`讀取輸入數據
2. 將訓練後的模型權重參數複製到`WEIGHT`矩陣中
3. 將輸入數據`INPUTVALUE`和模型權重`WEIGHT`送入`linear_Regression`函數計算預測價格
4. 將模型的bias參數加入預測價格中
5. 顯示選定的輸入數據、實際價格和模型預測價格

## 執行predict程式
```shell
.\build\Debug\predict.exe
```