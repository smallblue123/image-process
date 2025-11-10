# Image Sharpening — 從零實作影像銳化演算法

本專案為 `image-process` 系列中的「影像銳化」模組。
目標是使用 **純 Python + NumPy** 手寫捲算運算，完成從平滑、邊緣偵測到高頻強化的影像銳化流程。
透過此專案，你將能理解影像高頻成分的作用與銳化的數學基礎。

---

## 專案說明

* 不使用 OpenCV 的高階濾波函式，全部以自寫 `CNN()` 完成捲算。
* 包含 **Laplacian**、**Sobel** 與 **Mean Filter** 三種核心濾波運算。
* 程式將依步驟輸出 `step1.jpg`～`step6.jpg`，方便觀察每階段效果。

---

## 六大步驟流程

### 1. 對原始影像做 Mean Filter 平滑化

使用 3×3 均值濾波降低雜訊。
輸出：`step1.jpg`

---

### 2. 對平滑後影像做 Laplacian 邊緣偵測

強化高頻邊緣訊息。
輸出：`step2.jpg`

---

### 3. 計算 Sobel 梯度幅值圖

同時計算水平與垂直梯度，求其平方和開根號。
輸出：`step3.jpg`

---

### 4. 對梯度圖再做 Mean Filter 平滑化

降低梯度圖的雜訊，使權重分布更穩定。
輸出：`step4.jpg`

---

### 5. 將步驟(4)結果做正規化，並乘上 Laplacian 圖

產生一張加權高頻強化圖。
輸出：`step5.jpg`

---

### 6. 將步驟(5)結果加回原始影像

形成最終銳化結果。
輸出：`step6.jpg`

---

## 為什麼這樣做？

* **高頻強化**：邊緣與細節屬於影像中高頻成分，銳化的本質是放大這些成分。
* **手寫捲算**：以 `CNN()` 模擬濾波過程，更清楚理解捲算在影像處理的意義。
* **正規化控制**：將權重壓在 [0, 1] 區間，防止像素過曝或過度增強。
* **單一路徑輸出**：簡化為最清楚的銳化邏輯，使每一步的效果明確可見。

---

## 範例結果圖片

| Step | 說明                 | 輸出檔案        | 圖片預覽                                     |
| ---- | ------------------ | ----------- | ---------------------------------------- |
| 1    | 原圖經 mean filter 平滑 | `step1.jpg` | ![step1](./Image%20Sharpening/step1.jpg) |
| 2    | Laplacian 邊緣偵測     | `step2.jpg` | ![step2](./Image%20Sharpening/step2.jpg) |
| 3    | Sobel 梯度幅值圖        | `step3.jpg` | ![step3](./Image%20Sharpening/step3.jpg) |
| 4    | 平滑梯度圖              | `step4.jpg` | ![step4](./Image%20Sharpening/step4.jpg) |
| 5    | 正規化 × Laplacian    | `step5.jpg` | ![step5](./Image%20Sharpening/step5.jpg) |
| 6    | 加回原圖後的銳化結果         | `step6.jpg` | ![step6](./Image%20Sharpening/step6.jpg) |


---

## 使用方式

1. 準備灰階影像（如 `lena_black.jpg`）放於 `Image Sharpening/`。
2. 執行：

   ```bash
   python main.py
   ```
3. 產生輸出影像：

   ```
   step1.jpg ~ step6.jpg
   ```
4. 每張圖對應到前述的處理階段，可逐步檢視變化。
