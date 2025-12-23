import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(size=5, sigma=1.4):
    """
    步驟 1: 產生高斯濾波核 (Gaussian Kernel)
    降噪，避免 Laplacian 或梯度對雜訊過度反應。
    """
    kernel = np.zeros((size, size))
    center = size // 2
    sum_val = 0
    
    for x in range(size):
        for y in range(size):
            diff = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = np.exp(-diff / (2 * sigma ** 2))
            sum_val += kernel[x, y]
            
    return kernel / sum_val

def sobel_filters(img):
    """
    步驟 2: 計算梯度與方向 (Sobel)
    G = sqrt(Gx^2 + Gy^2)
    """
    # 定義 Sobel 算子
    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]], np.float32)
    
    # 卷積運算
    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)
    
    # 計算梯度大小 (G) 與 方向 (theta)
    G = np.hypot(Ix, Iy) # 等同於 sqrt(Ix**2 + Iy**2)
    G = np.clip(G, 0, 255)

    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

def non_max_suppression(img, D):
    """
    步驟 3: 非極大值抑制 (NMS)
    """
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180 # 將角度轉為 0-180 度
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                
                # 角度簡化 (Quantization)：分為 0, 45, 90, 135 四個方向
                # case 0 度 (垂直左右)
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # case 45 度 (右上左下)
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # case 90 度 (水平上下)
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # case 135 度 (左上右下)
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                # 如果自己是最大的，就保留；否則去掉
                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError:
                pass
    
    return Z

def double_threshold(img, low_val=None, high_val=None):
    """
    步驟 4: 雙閾值 (Double Threshold)
    區分 強邊緣、弱邊緣、雜訊。
    """

    if high_val is None:
        # 如果沒有給閾值，呼叫Otsu函式計算
        high_val = get_otsu_threshold(img)
    
    if low_val is None:
        low_val = high_val * 0.5
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(100)   # 弱邊緣暫定值
    strong = np.int32(255) # 強邊緣值
    
    # 找出強邊緣與弱邊緣的位置
    strong_i, strong_j = np.where(img >= high_val)
    zeros_i, zeros_j = np.where(img < low_val)
    weak_i, weak_j = np.where((img <= high_val) & (img >= low_val))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def edge_linking(img, weak, strong=255):
    """
    使用深度優先搜尋 (DFS) 進行連接。
    """
    M, N = img.shape
    
    # 步驟 1: 初始化 Stack，放入所有強邊緣點
    # 存儲 (i, j) 座標
    edge_stack = []
    
    # 預先找到所有強邊緣點作為起點
    for i in range(M):
        for j in range(N):
            if img[i, j] == strong:
                edge_stack.append((i, j))
                
    # 檢查 8 個鄰居的偏移量
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # 步驟 2 3 4: 迭代搜尋並擴散邊緣
    while edge_stack:
        r, c = edge_stack.pop() # 彈出當前強邊緣點
        
        # 檢查 8 個鄰居
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc # 鄰居座標
            
            # 檢查是否越界
            if 0 <= nr < M and 0 <= nc < N:
                
                # 步驟 4: 如果鄰居是弱邊緣，則將其救起並加入 Stack
                if img[nr, nc] == weak:
                    img[nr, nc] = strong
                    edge_stack.append((nr, nc)) # 推入 Stack，繼續擴散
                    
    # 步驟 5: 清除所有孤立的弱邊緣
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                img[i, j] = 0 

    return img
    

def get_otsu_threshold(image):
    # 確保輸入是 uint8 格式
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # 1. 計算直方圖
    hist = np.bincount(image.ravel(), minlength=256)
    
    # 總像素數
    total_pixels = image.size
    
    # 初始化變數
    current_max_variance = 0
    best_threshold = 0
    
    # 預先計算總和與總平均，加速迴圈內的運算
    pixel_values = np.arange(256)
    
    # sum_total = 圖片所有像素值的總和
    sum_total = np.dot(pixel_values, hist)
    
    # 背景的累計權重 (w0) 和 累計總和 (sum_background)
    w0 = 0
    sum_background = 0
    
    # 2. 遍歷所有可能的閥值 (0 到 255)
    for t in range(256):
        # 更新背景權重 (加上當前像素值的數量)
        w0 += hist[t]
        
        # 如果背景沒有像素，繼續下一個
        if w0 == 0:
            continue
            
        # 前景權重 w1 = 總數 - 背景數
        w1 = total_pixels - w0
        
        # 如果前景沒有像素 (所有點都歸類為背景了)，結束迴圈
        if w1 == 0:
            break
        
        # 更新背景的像素總和 (加上 當前值 * 數量)
        sum_background += t * hist[t]
        
        # 計算背景平均 (mu0) 與 前景平均 (mu1)
        mu0 = sum_background / w0
        mu1 = (sum_total - sum_background) / w1
        
        # 3. 計算類別間變異數 (Between-class Variance)
        # 公式: sigma^2 = w0 * w1 * (mu0 - mu1)^2
        variance = w0 * w1 * ((mu0 - mu1) ** 2)
        
        # 4. 找出最大變異數對應的閥值
        if variance > current_max_variance:
            current_max_variance = variance
            best_threshold = t
            
    return best_threshold


# main
# 1. 讀取測試圖片
img = cv2.imread('lena_black.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('1_canny_noisy_input.jpg', img)

# 2. 執行 Canny
# Step 1: 高斯模糊
kernel = gaussian_kernel(size=5, sigma=1.4)
img_smoothed = cv2.filter2D(img, -1, kernel)
cv2.imwrite('2_canny_gaussian_kernel.jpg', img_smoothed)  # 儲存高斯模糊影像

# Step 2: 算Sobel 梯度
img_gradient, theta = sobel_filters(img_smoothed)
cv2.imwrite('3_canny_sobel_gradient.jpg', img_gradient)  # 儲存梯度影像


# Step 3: NMS (細化邊緣)
img_nms = non_max_suppression(img_gradient, theta)
cv2.imwrite('4_canny_nms.jpg', img_nms)  # 儲存 NMS 影像

# Step 4: 雙閾值
# 用OTSU算前景背景閾值
img_thresh, weak, strong = double_threshold(img_nms)
cv2.imwrite('5_canny_threshold.jpg', img_thresh)  # 儲存雙閾值影像

# Step 5: 連接弱邊緣
img_final = edge_linking(img_thresh, weak, strong)
cv2.imwrite('6_canny_final.jpg', img_final)  # 儲存最終 Canny 邊緣影像