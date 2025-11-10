import random
import cv2
import numpy as np


def CNN(image, mask, threshold=0):
    M, N = image.shape[:2]
    border_M, border_N = [int(x / 2) for x in mask.shape[:2]]
    border_pic = add_border(image, mask)
    CNN_pic = np.empty_like(image, np.int32)
    for i in range(M):
        for j in range(N):
            # slice_range = (slice(i, i+border_M), slice(j, j+border_N))
            slice_range = (slice(i+1-border_M, i+1+border_M+1), slice(j+1-border_N, j+1+border_N+1))
            laplacian = np.sum(border_pic[slice_range] * mask)
            if laplacian > threshold:
                CNN_pic[i, j] = laplacian
            else:
                CNN_pic[i, j] = 0
    CNN_pic = np.clip(CNN_pic, 0, 255)
    return CNN_pic.astype(np.uint8)


def add_border(image, mask):
    img_M, img_N = image.shape[:2]
    border_M, border_N = [int(x / 2) for x in mask.shape[:2]]
    if image.ndim == 2:
        new_pic = np.empty((img_M+border_M*2, img_M+border_N*2))
    elif image.ndim == 3:
        new_pic = np.empty((img_M+border_M*2, img_M+border_N*2, image.shape[2]))
    new_pic[border_M:img_M+border_M, border_N:img_N+border_N] = image
    # print(new_pic)
    # left
    new_pic[border_M:img_M + border_M, :border_N] = new_pic[border_M:img_M+border_M, 2*border_N-1:border_N-1:-1]
    # print("left")
    # print(new_pic)
    # right
    new_pic[border_M:img_M + border_M, border_N+img_N:] = new_pic[border_M:img_M + border_M, img_N+border_N-1:img_N-1:-1]
    # print("right")
    # print(new_pic)
    # top
    new_pic[:border_M, border_N:border_N+img_N] = new_pic[2*border_M-1:border_M-1:-1, border_N:border_N+img_N]
    # print("top")
    # print(new_pic)
    # bottom
    new_pic[border_M+img_M:, border_N:border_N+img_N] = new_pic[img_M+border_M-1:img_M-1:-1, border_N:border_N+img_N]
    # print("bottom")
    # print(new_pic)

    # 左上
    new_pic[:border_M, :border_N] = new_pic[2*border_M-1:border_M-1:-1, 2*border_N-1:border_N-1:-1]
    # 左下
    new_pic[border_M+img_M:, :border_N] = new_pic[img_M+border_M-1:img_M-1:-1, 2 * border_N-1:border_N-1:-1]
    # 右上
    new_pic[:border_M, border_N+img_N:] = new_pic[2*border_M-1:border_M-1:-1, img_N+border_N-1:img_N-1:-1]
    # 右下
    new_pic[border_M+img_M:, border_N + img_N:] = new_pic[img_N + border_N-1:img_N-1:-1, img_N+border_N-1:img_N-1:-1]

    return new_pic


def set_n_mean_filter(n):
    mean_filter = np.ones((n, n), dtype=np.float32)/n
    return mean_filter


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    normalize_img = img / (max-min)
    return normalize_img


laplacian_mask_1d = np.array([[ 0, -1,  0],
                              [-1,  4, -1],
                              [ 0, -1,  0]])

laplacian_mask_2d = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])

sobel_mask_vertical = np.array([[ 1,  2,  1],
                                [ 0,  0,  0],
                                [-1, -2, -1]])

sobel_mask_horizontal = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]])

mean_filter = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

img = cv2.imread("lena_black.jpg", cv2.IMREAD_GRAYSCALE).astype(np.int32)
mean_img = CNN(img, mean_filter)
laplacian_img = CNN(mean_img, laplacian_mask_2d)
cv2.imwrite("step1.jpg", laplacian_img)
step_2_img = img+laplacian_img
cv2.imwrite("step2.jpg", step_2_img)
sobel_horionoatal = CNN(img, sobel_mask_horizontal).astype(np.int32)
sobel_vertical = CNN(img, sobel_mask_vertical).astype(np.int32)
gradient_magnitude = np.sqrt(sobel_horionoatal ** 2 + sobel_vertical ** 2)
gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
cv2.imwrite("step3.jpg", gradient_magnitude)
mean_img = CNN(gradient_magnitude, mean_filter)
cv2.imwrite("step4.jpg", mean_img)

# 課本做法
normalize_val = normalize(mean_img)
normalize_img = normalize_val * step_2_img
cv2.imwrite("step5_1.jpg", normalize_img)
final_img = normalize_img + img
cv2.imwrite("step6_1.jpg", final_img)


# 老師作法
normalize_val = normalize(mean_img)
normalize_img = normalize_val * laplacian_img
cv2.imwrite("step5_2.jpg", normalize_img)
final_img = normalize_img + img
cv2.imwrite("step6_2.jpg", final_img)

