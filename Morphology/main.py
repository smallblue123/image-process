import numpy as np
import cv2
import os

def convolution(image, mask):
    M, N = image.shape[:2]
    border_M, border_N = [int(x / 2) for x in mask.shape[:2]]
    border_pic = add_border(image, mask)
    convolution_pic = np.empty_like(image, np.int32)
    for i in range(M):
        for j in range(N):
            slice_range = (slice(i+1-border_M, i+1+border_M+1), slice(j+1-border_N, j+1+border_N+1))
            convolution_pic[i, j] = np.sum(border_pic[slice_range] * mask)

    CNN_pic = np.clip(convolution_pic, 0, 255)
    return CNN_pic.astype(np.uint8)


def add_border(image, mask):
    img_M, img_N = image.shape[:2]
    border_M, border_N = [int(x / 2) for x in mask.shape[:2]]
    if image.ndim == 2:
        new_pic = np.empty((img_M+border_M*2, img_N+border_N*2))
    elif image.ndim == 3:
        new_pic = np.empty((img_M+border_M*2, img_N+border_N*2, image.shape[2]))

    new_pic[border_M:img_M+border_M, border_N:img_N+border_N] = image

    #left
    new_pic[border_M:img_M + border_M, :border_N] = new_pic[border_M:img_M+border_M, 2*border_N-1:border_N-1:-1]

    # right
    new_pic[border_M:img_M + border_M, border_N+img_N:] = new_pic[border_M:img_M + border_M, img_N+border_N-1:img_N-1:-1]

    # top
    new_pic[:border_M, border_N:border_N+img_N] = new_pic[2*border_M-1:border_M-1:-1, border_N:border_N+img_N]

    # bottom
    new_pic[border_M+img_M:, border_N:border_N+img_N] = new_pic[img_M+border_M-1:img_M-1:-1, border_N:border_N+img_N]
    # print("bottom")

    # 左上
    new_pic[:border_M, :border_N] = new_pic[2*border_M-1:border_M-1:-1, 2*border_N-1:border_N-1:-1]
    # 左下
    new_pic[border_M+img_M:, :border_N] = new_pic[img_M+border_M-1:img_M-1:-1, 2 * border_N-1:border_N-1:-1]
    # 右上
    new_pic[:border_M, border_N+img_N:] = new_pic[2*border_M-1:border_M-1:-1, img_N+border_N-1:img_N-1:-1]
    # 右下
    new_pic[border_M+img_M:, border_N + img_N:] = new_pic[img_M + border_M-1:img_M-1:-1, img_N+border_N-1:img_N-1:-1]

    return new_pic


def set_filling_mask(n):
    return np.ones((n,n))


def filling(img):
    filling_mask = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=bool)
    last_filling_img = None
    inverse_mask = swap_0_1(img)
    seed_point = (250, 500)
    # filling_img = img
    filling_img = np.zeros_like(img, dtype=bool)
    filling_img[seed_point] = True

    count = 0
    while not np.array_equal(filling_img, last_filling_img):
        last_filling_img = filling_img.copy()
        dil = dilation(filling_img, filling_mask)
        filling_img = np.logical_and(dil, inverse_mask)

        count += 1
        print(count)

        cv2.imwrite(f"filling_results/{count}.png", binary2img(np.logical_or(img, filling_img)))

    return filling_img


def extract(img):
    erosion_img = erosion(img)
    mask_result = img - erosion_img
    return mask_result


def erosion(erosion_img, mask_size=3, times=1):
    erosion_mask = set_filling_mask(mask_size)
    threshold = np.sum(erosion_mask)
    for _ in range(times):
        CNN_mask = convolution(erosion_img, erosion_mask)
        erosion_img = np.where(CNN_mask < threshold, 0, 1)

    return erosion_img


def dilation(dilation_img, mask_size=3, times=1):
    dilation_mask = set_filling_mask(mask_size)
    for _ in range(times):
        CNN_mask = convolution(dilation_img, dilation_mask)
        dilation_img = CNN_mask >= 1
    return dilation_img


def open(img, mask_size=3, times=1):
    binary_erosion_img = erosion(img, mask_size=mask_size, times=times)
    erosion_img = binary2img(binary_erosion_img)
    binary_dilation_img = dilation(erosion_img, mask_size=mask_size, times=times)
    dilation_img = binary2img(binary_dilation_img)
    return dilation_img


def close(img, mask_size=3, times=1):
    binary_dilation_img = dilation(img, mask_size=mask_size, times=times)
    dilation_img = binary2img(binary_dilation_img)
    binary_erosion_img = erosion(dilation_img, mask_size=mask_size, times=times)
    erosion_img = binary2img(binary_erosion_img)
    return erosion_img


def swap_0_1(arr):
    result = arr.copy().astype(np.int32)

    result[result == 0] = -1
    result[result == 1] = 0
    result[result == -1] = 1

    return result.astype(np.uint8)


def img2binary(img):
    return np.int32(img/255)


def binary2img(img):
    return img.astype(np.uint8)*255


# 自動切換到 main.py 所在目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
binary_img = img2binary(img)
binary_dilation_img = dilation(binary_img, times=20)
dilation_img = binary2img(binary_dilation_img)
cv2.imwrite("dilation.png", dilation_img)
binary_erosion_img = erosion(binary_img, times=20)
erosion_img = binary2img(binary_erosion_img)
cv2.imwrite("erosion.png", erosion_img)


binary_extract_img = extract(binary_img)
extract_img = binary2img(binary_extract_img)
cv2.imwrite("extract.png", extract_img)

# open
open_img = open(binary_img, times=20)
cv2.imwrite("open.png", open_img)

# close
close_img = close(binary_img, times=20)
cv2.imwrite("close.png", close_img)

# open+close
close_img = close(open_img, times=20)
cv2.imwrite("open after close.png", close_img)

# binary_filling_img = filling(binary_img)
# filling_img = binary2img(binary_filling_img)
# cv2.imwrite("filling_results/filling.png", filling_img)




