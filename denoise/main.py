import random
import cv2
import numpy as np


def add_impulse_noise(img, black_rate, white_rate):
    M, N = img.shape
    noise_pic = np.empty_like(img)
    for i in range(M):
        for j in range(N):
            rand = random.random()

            if rand < black_rate:
                noise_pic[i, j] = 0
            elif rand < black_rate+white_rate:
                noise_pic[i, j] = 255
            else:
                noise_pic[i, j] = img[i, j]
    return noise_pic


def median_filter(img, filter_size):
    img_M, img_N = img.shape[:2]
    border_M = border_N = int(filter_size / 2)

    border_img = add_border(img, filter_size)
    new_img = np.empty_like(img)
    for i in range(img_M):
        for j in range(img_N):
            new_img[i, j] = np.median(border_img[i:i+filter_size, j:j+filter_size])
    return new_img


def adaptive_median_filter(img, max_filter_size=7):
    img_M, img_N = img.shape[:2]
    border_M = border_N = int(max_filter_size/2)
    border_img = add_border(img, max_filter_size)

    new_img = np.empty_like(img)
    for i in range(img_M):
        for j in range(img_N):
            # print(i,j)
            filter_size = 3
            Zxy = border_img[i + border_M, j + border_N]
            while filter_size <= max_filter_size:
                bias = int(max_filter_size/2)-int(filter_size/2)
                mask = border_img[i+bias:i+filter_size+bias, j+bias:j+filter_size+bias]

                Zmin = np.min(mask)
                Zmax = np.max(mask)
                Zmid = np.median(mask)
                # A
                if Zmin < Zmid < Zmax:
                    # check Zxy
                    if Zmin < Zxy < Zmax:
                        new_img[i, j] = Zxy
                        break
                    else:
                        new_img[i, j] = Zmid
                        break
                else:
                    filter_size += 2
            if filter_size > max_filter_size:
                new_img[i, j] = Zxy
    return new_img


def add_border(image, N):
    img_M, img_N = image.shape[:2]
    border_M = border_N = int(N / 2)
    if image.ndim == 2:
        new_pic = np.empty((img_M + border_M * 2, img_M + border_N * 2))
    elif image.ndim == 3:
        new_pic = np.empty((img_M + border_M * 2, img_M + border_N * 2, image.shape[2]))
    new_pic[border_M:img_M + border_M, border_N:img_N + border_N] = image
    # print(new_pic)
    # left
    new_pic[border_M:img_M + border_M, :border_N] = new_pic[border_M:img_M + border_M,
                                                    2 * border_N - 1:border_N - 1:-1]
    # print("left")
    # print(new_pic)
    # right
    new_pic[border_M:img_M + border_M, border_N + img_N:] = new_pic[border_M:img_M + border_M,
                                                            img_N + border_N - 1:img_N - 1:-1]
    # print("right")
    # print(new_pic)
    # top
    new_pic[:border_M, border_N:border_N + img_N] = new_pic[2 * border_M - 1:border_M - 1:-1,
                                                    border_N:border_N + img_N]
    # print("top")
    # print(new_pic)
    # bottom
    new_pic[border_M + img_M:, border_N:border_N + img_N] = new_pic[img_M + border_M - 1:img_M - 1:-1,
                                                            border_N:border_N + img_N]
    # print("bottom")
    # print(new_pic)

    # 左上
    new_pic[:border_M, :border_N] = new_pic[2 * border_M - 1:border_M - 1:-1, 2 * border_N - 1:border_N - 1:-1]
    # 左下
    new_pic[border_M + img_M:, :border_N] = new_pic[img_M + border_M - 1:img_M - 1:-1,
                                            2 * border_N - 1:border_N - 1:-1]
    # 右上
    new_pic[:border_M, border_N + img_N:] = new_pic[2 * border_M - 1:border_M - 1:-1,
                                            img_N + border_N - 1:img_N - 1:-1]
    # 右下
    new_pic[border_M + img_M:, border_N + img_N:] = new_pic[img_N + border_N - 1:img_N - 1:-1,
                                                    img_N + border_N - 1:img_N - 1:-1]

    return new_pic


def MSE(img1, img2):
    return np.mean((img1-img2) ** 2)


def PSNR(img1, img2):
    mse = MSE(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255
    psnr = 10 * np.log10((max_pixel**2) / mse)
    return psnr


img = cv2.imread("lena_black_with_CSE.png", cv2.IMREAD_GRAYSCALE)
# median_filter_img = median_filter(img, 3)
# cv2.imwrite("median_filter_lena_black.jpg", median_filter_img)
# adaptive_median_filter_img = adaptive_median_filter(img)
# cv2.imwrite("adaptive_median_filter_lena_black.png", adaptive_median_filter_img)

noise_img = add_impulse_noise(img, 1/3, 1/4)
cv2.imwrite("noise_lena_black.png", noise_img)
# noise_img = cv2.imread("noise_lena_black_with_CSE.jpg", cv2.IMREAD_GRAYSCALE)
median_filter_img = median_filter(noise_img, 3)
cv2.imwrite("median_filter_noise_lena_black_with_CSE.jpg", median_filter_img)
adaptive_median_filter_img = adaptive_median_filter(noise_img)
cv2.imwrite("adaptive_median_filter_noise_lena_black_with_CSE.png", adaptive_median_filter_img)


print(" origin and impulse noise pic:", PSNR(img, noise_img))
print("use mean filter:")
print("origin and denoise:", PSNR(img, median_filter_img))
print("use adaptive median filter:")
print("origin and denoise:", PSNR(img, adaptive_median_filter_img))








