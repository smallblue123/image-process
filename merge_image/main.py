import copy
import random
import cv2
import numpy as np
import time
import os

class SplicePic:
    def splice(self, left_pic, right_pic, pic_name):
        # 初始化SIFT
        ORB = cv2.ORB_create()
        display = Display()
        blender = Blender()

        print("find key points step:")
        start = time.time()
        # 在兩張影像上檢測關鍵點和描述子
        left_keypoints, left_descriptors = ORB.detectAndCompute(left_pic, None)
        right_keypoints, right_descriptors = ORB.detectAndCompute(right_pic, None)
        end = time.time()
        print(end-start)
        # display.show_key_points(left_pic, left_keypoints, f"left{pic_name}")
        # display.show_key_points(right_pic, left_keypoints, f"right{pic_name}")

        # # 創建一個空白影像，用於繪製關鍵點
        # keypoints_image1 = cv2.drawKeypoints(image1, keypoints1, None)
        # keypoints_image2 = cv2.drawKeypoints(image2, keypoints2, None)
        print("match step:")
        start = time.time()
        match_pos = self.BF_match(left_keypoints, right_keypoints, left_descriptors, right_descriptors)
        display.show_match_points(left_pic, right_pic, match_pos, f"Matched Image{pic_name}")
        end = time.time()
        print(end-start)

        print("find homo matrix step:")
        start = time.time()
        H, sample = self.RANSAC_find_TRMatrix(match_pos)
        display.show_match_points(left_pic, right_pic, sample, f"matrix points{pic_name}")
        end = time.time()
        print(end-start)

        print("interpolation step:")
        start = time.time()
        Tr_right_pic = self.interpolation(left_pic, right_pic, H)
        cv2.imwrite(f"interpolation{pic_name}.png", Tr_right_pic)
        end = time.time()
        print(end-start)

        print("merge and blending step:")
        start = time.time()
        merged_pic = blender.linearBlending(left_pic, Tr_right_pic)
        cv2.imwrite(f"warp{pic_name}.png", merged_pic)
        final_pic = self.delete_black_border(merged_pic)
        cv2.imwrite(f"delete_border_warp{pic_name}.png", final_pic)
        end = time.time()
        print(end-start)

        return final_pic.astype(np.uint8)

    def compute_homography_matrix(self, src_points, dst_points):
        # 係數矩陣 A 和結果向量 b
        A = np.zeros((8, 8))
        b = np.zeros((8, 1))

        for i in range(4):
            A[i * 2] = [src_points[i][0], src_points[i][1], 1, 0, 0, 0, -src_points[i][0] * dst_points[i][0],
                        -src_points[i][1] * dst_points[i][0]]
            A[i * 2 + 1] = [0, 0, 0, src_points[i][0], src_points[i][1], 1, -src_points[i][0] * dst_points[i][1],
                            -src_points[i][1] * dst_points[i][1]]
            b[i * 2] = dst_points[i][0]
            b[i * 2 + 1] = dst_points[i][1]

        # 使用最小二乘法求解線性方程組 Ax = b
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        x = np.append(x, 1)  # 補充透視轉換矩陣的最后一個參數

        # 將一維的參數轉為 3x3 的透視轉換矩陣
        transform_matrix = x.reshape((3, 3))
        return transform_matrix

    def descriptor_hamming_distance(self, arr1, arr2):
        descriptors1 = ''.join([bin(i)[2:].zfill(8) for i in arr1])
        descriptors2 = ''.join([bin(i)[2:].zfill(8) for i in arr2])

        return self.hamming_distance(descriptors1, descriptors2)

    def hamming_distance(self, str1, str2):
        if len(str1) != len(str2):
            raise ValueError("The strings must have the same length.")

        distance = 0
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                distance += 1

        return distance

    def BF_match(self, keypoints1, keypoints2, descriptors1, descriptors2, ratio=0.75):
        match_id_distance = []
        for i in range(len(descriptors1)):
            min_id_distance = [-1, np.inf]
            second_min_id_distance = [-1, np.inf]
            for j in range(len(descriptors2)):
                distance = self.descriptor_hamming_distance(descriptors1[i], descriptors2[j])
                if distance < min_id_distance[1]:
                    second_min_id_distance = min_id_distance
                    min_id_distance = [j, distance]
                elif distance < second_min_id_distance[1]:
                    second_min_id_distance = [j, distance]
            match_id_distance.append([*min_id_distance, *second_min_id_distance])
        # print(match_id_distance)
        good_matches = []
        for i in range(len(match_id_distance)):
            if match_id_distance[i][1] < match_id_distance[i][3] * ratio:
                good_matches.append([i, match_id_distance[i][0]])
        # print(good_matches)
        match_pos = []
        for (id1, id2) in good_matches:
            pos1 = [round(keypoints1[id1].pt[0]), round(keypoints1[id1].pt[1])]
            pos2 = [round(keypoints2[id2].pt[0]), round(keypoints2[id2].pt[1])]
            match_pos.append([pos1, pos2])
        return np.array(match_pos)

    def RANSAC_find_TRMatrix(self, match_pos):
        source_poses = match_pos[:, 1]
        dest_poses = match_pos[:, 0]

        num_samples = len(match_pos)
        sub_samples = 4
        threshold = 2
        Iter_num = 8000
        best_H = np.array([])
        max_Iner_num = 0
        best_sample = []

        for turn in range(Iter_num):
            sample = random.sample(range(num_samples), sub_samples)
            Iner_num = 0
            H = self.compute_homography_matrix(source_poses[sample], dest_poses[sample])
            for i in range(num_samples):
                if i not in sample:
                    source_pos = np.hstack((source_poses[i], 1))
                    dest_pos = np.dot(H, source_pos.T)
                    dest_pos = dest_pos[:2] / dest_pos[2]

                    if np.linalg.norm(dest_poses[i]-dest_pos) < threshold:
                        Iner_num += 1
            if Iner_num > max_Iner_num:
                max_Iner_num = Iner_num
                best_H = H
                best_sample = sample

        sample_poses = []
        for i in best_sample:
            sample_poses.append([dest_poses[i], source_poses[i]])
            if [dest_poses[i], source_poses[i]] not in match_pos:
                print("error")
        print(sample_poses)
        print("max iner", max_Iner_num)

        return best_H, sample_poses

    def interpolation(self, left_img, right_img, transform_matrix):
        transform_matrix = np.linalg.inv(transform_matrix)
        lr, lc = left_img.shape[:2]
        rr, rc = right_img.shape[:2]

        warped_row = max(lr, rr)
        warped_col = lc+rc
        warped_image = np.zeros((warped_row, warped_col, 3))

        for r in range(warped_row):
            for c in range(warped_col):
                # 計算轉換後的座標
                dst_point = np.array([c, r, 1])
                src_point = np.dot(transform_matrix, dst_point.T)
                src_x, src_y = round(src_point[0]/src_point[2]), round(src_point[1]/src_point[2])
                int_src_x, int_src_y = int(src_point[0]/src_point[2]), int(src_point[1]/src_point[2])

                # 雙線性差值計算像素值
                if 0 <= src_x < rc and 0 <= src_y < rr:
                    top_left = right_img[int_src_y, int_src_x]
                    top_right = right_img[int_src_y, min(rr-1, int_src_x+1)]
                    bottom_left = right_img[min(int_src_y+1, rc-1), int_src_x]
                    bottom_right = right_img[min(int_src_y+1, rc-1), min(rr-1, int_src_x+1)]
                    x_diff = src_point[0]/src_point[2] - int_src_x
                    y_diff = src_point[1]/src_point[2] - int_src_y
                    warped_image[r, c] = top_left * (1-x_diff) * (1-y_diff) + \
                                         top_right * x_diff * (1-y_diff) + \
                                         bottom_left * (1-x_diff) * y_diff + \
                                         bottom_right * x_diff * y_diff
                    warped_image[r, c] = right_img[src_y, src_x]
        return warped_image
    
    # 去除合併後影像凸出的部分
    def delete_black_border(self, pic):
        r, c = pic.shape[:2]
        min_r = 0
        max_r = r-1
        min_c = 0
        max_c = c-1
        # find top row
        for i in range(r):
            all_black = True
            for j in range(c):
                if np.count_nonzero(pic[i, j]) > 0:
                    all_black = False
                    break
            if all_black:
                min_r = i
            else:
                break
        # find bottom row
        for i in range(r-1, -1, -1):
            all_black = True
            for j in range(c):
                if np.count_nonzero(pic[i, j]) > 0:
                    all_black = False
                    break
            if all_black:
                max_r = i
            else:
                break
        # find left column
        for i in range(c):
            all_black = True
            for j in range(r):
                if np.count_nonzero(pic[j, i]) > 0:
                    all_black = False
                    break
            if all_black:
                min_c = i
            else:
                break
        # find right column
        for i in range(c-1, -1, -1):
            all_black = True
            for j in range(r):
                if np.count_nonzero(pic[j, i]) > 0:
                    all_black = False
                    break
            if all_black:
                max_c = i
            else:
                break
        return pic[min_r:max_r, min_c:max_c]

class Blender():
    def linearBlending(self, left_img, right_img):
        lr, lc = left_img.shape[:2]
        rr, rc = right_img.shape[:2]

        adjust_left_img = np.zeros_like(right_img)
        adjust_left_img[:lr, :lc] = left_img

        overlap_mask = self.find_overlap_mask(adjust_left_img, right_img)
        alpha_mask = self.find_alpha_mask(overlap_mask)

        splice_pic = np.empty_like(right_img)
        for i in range(rr):
            for j in range(rc):
                alpha = alpha_mask[i, j]
                # if alpha not in {0 ,1}:
                #     print("i, j", i, j)
                #     print("alpha:", alpha)
                #     print("origin:", splice_pic[i, j])
                splice_pic[i, j] = np.round((1-alpha)*adjust_left_img[i, j] + alpha*right_img[i,j])
                # if alpha not in {0, 1}:
                #     print("new:", splice_pic[i, j])
        return splice_pic

    def find_overlap_mask(self, left_img, right_img):
        lr, lc = left_img.shape[:2]
        rr, rc = right_img.shape[:2]

        right_mask = np.count_nonzero(right_img, axis=2)
        left_mask = np.count_nonzero(left_img, axis=2)

        mask = np.zeros_like(right_mask)
        for i in range(lr):
            for j in range(rc):
                if right_mask[i, j] and left_mask[i, j]:
                    mask[i, j] = 1
        return mask

    def find_alpha_mask(self, mask):
        r, c = mask.shape[:2]
        alpha_mask = np.zeros_like(mask, dtype=np.float32 )
        for i in range(r):
            min = -1
            max = -1
            for j in range(c):
                if mask[i, j]:
                    if min == -1:
                        min = j
                    else:
                        max = j
            if min == max:
                continue

            step = 1/(max-min)
            for j in range(min, c):
                if j < max+1:
                    alpha_mask[i, j] = (j-min)*step
                else:
                    alpha_mask[i, j] = 1
        return alpha_mask


class Display():
    def show_match_points(self, img1, img2, match_pos, pic_name):
        img1_row, img1_col = img1.shape[:2]
        img2_row, img2_col = img2.shape[:2]

        new_pic = np.zeros([max(img1_row, img2_row), img1_col+img2_col, 3], dtype=np.uint8)
        new_pic[0:img1_row, 0:img1_col] = img1
        new_pic[0:img2_row, img1_col:] = img2

        match_poses = copy.copy(match_pos)
        for pos1, pos2 in match_poses:
            pos2[0] += img1_col
            cv2.circle(new_pic, pos1, 2, (0, 0, 255), -1)
            cv2.circle(new_pic, pos2, 2, (0, 0, 255), -1)
            cv2.line(new_pic, pos1, pos2, (0, 255, 0), 2)

        # cv2.imshow('Matched Image', new_pic)
        cv2.imwrite(f'{pic_name}.png', new_pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def key_points2pos(self, key_points):
        poses = []
        for point in key_points:
            pos = [round(point.pt[0]), round(point.pt[1])]
            poses.append(pos)
        return poses

    def show_key_points(self, img, key_points, pic_name):
        poses = self.key_points2pos(key_points)
        self.show_points(img, poses)
        cv2.imwrite(f'{pic_name}.jpg', img)
        # cv2.imshow('Matched Image', new_pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def show_points(self, img, poses):
        for pos in poses:
            cv2.circle(img, pos, 2, (0, 0, 255), -1)
        cv2.imwrite('poses Image.jpg', img)


# 自動切換到 main.py 所在目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 讀取兩張影像
source_dir = "./source/"

image1 = cv2.imread(source_dir+"img1.jpg", cv2.IMREAD_UNCHANGED)
image2 = cv2.imread(source_dir+"img2.jpg", cv2.IMREAD_UNCHANGED)
image3 = cv2.imread(source_dir+"img3.jpg", cv2.IMREAD_UNCHANGED)

start = time.time()
splice = SplicePic()
warped_pic = splice.splice(image1, image2, "1_2")
warped_pic = splice.splice(warped_pic, image3, "2_3")

end = time.time()
print(end-start)








