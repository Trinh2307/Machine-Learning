import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from pathlib import Path

def calculate_LBP(image, num_points, radius):
    # Tính toán LBP
    lbp = local_binary_pattern(image, num_points, radius, method='uniform')
    return lbp

img_path = str(Path(__file__).parent.parent / 'C:\\Users\\Admin\\Documents\\Nguyen li may hoc\\BAO CAO PHUONG PHAP TRICH CHON DAC TRUNG\\anh duong pho.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Số điểm lân cận và bán kính
num_points = 8
radius = 1

# Tính toán LBP cho hình ảnh đầu vào với các cấp độ khác nhau
lbp_imgs = []
for i in range(1, 6):  # Số lượng cấp độ
    lbp_img = calculate_LBP(img, num_points, radius * i)
    lbp_imgs.append(lbp_img)

# Hiển thị ảnh gốc và các cấp độ của ảnh LBP
plt.figure(figsize=(15, 10))

# Subplot cho ảnh gốc
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

# Subplot cho các cấp độ của ảnh LBP
for i, lbp_img in enumerate(lbp_imgs):
    plt.subplot(2, 3, i + 2)
    plt.imshow(lbp_img, cmap='gray')
    plt.title(f'LBP (R={radius*(i+1)})')
    plt.axis('off')

plt.tight_layout()
plt.show()
