import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def remove_back(img_front: np.ndarray, img_back: np.ndarray, img_mask: np.array):
    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    #mask_r = img_mask/255
    im = (img_front - img_back)*img_mask
    return im.astype('uint8')

def make_gt(lab_im):
    center = lab_im[[1, 2]].to_numpy() #col0 = y and col1 = x
    base = np.zeros((lab_im.iloc[0][5], lab_im.iloc[0][4]))
    for i in range(center.shape[0]):
        base = cv2.circle(base, (center[i, 0], center[i, 1]), 7, 255, -1)
    return base/255

def do_thresh(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    _, n_thre = cv2.threshold(image, 200, 255, cv2.THRESH_TOZERO_INV)
    _, n_thre2 = cv2.threshold(n_thre, 100, 255, cv2.THRESH_TOZERO)
    close = cv2.morphologyEx(n_thre2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dilated = cv2.morphologyEx(close, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    mult = thresh * dilated + n_thre2
    mult[np.where(mult >= 150)] = 255
    mdi = cv2.morphologyEx(mult, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    return mdi

def get_boxes(image_base, image_out):
    contours, hierarchy = cv2.findContours(image_base, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        xx = int(x + w/2)
        yy = int(y + h/2)
        if (w > 20 and h > 20):
            cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(image_out,(xx,yy), 3, (255, 0, 0), -1)
            count += 1
    return image_out, count

if __name__ == '__main__':

    # Read target images

    img_b1 = cv2.imread('../P1_images/1660543200.jpg', cv2.IMREAD_COLOR)
    img_b2 = cv2.imread('../P1_images/1660546800.jpg', cv2.IMREAD_COLOR)
    img_b3 = cv2.imread('../P1_images/1660550400.jpg', cv2.IMREAD_COLOR)
    img_b4 = cv2.imread('../P1_images/1660554000.jpg', cv2.IMREAD_COLOR)
    img_b5 = cv2.imread('../P1_images/1660557600.jpg', cv2.IMREAD_COLOR)
    img_b6 = cv2.imread('../P1_images/1660561200.jpg', cv2.IMREAD_COLOR)
    img_b7 = cv2.imread('../P1_images/1660564800.jpg', cv2.IMREAD_COLOR)
    img_b8 = cv2.imread('../P1_images/1660568400.jpg', cv2.IMREAD_COLOR)
    img_b9 = cv2.imread('../P1_images/1660572000.jpg', cv2.IMREAD_COLOR)
    img_b10 = cv2.imread('../P1_images/1660575600.jpg', cv2.IMREAD_COLOR)

    # Read labels
    lab_b1 = pd.read_csv('../P1_images/1660543200_ground.csv', header=None)  # x == column1 , y == colum2
    lab_b2 = pd.read_csv('../P1_images/1660546800_ground.csv', header=None)
    lab_b3 = pd.read_csv('../P1_images/1660550400_ground.csv', header=None)
    lab_b4 = pd.read_csv('../P1_images/1660554000_ground.csv', header=None)
    lab_b5 = pd.read_csv('../P1_images/1660557600_ground.csv', header=None)
    lab_b6 = pd.read_csv('../P1_images/1660561200_ground.csv', header=None)
    lab_b7 = pd.read_csv('../P1_images/1660564800_ground.csv', header=None)
    lab_b8 = pd.read_csv('../P1_images/1660568400_ground.csv', header=None)
    lab_b9 = pd.read_csv('../P1_images/1660572000_ground.csv', header=None)
    lab_b10 = pd.read_csv('../P1_images/1660575600_ground.csv', header=None)

    # Get groundtruth images
    gt_b1 = make_gt(lab_b1)
    gt_b2 = make_gt(lab_b2)
    gt_b3 = make_gt(lab_b3)
    gt_b4 = make_gt(lab_b4)
    gt_b5 = make_gt(lab_b5)
    gt_b6 = make_gt(lab_b6)
    gt_b7 = make_gt(lab_b7)
    gt_b8 = make_gt(lab_b8)
    gt_b9 = make_gt(lab_b9)
    gt_b10 = make_gt(lab_b10)

    # Merge all label images into one and dilate it, gt_t is used to create mask3.jpg
    gt_t = gt_b1+gt_b2+gt_b3+gt_b4+gt_b5+gt_b6+gt_b7+gt_b8+gt_b9+gt_b10
    ret, gt_t = cv2.threshold(gt_t, 0.5, 10, cv2.THRESH_BINARY)
    gt_t = cv2.dilate(gt_t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=5)

    # Get empty beach image and mask
    img_back = cv2.imread('../P1_images/empty_beach.jpg', 0)
    mask = cv2.imread('../P1_images/mask2.jpg', 0)

    final = cv2.cvtColor(cv2.normalize(img_b2, img_b2, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_BGR2RGB)
    final = blur = cv2.GaussianBlur(final,(3,3),0)
    hls = cv2.cvtColor(final, cv2.COLOR_RGB2HLS)
    height = hls.shape[0]
    width = hls.shape[1]
    l_channel = hls[:, :, 1]
    h_channel = hls[:, :, 0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines  USEFUL
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 50)] = 1
    sxbinary = sxbinary*(mask/255)
    median = cv2.medianBlur(sxbinary.astype('uint8'), 7)

    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= 120) | (h_channel <= 15)] = 1

    print(np.max(median))
    _, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(sxbinary, cmap='gray')
    axs[0, 1].imshow(abs_sobelx, cmap='gray')
    axs[1, 0].imshow(scaled_sobel, cmap='gray')
    axs[1, 1].imshow(h_binary, cmap='gray')
    plt.show()