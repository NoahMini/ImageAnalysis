import cv2
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def remove_back(img_front: np.ndarray, img_back: np.ndarray):
    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    im = (img_front - img_back)
    return im.astype('uint8')

def apply_mask(img, mask):
    imout = img*(mask/255)
    return imout.astype('uint8')

def make_gt(lab_im):
    center = lab_im[[1, 2]].to_numpy()
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
    _, mult = cv2.threshold(mult, 127, 255, cv2.THRESH_BINARY)
    mdi = cv2.morphologyEx(mult, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mdi[np.where(mdi != 0)] = 1
    return mdi

def do_sobelx(img):
    fan = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    final = cv2.cvtColor(fan , cv2.COLOR_BGR2RGB)
    final = cv2.GaussianBlur(final,(3,3),0)
    hls = cv2.cvtColor(final, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines  USEFUL
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    _, scaled_sobelx = cv2.threshold(scaled_sobelx, 50, 255, cv2.THRESH_BINARY)
    sxbinary = scaled_sobelx*(mask/255)
    medianx = cv2.medianBlur(sxbinary.astype('uint8'), 7)

    # Sobel y
    sobelxx = cv2.Sobel(l_channel, cv2.CV_64F, 2, 0)  # Take the derivative in x
    abs_sobelxx = np.absolute(sobelxx)  # Absolute x derivative to accentuate lines  USEFUL
    scaled_sobelxx = np.uint8(255 * abs_sobelxx / np.max(abs_sobelxx))

    # Threshold x gradient
    _, scaled_sobelxx = cv2.threshold(scaled_sobelxx, 50, 255, cv2.THRESH_BINARY)
    sxxbinary = scaled_sobelxx * (mask / 255)
    medianxx = cv2.medianBlur(sxxbinary.astype('uint8'), 7)

    return medianx, medianxx

def get_boxes(image_base, image_out):
    contours, hierarchy = cv2.findContours(image_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    im_check = cv2.cvtColor(np.zeros_like(image_base), cv2.COLOR_GRAY2RGB)
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (w > 10 and h > 10):
            cv2.rectangle(image_out, (x-15, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(im_check, (x-15, y), (x + w, y + h), (0, 255, 255), -1)
            count += 1
    return image_out, count, im_check

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

    # Get empty beach image and mask
    img_back = cv2.imread('../P1_images/empty_beach.jpg', 0)
    mask = cv2.imread('../P1_images/mask4.jpg', 0)

    images = [img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7, img_b8, img_b9, img_b10]
    grounds = [gt_b1, gt_b2, gt_b3, gt_b4, gt_b5, gt_b6, gt_b7, gt_b8, gt_b9, gt_b10]

    mse = 0
    av = 0

    #for im_targ, gt in zip(images, grounds):
    im_targ = img_b9
    gt = gt_b9
    #cv2.imwrite('out_gt9.jpg', gt.astype('uint8'))

    # Do thresholding
    im1 = remove_back(im_targ, img_back) #output grey masked image
    im1 = apply_mask(im1, mask)


    # Apply thresholding
    mdi = do_thresh(im1)
    #cv2.imwrite('out_mdi.jpg', mdi.astype('uint8'))

    # Apply Sobelx and Sobelxx
    medianx, medianxx = do_sobelx(im_targ)
    im2 = medianx + medianxx


    # Add results of thresholding and sobel
    summ = cv2.bitwise_or(mdi, im2)
    sumd = cv2.dilate(summ, np.ones((5, 5), np.uint8), iterations=1)


    # Find contours and put boxes around these contours
    img_out, count, im_check = get_boxes(sumd, im_targ)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    im_check[:, :, 0] = gt * 255
    imx = im_check[:, :, 0]/255 + im_check[:, :, 1]/255 + im_check[:, :, 2]/255
    found = math.ceil(np.count_nonzero(imx == 3) / 154)
    nfound = math.floor(np.count_nonzero(imx == 1)/154)
    num = found + nfound
    mse += 1/10*(abs(count-num)**2)
    av += 1/10*(abs(count-num))

    print("Boxes: ", count)
    print("Not found: ", nfound)
    print("Found: ", found)
    _, axs = plt.subplots(2, 1)
    axs[0].imshow(gt, cmap='gray')
    axs[1].imshow(mdi, cmap='gray')
    plt.show()

    print("Mean Squared Error  : ", mse)
    print("Average error : ", av)