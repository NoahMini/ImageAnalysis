import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def train_and_test_model(img_bgr, ground_truth):
    """ Trains and tests a model to classify superpixels. """
    # Read sample image
    #img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)
    # Infer (automatically) which pixels will be set as positives.

    # Superpixel segmentation
    sh = img_bgr.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image_width=sh[1], image_height=sh[0], image_channels=sh[2], num_superpixels=300, num_levels=3)
    seeds.iterate(img=img_bgr, num_iterations=10)
    region_labels = seeds.getLabels()

    # Get features
    # X = get_geometric_features(region_labels)
    # X = get_photometric_features(img_bgr, region_labels)
    X = np.concatenate([get_geometric_features(region_labels), get_photometric_features(img_bgr, region_labels)], axis=1)

    # Get labels
    positives_per_region = []
    for idx_l in range(np.max(region_labels) + 1):
        region_mask = (region_labels == idx_l)
        positives_per_region.append(np.sum(ground_truth[region_mask])!=0)
    y = positives_per_region

    # Create model
    # model = LogisticRegression()
    model = RandomForestClassifier()

    # Train model
    model.fit(X, y)

    # Make predictions
    predictions_per_region = model.predict(X)
    predictions = predictions_per_region[region_labels]

    # Better visualizations
    border_mask = region_labels != cv2.erode(region_labels.astype('uint8'), np.ones((3, 3)))
    img_with_borders = np.copy(img_bgr)
    img_with_borders[border_mask, ...] = (255, 0, 0)
    img_with_gt = np.copy(img_bgr)
    img_with_gt[ground_truth == 255] = (0, 0, 255)
    img_with_gt[border_mask, ...] = (255, 0, 0)
    img_with_pred = np.copy(img_bgr)
    img_with_pred[predictions == 1] = (0, 255, 0)
    img_with_pred[border_mask, ...] = (255, 0, 0)

    _, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    axs[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 1].imshow(cv2.cvtColor(img_with_borders, cv2.COLOR_BGR2RGB))
    axs[1, 0].imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Ground truth')
    axs[1, 1].imshow(cv2.cvtColor(img_with_pred, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Predictions')
    plt.show()


def get_geometric_features(region_labels:np.ndarray) -> np.ndarray:
    """ Computes geometric features for each region in the image. """
    features = []
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)

        # Compute geometric features for this region.
        area = np.sum(region_mask)
        perimeter = np.sum(region_mask != cv2.erode(region_mask.astype('uint8'), np.ones((3, 3))))
        pixel_locations = np.where(region_mask)
        centroid_x = np.mean(pixel_locations[0])
        centroid_y = np.mean(pixel_locations[1])
        std_x = np.std(pixel_locations[0])
        std_y = np.std(pixel_locations[1])
        roundness = 4 * np.pi * area / (perimeter**2)

        # Store them
        features.append([
            area,
            perimeter,
            centroid_x,
            centroid_y,
            std_x,
            std_y,
            roundness,
        ])

    # Output as a numpy array indexed as [region_idx, feature_idx].
    X = np.array(features)
    return X


def get_photometric_features(img_bgr: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    """ Computes photometric features for each region in the image. """
    features = []
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)

        # Compute photometric features for this region.
        max_red_value = img_bgr[region_mask, 2].max()
        mean_red_value = np.mean(img_bgr[region_mask, 2])
        std_red_value = np.std(img_bgr[region_mask, 2])
        max_green_value = img_bgr[region_mask, 1].max()
        mean_green_value = np.mean(img_bgr[region_mask, 1])
        std_green_value = np.std(img_bgr[region_mask, 1])
        max_blue_value = img_bgr[region_mask, 0].max()
        mean_blue_value = np.mean(img_bgr[region_mask, 0])
        std_blue_value = np.std(img_bgr[region_mask, 0])

        # Store them
        features.append([
            max_red_value,
            mean_red_value,
            std_red_value,
            max_green_value,
            mean_green_value,
            std_green_value,
            max_blue_value,
            mean_blue_value,
            std_blue_value,
        ])

    # Output as a numpy array indexed as [region_idx, feature_idx].
    X = np.array(features)
    return X

def features_gabor_filter_bank(img):
    kernels = [
        cv2.getGaborKernel(ksize=(15, 15), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=0)
        for sigma in [3, 5, 7]
        for theta in [np.pi, np.pi / 2, 0]
        for lambd in [1.5, 2]
        for gamma in [1, 1.5]
    ]
    filtered_images = [cv2.filter2D(img, cv2.CV_64F, kernel) for kernel in kernels]

    # Create features
    X = np.stack([f.flatten() for f in filtered_images], axis=-1)
    return X


def features_eigenvalues_hessian(img):
    hessian_dxdx = cv2.Sobel(img, cv2.CV_16S, 2, 0, ksize=3)
    hessian_dxdy = cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=3)
    hessian_dydx = hessian_dxdy
    hessian_dydy = cv2.Sobel(img, cv2.CV_16S, 0, 2, ksize=3)

    hessian_det = hessian_dxdx * hessian_dydy - hessian_dxdy * hessian_dydx
    hessian_trace = hessian_dxdx + hessian_dydy
    # Solve `x^2 - trace * x + det = 0`
    hessian_eigenvalue_1 = 0.5 * (hessian_trace + np.sqrt(hessian_trace**2 - 4 * hessian_det))
    hessian_eigenvalue_2 = 0.5 * (hessian_trace - np.sqrt(hessian_trace**2 - 4 * hessian_det))

    X = np.stack([hessian_det.flatten(), hessian_trace.flatten(), hessian_eigenvalue_1.flatten(), hessian_eigenvalue_2.flatten()], axis=-1)
    return X

def remove_back(img_front: np.ndarray, img_back: np.ndarray, img_mask: np.array):
    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    mask_r = img_mask/255
    im = (img_front - img_back)*mask_r
    return im.astype('uint8')

def make_gt(lab_im):
    center = lab_im[[1, 2]].to_numpy() #col0 = y and col1 = x
    base = np.zeros((lab_im.iloc[0][5], lab_im.iloc[0][4]))
    for i in range(center.shape[0]):
        base = cv2.circle(base, (center[i, 0], center[i, 1]), 7, 255, -1)
    return base/255



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
    center9 = lab_b9[[1, 2]].to_numpy()
    gt_b10 = make_gt(lab_b10)


    # Get empty beach image and mask
    img_back = cv2.imread('../P1_images/empty_beach.jpg', 0)
    mask = cv2.imread('../P1_images/mask.jpg', 0)

    # Do image prep
    n = remove_back(img_b9, img_back, mask) #output grey masked image
    thresh = cv2.adaptiveThreshold(n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    _, n_thre = cv2.threshold(n, 200, 255, cv2.THRESH_TOZERO_INV)
    _, n_thre2 = cv2.threshold(n_thre, 100, 255, cv2.THRESH_TOZERO)
    close = cv2.morphologyEx(n_thre2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dilated = cv2.morphologyEx(close, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    mult = thresh*dilated + n_thre2
    mult[np.where(mult >= 150)] = 255
    mdi = cv2.morphologyEx(mult, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    img_color = cv2.cvtColor(img_b9, cv2.COLOR_BGR2HLS)
    img_color[:, :, 1] = img_color[:, :, 1]*(dilated/255)
    imcol = cv2.cvtColor(img_color, cv2.COLOR_HLS2RGB)

    contours, hierarchy = cv2.findContours(mult, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(img_b9,(x,y),(x+w,y+h),(0, 0, 255), 2)
    #cv2.drawContours(img_b9, contours, -1, (0, 255, 0), 3)
    #train_and_test_model(imcol, gt_b9)
    plt.imshow(img_b9)
    plt.show()


