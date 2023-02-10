import numpy as np
import cv2 as cv
import math
from Detector import detect
from KalmanFilter import KalmanFilter


# Create a mask from 4 points given, returns a mask with trapezoidal shape fit to the court
def make_mask(img, points):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.ones_like(img)

    n1 = (height - points[1][1]) / (points[1][1] - points[0][1])
    xe = int(points[1][0] + n1 * (points[1][0] - points[0][0]))
    xs = int(points[0][0] + n1 * (points[0][0] - points[1][0]))

    n2 = (height - points[3][1]) / (points[3][1] - points[2][1])
    xe2 = int(points[3][0] + n2 * (points[3][0] - points[2][0]))
    xs2 = int(points[2][0] + n2 * (points[2][0] - points[3][0]))

    contours = np.array([[0, 0], [xs, 0], [xe, height], [0, height]])
    cv.fillPoly(mask, pts=[contours], color=0)
    contours = np.array([[xs2, 0], [width, 0], [width, height], [xe2, height]])
    cv.fillPoly(mask, pts=[contours], color=0)
    return mask


def find_players(frame):
    # Apply mask for background subtraction
    MOG_mask = BS_MOG_play.apply(frame)
    MOG_mask = np.where(MOG_mask > 127, 1, 0)
    MOG_mask = np.stack((MOG_mask, MOG_mask, MOG_mask), axis=2)
    frame = (frame * MOG_mask).astype('uint8')

    # Find contours
    frame_players = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, frame_players = cv.threshold(frame_players, 20, 255, cv.THRESH_TOZERO)
    contours, junk = cv.findContours(frame_players, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Make output array
    out = np.zeros_like(frame)
    # Players contour area, index of contour and coordinates of center
    pl1_ar = 0
    pl1_c = -1
    pl1_x = 0
    pl1_y = 0
    pl2_ar = 0
    pl2_c = -1
    pl2_x = 0
    pl2_y = 0
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > 20: # Avoid errors with the moments
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.drawContours(out, contours[i], -1, (255, 255, 255), 1)

            if cY > top_net: # Find biggest contour on the upper part of the frame
                if area > pl1_ar:
                    pl1_ar = area
                    pl1_c = i
                    pl1_x = cX
                    pl1_y = cY
            if cY < top_net: # Find biggest contour on the lower part of the frame
                if area > pl2_ar:
                    pl2_ar = area
                    pl2_c = i
                    pl2_x = cX
                    pl2_y = cY
    if pl1_c != -1:
        out_blue = out[:, :, 0].astype('uint8')
        if pl1_y > bottom_net + 20:
            cv.circle(out_blue, (pl1_x, pl1_y), 75, 255, -1)  # Not draw the circles if the players are too cose to the net
        out[:, :, 0] = out_blue
    if pl2_c != -1:
        out_red = out[:, :, 2].astype('uint8')
        if pl2_y < top_net - 20:
            cv.circle(out_red, (pl2_x, pl2_y), 60, 255, -1)
        out[:, :, 2] = out_red
    return out


def find_ball(frame, centers):
    # Apply mask for background subtraction
    MOG_mask_ball = BS_MOG_ball.apply(frame)
    MOG_mask_ball = np.where(MOG_mask_ball > 127, 1, 0)
    MOG_mask_ball = np.stack((MOG_mask_ball, MOG_mask_ball, MOG_mask_ball), axis=2)
    frame_ball = (frame * MOG_mask_ball).astype('uint8')

    # Dilate and smooth image with mean filter
    kernel = np.ones((15, 15), np.uint8)
    r = frame_ball[:, :, 0]
    g = frame_ball[:, :, 1]
    b = frame_ball[:, :, 2]

    r = cv.dilate(r, kernel)
    g = cv.dilate(g, kernel)
    b = cv.dilate(b, kernel)

    frame_ball[:, :, 0] = r
    frame_ball[:, :, 1] = g
    frame_ball[:, :, 2] = b

    frame_ball = cv.filter2D(frame_ball, -1, kernel / 225)

    # Find contours
    frame_bell = cv.cvtColor(frame_ball, cv.COLOR_BGR2GRAY)
    contours, junk = cv.findContours(frame_bell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame_bell, contours, -1, (255, 0, 0), 3)

    # Combine images to get color back
    frame_bell = cv.cvtColor(frame_bell, cv.COLOR_GRAY2BGR)
    frame_bell = cv.cvtColor(frame_bell, cv.COLOR_BGR2HLS)
    frame_ball = cv.cvtColor(frame_ball, cv.COLOR_BGR2HLS)
    frame_bell[:, :, 0] = frame_ball[:, :, 0]
    frame_bell[:, :, 2] = 127
    frame = cv.cvtColor(frame_bell, cv.COLOR_HLS2BGR)

    # Detect object
    center = detect(frame, debugMode)
    out = np.zeros((frame.shape[0:2]))

    # If centroids are detected then track them
    if len(centers) == 0:
        centers = center

    if (len(center) > 0):

        if math.dist(centers[0], center[0]) < 250:  # This helps to erase false detections
            centers = center

        # Draw the detected circle
        cv.circle(out, (int(centers[0][0]), int(centers[0][1])), 10, 255, -1)

        # Predict
        (x, y) = KF.predict()

        # Update
        (x1, y1) = KF.update(centers[0])
    return out, centers


def register_hit(frame, hitlist, lasthit):
    hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    # Defining mask for detecting color
    mask_yel = cv.inRange(hsv, low_yel, up_yel)
    mask_cy = cv.inRange(hsv, low_cy, up_cy)

    # First hit
    if lasthit == -1:
        if 255 in mask_yel:
            hitlist.append(1)  # It means the top player hit the ball first
            lasthit = 1
            print("The top player made the first hit")

        if 255 in mask_cy:
            hitlist.append(2)  # It means the bottom player hit the ball first
            lasthit = 2
            print("The bottom player made the first hit")

    elif lasthit == 1:  # Last hit from top player, wait for bottom player
        if 255 in mask_cy:
            hitlist.append(2)  # It means the bottom player hit the ball
            lasthit = 2
            print("Hit by bottom player")

    elif lasthit == 2:  # Last hit from bottom player, wait for top player
        if 255 in mask_yel:
            hitlist.append(1)  # It means the top player hit the ball
            lasthit = 1
            print("Hit by top player")

    return hitlist, lasthit


cap = cv.VideoCapture('video_cut2.mp4')

# Take first frame of the video and resize it
_, frame = cap.read()
frame = cv.resize(frame, (960, 540), fx=0, fy=0, interpolation=cv.INTER_CUBIC)

# Create mask specific to camera position
#mask = make_mask(frame, [[260, 150], [150, 430], [700, 150], [820, 430]])  # blue vid
#top_net, bottom_net = [220, 260]
mask = make_mask(frame, [[310, 100], [130, 420], [640, 100], [830, 420]])  # red vid
top_net, bottom_net = [170, 200]

frame = frame * mask

# Create MOG object
BS_MOG_play = cv.bgsegm.createBackgroundSubtractorMOG(history=200)
BS_MOG_ball = cv.bgsegm.createBackgroundSubtractorMOG(history=10)

ControlSpeedVar = 100  # Lowest: 1 - Highest:100
HiSpeed = 100
KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
debugMode = 0
centers = []

hitlist = []
lasthit = -1

low_cy = np.array([20, 0, 0], dtype = "uint8")
up_cy = np.array([40, 255, 255], dtype = "uint8")
low_yel = np.array([80, 0, 0], dtype = "uint8")
up_yel = np.array([100, 255, 255], dtype = "uint8")

while 1:
    # Capture frame, resize and apply mask
    ret, frame = cap.read()

    if ret:
        frame = cv.resize(frame, (960, 540), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        frame = frame * mask
        out = find_players(frame)
        out[:, :, 1], centers = find_ball(frame, centers)
        hitlist, lasthit = register_hit(out, hitlist, lasthit)

        out[top_net - 20, :, :] = 255  # these two lines are just to give the viewer an idea of where the net is
        out[bottom_net + 20, :, :] = 255

        cv.imshow('img', out)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cv.destroyAllWindows()
cap.release()


print("\nTop player hit the ball", hitlist.count(1), "times")
print("Bottom player hit the ball", hitlist.count(2), "times")

if lasthit == 1:
    print("The top player won the point")
elif lasthit == 2:
    print("The bottom player won the point")
