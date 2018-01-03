import os
import cv2
import numpy as np


def nothing(x):
    pass


def init(toolbar_window_name):
    cv2.namedWindow(toolbar_window_name)
    cv2.createTrackbar('HU', toolbar_window_name, 0, 255, nothing)
    cv2.createTrackbar('HL', toolbar_window_name, 0, 255, nothing)
    cv2.createTrackbar('SU', toolbar_window_name, 0, 255, nothing)
    cv2.createTrackbar('SL', toolbar_window_name, 0, 255, nothing)
    cv2.createTrackbar('VU', toolbar_window_name, 0, 255, nothing)
    cv2.createTrackbar('VL', toolbar_window_name, 0, 255, nothing)

    cv2.setTrackbarPos('HU', toolbar_window_name, 89)
    cv2.setTrackbarPos('HL', toolbar_window_name, 55)
    cv2.setTrackbarPos('SU', toolbar_window_name, 179)
    cv2.setTrackbarPos('SL', toolbar_window_name, 87)
    cv2.setTrackbarPos('VU', toolbar_window_name, 255)
    cv2.setTrackbarPos('VL', toolbar_window_name, 200)


def threshold(img, toolbar_window_name):
    hu = cv2.getTrackbarPos('HU', toolbar_window_name)
    hl = cv2.getTrackbarPos('HL', toolbar_window_name)
    su = cv2.getTrackbarPos('SU', toolbar_window_name)
    sl = cv2.getTrackbarPos('SL', toolbar_window_name)
    vu = cv2.getTrackbarPos('VU', toolbar_window_name)
    vl = cv2.getTrackbarPos('VL', toolbar_window_name)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hl, sl, vl])
    upper = np.array([hu, su, vu])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((4, 4), np.float32) / 255
    # mask = cv2.filter2D(mask, -1, kernel)
    # mask = fgbg.apply(mask)

    return mask


def interpolate(img):
    points = np.array(np.argwhere(img > 250))
    y = points[:, 0]
    x = points[:, 1]
    if len(x) == 0:
        return None
    z = np.polyfit(x, y, 50)
    f = np.poly1d(z)
    return f


def draw_line(img, f):
    result = img
    h, w, c = result.shape
    for i in range(800, 1400):
        value = f(i)
        if h > value > 0:
            result[int(value), i] = [0, 0, 255]
    return result


filepath = "resources"
filename = "047_5623.MOV"

img_window_name = 'image'
toolbar_window_name = 'toolbar'

init(toolbar_window_name)

cap = cv2.VideoCapture(os.path.join(filepath, filename))
fgbg = cv2.createBackgroundSubtractorMOG2()

bit_mask = cv2.imread("resources/ImageMask.jpg", 0)

read_frame = True
while True:
    if read_frame:

        try:
            ret, frame = cap.read()
            height, width, channels = frame.shape
            img = frame
            img = cv2.bitwise_and(img, img, mask=bit_mask)
            img = threshold(img, toolbar_window_name)

            points = np.array(np.argwhere(img == 255))

            interpolation = interpolate(img)
            result = frame
            if interpolation != None:
                frame = draw_line(frame, interpolation)
        except Exception as exc:
            print(type(exc))
            print(exc.args)
            print(exc)

        # frame= cv2.resize(frame, (int(width * 0.6), int(height * 0.6)))
        cv2.imshow('frame', frame)
        # cv2.imwrite("ImageOrig.jpg", frame)
        # cv2.imwrite("ImageThresholded.jpg", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == 32:
        read_frame = not read_frame

cap.release()
cv2.destroyAllWindows()
