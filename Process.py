import ctypes
import os
import time
import tkinter

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk

IS_PAUSE = False
TOOLBAR_WIN_NAME = 'toolbar'
H_LOW = 55
H_HIGH = 89
S_LOW = 89
S_HIGH = 179
V_LOW = 200
V_HIGH = 255
TOP_LIMIT_LEN = 800
BOTTOM_LIMIT_LEN = 500
START_INDEX = 300

sum_frame = None
ret = None
frame = None
mask_dict = None
bit_mask = None
line_masks = []


def nothing(x):
    pass


def start_pause():
    global IS_PAUSE
    IS_PAUSE = not IS_PAUSE


def save_frame():
    global sum_frame
    filename = time.time()
    cv2.imwrite("resources/" + str(filename) + ".jpg", sum_frame)


def init():
    cv2.namedWindow(TOOLBAR_WIN_NAME)
    cv2.createTrackbar('HU', TOOLBAR_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar('HL', TOOLBAR_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar('SU', TOOLBAR_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar('SL', TOOLBAR_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar('VU', TOOLBAR_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar('VL', TOOLBAR_WIN_NAME, 0, 255, nothing)

    cv2.setTrackbarPos('HU', TOOLBAR_WIN_NAME, H_HIGH)
    cv2.setTrackbarPos('HL', TOOLBAR_WIN_NAME, H_LOW)
    cv2.setTrackbarPos('SU', TOOLBAR_WIN_NAME, S_HIGH)
    cv2.setTrackbarPos('SL', TOOLBAR_WIN_NAME, S_LOW)
    cv2.setTrackbarPos('VU', TOOLBAR_WIN_NAME, V_HIGH)
    cv2.setTrackbarPos('VL', TOOLBAR_WIN_NAME, V_LOW)


def threshold(img):
    hu = cv2.getTrackbarPos('HU', TOOLBAR_WIN_NAME)
    hl = cv2.getTrackbarPos('HL', TOOLBAR_WIN_NAME)
    su = cv2.getTrackbarPos('SU', TOOLBAR_WIN_NAME)
    sl = cv2.getTrackbarPos('SL', TOOLBAR_WIN_NAME)
    vu = cv2.getTrackbarPos('VU', TOOLBAR_WIN_NAME)
    vl = cv2.getTrackbarPos('VL', TOOLBAR_WIN_NAME)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hl, sl, vl])
    upper = np.array([hu, su, vu])
    mask = cv2.inRange(hsv, lower, upper)

    # kernel = np.ones((4, 4), np.float32) / 255
    # mask = cv2.filter2D(mask, -1, kernel)
    # mask = fgbg.apply(mask)
    return mask


y_mean = None


def interpolate(points):
    f = None
    if len(points) > BOTTOM_LIMIT_LEN:
        y = points[:, 0]
        x = points[:, 1]

        if (str(y_mean) in mask_dict):
            x = np.append([mask_dict[str(y_mean)][0]-200], x)
            y = np.append([y_mean], y)
            x = np.append(x, [[mask_dict[str(y_mean)][1]+200]])
            y = np.append(y, [y_mean])

        if len(x) == 0:
            return None
        z = np.polyfit(x, y, 50)  # RankWarning: Polyfit may be poorly conditioned
        f = np.poly1d(z)
    return f


def center_mass_filter(img):
    res = img
    if len(res.shape) == 2:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    points = np.array(np.argwhere(res > 250))
    try:
        global y_mean
        y_mean = (np.mean(points[:, 0]))
        x_mean = (np.mean(points[:, 1]))
        if np.isnan(x_mean) or np.isnan(y_mean) or x_mean > 1150 or x_mean < 950:
            return np.array([])
        x_mean = int(x_mean)
        y_mean = int(y_mean)
        c_x = 0.3
        c_y = 0.1
        x_ = int(c_x * np.ptp(points[:, 1]))
        y_ = int(c_y * np.ptp(points[:, 0]))
        mask1 = abs(points[:, 1] - x_mean) < x_
        mask2 = abs(points[:, 0] - y_mean) < y_
        mask_total = mask1 & mask2
        res_points = points[mask_total, :]
    except Exception as exc:
        print(type(exc))
        print(exc.args)
        print(exc)
        res_points = np.array([])

    return res_points


def fast():
    example = np.ones([500, 500, 500], dtype=np.uint8)
    img = example.copy()
    height, width, depth = img.shape
    img[0:height, 0:width // 4, 0:depth] = 0  # DO THIS INSTEAD
    return img


def draw_line(img, f):
    result = img
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    try:
        h, w, c = result.shape
        frm, to = mask_dict['0'][0], mask_dict['0'][1]
        for i in range(frm, to):
            value = f(i)
            if h > value > 0 and str(int(value)) in mask_dict and mask_dict[str(int(value))][0] < i and \
                            mask_dict[str(int(value))][1] > i:
                result[int(value), i] = [0, 0, 255]
    except Exception as exc:
        print(type(exc))
        print(exc.args)
        print(exc)

    return result


def convert_image(image):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


def init_frame():
    try:
        global frame
        ret, frame = cap.read()
        global sum_frame
        if sum_frame is None:
            sum_frame = frame
        show_image(frame, lmain)
    except Exception as exc:
        print(type(exc))
        print(exc.args)
        print(exc)
    lmain.after(10, show_frame)


def show_image(show_img, lmain):
    cv2image = cv2.resize(show_img, (int(screensize[0] * 0.8), int(screensize[1] * 0.8)))

    cv2image = cv2.flip(cv2image, 0)
    cv2image = cv2.flip(cv2image, 1)
    imgtk = convert_image(cv2image)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)


count = 0

x_s = []


def process_lines(img):
    res = cv2.bitwise_and(img, img, mask=line_masks[0])
    line = np.array(np.argwhere(res > 250))
    return line



def show_frame():
    if IS_PAUSE:
        try:
            global frame
            ret, frame = cap.read()
            global count
            count = count + 1
            if count > START_INDEX:
                img = frame
                img = threshold(img)
                line = process_lines(img)
                if len(line) <= 5:
                    img = cv2.bitwise_and(img, img, mask=bit_mask)
                    points = center_mass_filter(img)
                    interpolation = interpolate(points)
                else:
                    interpolation = None
                global sum_frame
                show_img = sum_frame
                if interpolation != None:
                    show_img = draw_line(sum_frame, interpolation)
                #     show_img = draw_line(sum_frame, interpolation)
                    # show_img = draw_line(img, interpolation)
                show_image(show_img, lmain)
        except Exception as exc:
            print(type(exc))
            print(exc.args)
            print(exc)
    lmain.after(5, show_frame)


def init_menu(main_window):
    menu = tkinter.Menu(main_window)
    main_window.config(menu=menu)
    subMenu = tkinter.Menu(menu)
    menu.add_cascade(label="Lines", menu=subMenu)
    subMenu.add_command(label="Def line", command=def_line)
    menu.add_command(label=" > ", command=start_pause)
    menu.add_command(label=" saveFrame", command=save_frame)


def def_line():
    cv2.namedWindow('DefLine')
    cv2.setMouseCallback("DefLine", get_point)
    while (1):
        cv2.imshow("DefLine", frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


ix_1 = None
iy_1 = None
ix_2 = None
iy_2 = None


def get_point(event, x, y, flags, param):
    global ix_1, iy_1, ix_2, iy_2, frame
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if ix_1 is None:
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            ix_1, iy_1 = x, y
        elif ix_2 is None:
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            ix_2, iy_2 = x, y
        else:
            k = (iy_2 - iy_1) / (ix_2 - ix_1)
            b = iy_1 - k * ix_1
            ix_0 = int(-b / k)
            height, width, depth = frame.shape
            ix_ = int((height - b) / k)
            blank_image = np.zeros((height, width, 3), np.uint8)
            cv2.line(blank_image, (ix_, height), (ix_0, 0), (255, 255, 255), thickness=3, lineType=8)
            cv2.imwrite("resources/Line1.jpg", blank_image)


def get_file():
    filepath = "resources"
    filename = "047_5623.MOV"
    return os.path.join(filepath, filename)


def get_cap():
    return cv2.VideoCapture(get_file())


root = tkinter.Tk()  # Makes main window
root.wm_title("Hello World MF")
root.config(background="#FFFFFF")
init_menu(root)

imageFrame = tkinter.Frame(root, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

lmain = tkinter.Label(imageFrame)
lmain.grid(row=0, column=0)

cap = get_cap()

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

#######


init()


def read_lines():
    line_channel_paths = ["resources/Line1.jpg"]
    global line_masks
    for path in line_channel_paths:
        line_masks.append(cv2.imread(path, 0))


def read_mask():
    global bit_mask, mask_dict
    bit_mask = cv2.imread("resources/ImageMask.jpg", 0)
    interest_area = np.array(np.argwhere(bit_mask > 250))
    mask_dict = dict()
    for i in range(len(interest_area)):
        key = str(interest_area[i][0])
        value = interest_area[i][1]
        if key in mask_dict:
            val = mask_dict[key]
            new_val = None
            if val[0] > value:
                new_val = (value, val[1])
            elif val[1] < value:
                new_val = (val[0], value)
            if new_val is not None:
                mask_dict[key] = new_val
        else:
            mask_dict[key] = (value, value)
    read_lines()


read_mask()
init_frame()
root.mainloop()
