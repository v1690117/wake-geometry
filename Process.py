import ctypes
import os
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


def nothing(x):
    pass

def start_pause():
    global IS_PAUSE
    IS_PAUSE = not IS_PAUSE

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


def convert_image(image):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


def init_frame():
    try:
        ret, frame = cap.read()
        height, width, channels = frame.shape
        frame = cv2.flip(frame, 1)
        cv2image = cv2.resize(frame, (int(screensize[0] * 0.8), int(screensize[1] * 0.8)))
        imgtk = convert_image(cv2image)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    except Exception as exc:
        print(type(exc))
        print(exc.args)
        print(exc)
    lmain.after(10, show_frame)

def show_frame():
    if IS_PAUSE:
        try:
            ret, frame = cap.read()
            height, width, channels = frame.shape
            frame = cv2.flip(frame, 1)
            img = frame
            img = cv2.bitwise_and(img, img, mask=bit_mask)
            img = threshold(img)
            interpolation = interpolate(img)
            if interpolation != None:
                frame = draw_line(frame, interpolation)
            cv2image = cv2.resize(frame, (int(screensize[0] * 0.8), int(screensize[1] * 0.8)))
            imgtk = convert_image(cv2image)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        except Exception as exc:
            print(type(exc))
            print(exc.args)
            print(exc)
    lmain.after(5, show_frame)


def init_menu(main_window):
    menu = tkinter.Menu(main_window)
    main_window.config(menu=menu)
    # subMenu = tkinter.Menu(menu)
    # menu.add_cascade(label="Action", menu=subMenu)
    # subMenu.add_command(label="Test", command=do_smth)
    menu.add_command(label=" > ", command=start_pause)


def do_smth():
    print("test")


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
bit_mask = cv2.imread("resources/ImageMask.jpg", 0)

######

init_frame()
root.mainloop()
