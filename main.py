import cv2
from matplotlib import pyplot as plt
import numpy as np


def question_1():
    img1 = cv2.imread(r"C:\Users\Nitzan\PycharmProjects\MIP_HW3\MIP_HW3\img1.png")
    # Display images with titles
    fig = plt.figure(figsize=(15, 15))
    rgb_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    fig.add_subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(rgb_img)

    # Section 1a
    h, w, c = img1.shape
    print(f"the image dimensions are ({w}, {h})")

    # Section 1
    r, g, b = rgb_img[100, 150]
    print(f"RGB values:  {r}, {g}, {b}")

    # Section 2a
    img_resized = cv2.resize(rgb_img, (256, 256))
    fig.add_subplot(2, 4, 2)
    plt.title("Resized")
    plt.imshow(img_resized)

    # Section 2b
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    fig.add_subplot(2, 4, 3)
    plt.title("Grayscale")
    plt.imshow(gray_img, cmap='gray')

    # Section 2c
    rotated_img = cv2.rotate(rgb_img, cv2.ROTATE_180)
    fig.add_subplot(2, 4, 4)
    plt.title("Rotated")
    plt.imshow(rotated_img)

    # Section 3a
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mag_thresh = np.max(mag) * 0.1
    sobel_edges = np.zeros_like(mag)
    sobel_edges[mag > mag_thresh] = 255
    sobel_edges[mag <= mag_thresh] = 0
    fig.add_subplot(2, 4, 5)
    plt.title("Sobel")
    plt.imshow(sobel_edges, cmap='gray')

    blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    thresh = np.max(laplacian) * 0.1
    log_edges = np.zeros_like(laplacian)
    log_edges[laplacian > thresh] = 255
    log_edges[laplacian <= thresh] = 0
    fig.add_subplot(2, 4, 6)
    plt.title("LOG")
    plt.imshow(log_edges, cmap='gray')

    # Section 3b
    img2 = cv2.imread(r"C:\Users\Nitzan\PycharmProjects\MIP_HW3\MIP_HW3\img2.png")
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    thresh = np.mean(gray_img)
    while True:
        mu1 = np.mean(gray_img[gray_img > thresh])
        mu2 = np.mean(gray_img[gray_img < thresh])
        new_thresh = np.round((mu1+mu2)/2)
        if new_thresh == thresh:
            break
        else:
            thresh = new_thresh
    thresholded = np.zeros_like(gray_img)
    thresholded[gray_img > thresh] = 255
    thresholded[gray_img <= thresh] = 0
    fig.add_subplot(2, 4, 7)
    plt.title("Threshold")
    plt.imshow(thresholded, cmap='gray')

    # Section 4a
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 60, param1=10, param2=17, minRadius=30, maxRadius=40)
    # Draw detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img2, (x, y), r, (0, 0, 255), 2)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    fig.add_subplot(2, 4, 8)
    plt.title("Hough")
    plt.imshow(rgb_img2)
    plt.show()


if __name__ == '__main__':
    question_1()


