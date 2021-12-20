import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib.colors import LogNorm

def DFFTnp(image):
    f = np.fft.fft2(image)
    forceShift = np.fft.fftshift(f)
    return forceShift

def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft)
    reverse_image = np.fft.ifft2(f_ishift)
    return reverse_image

def showDFFT(fft, name):
    image = np.uint8(cv.imread(name, 1))
    plt.subplot(121),plt.imshow(image,'Greys')
    plt.title('Input Image '+ os.path.basename(name)),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(np.abs(fft), norm=LogNorm(vmin=5))
    plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
    plt.show()


folder_path = r".//images/"
images = glob.glob(folder_path + '*.png')
for name in images:
    image = np.float32(cv.imread(name, 0))
    forceShift = DFFTnp(image)
    showDFFT(forceShift, name)
    ratio = 0.9980
    width, height = forceShift.shape
    middle_pixel = forceShift[width//2][height//2]

    for i in range(width):
        for j in range(height):
            if i == width//2 and j == height//2:
                continue;
            if abs(np.abs(forceShift[i][j])-np.abs(middle_pixel)) < np.abs(middle_pixel)*ratio:
                forceShift[i][j] = 0
    reverse_image = reverseDFFTnp(forceShift)
    plt.imshow(abs(reverse_image), cmap ='gray')
    plt.title('Output Image '+ os.path.basename(name)),plt.xticks([]),plt.yticks([])
    plt.show()
