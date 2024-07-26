import numpy as np
import cv2
import matplotlib.pyplot as plt

image_01 = cv2.imread("Images\cat.bmp")
image_01_rgb =  cv2.cvtColor(image_01, cv2.COLOR_BGR2RGB)

image_02 = cv2.imread("Images\dog.bmp")
image_02_rgb =  cv2.cvtColor(image_02, cv2.COLOR_BGR2RGB)


# gettig the dimensions of the images
print(image_01_rgb.shape)
print(image_02_rgb.shape)

# large blur filter on the first image to get the low frequency if the image
# high ferequence
# add two proceesed images together to get the hybrid

image_01_blurred = cv2.GaussianBlur(image_01_rgb, (25, 25),0)
image_02_blurred = cv2.GaussianBlur(image_02_rgb, (15, 15),0)
image_02_highpass = cv2.subtract(image_02_rgb, image_02_blurred)

# Applying sobel kernel using filer2D
# edges_x = cv2.filter2D(image_02_rgb, -1, (5, 5))
# edges_y = cv2.filter2D(image_02_rgb, -1, 

def highpass(im, sigma):
    ker_size = int(np.ceil(4*sigma +1))
    if ker_size %2 == 0:
        ker_size += 1

    # creating gaussian kernel
    gaussian_kernel_1D = cv2.getGaussianKernel(ker_size, sigma)
    gaussian_kernel_2D = gaussian_kernel_1D @gaussian_kernel_1D.T

    low_pass = cv2.filter2D(im, -1, gaussian_kernel_2D, borderType = cv2.BORDER_CONSTANT)
    im_res = im.astype(np.float32) - low_pass.astype(np.float32)
    im_res[im_res == im] = 0
    return im_res


plt.figure(figsize= (10, 10))
plt.subplot(3,3,1)
plt.title("Image 01")
plt.imshow(image_01_rgb)

plt.subplot(3,3,2)
plt.title("Image 02")
plt.imshow(image_02_rgb)

plt.subplot(3,3,4)
plt.title("Image_01_blur")
plt.imshow(image_01_blurred)

# plt.subplot(3,3,5)
# plt.title("Image_02_sharp")
# plt.imshow(image_02_blurred)

# plt.subplot(3,3,5)
# plt.title("Image_02_sharp")
# plt.imshow(image_02_highpass)

plt.subplot(3,3,5)
plt.title("Image_02_sharp_2")
sigma = 1.2
im_res = highpass(image_02_rgb, sigma)
plt.imshow(im_res.astype(np.uint8))

plt.show()