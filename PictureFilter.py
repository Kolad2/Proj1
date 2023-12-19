import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('includes/Pictures/B21-166b_cut.tif', 1)

def PictureContr(img):
      # converting to LAB color space
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l_channel, a, b = cv2.split(lab)
      # Applying CLAHE to L-channel
      # feel free to try different values for the limit and grid size:
      # clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(16,16))
      # cl = clahe.apply(l_channel)
      # merge the CLAHE enhanced L-channel with the a and b channel
      limg = cv2.merge((l_channel,a,b))
      # Converting image from LAB Color model to BGR color space
      return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def SimpleSharpFilter(img):
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l_c, a, b = cv2.split(lab)
      # Create the sharpening kernel
      kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
      # Sharpen the image
      l_c = cv2.filter2D(l_c, -1, kernel)
      return cv2.cvtColor(cv2.merge((l_c, a, b)), cv2.COLOR_LAB2BGR)


def HightPass_Ps(img):
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l_c, a, b = cv2.split(lab)
      l_g = cv2.GaussianBlur(l_c, (0,0), 3)
      l_c0 = ((l_c) - (l_g))
      l_c0[l_c < l_g] = 0
      l_c = l_c0 + 127
      l_c[l_c0 > 128] = 255
      return cv2.cvtColor(cv2.merge((l_c, a, b)), cv2.COLOR_LAB2BGR)


def GetGrrayFFT(img):
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l_c, a, b = cv2.split(lab)
      return np.fft.fft2(l_c)



lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_c, a, b = cv2.split(lab)
lap = cv2.Laplacian(l_c,cv2.CV_64F)
sx = cv2.Sobel(l_c,cv2.CV_64F,1,0,ksize=3)
sy = cv2.Sobel(l_c,cv2.CV_64F,0,1,ksize=3)
sxy = cv2.Sobel(l_c,cv2.CV_64F,1,1,ksize=3)
s = sx**2 + sy**2

x0 = 2000
y0 = 4300

img_f = GetGrrayFFT(img)
img_f = np.fft.fftshift(img_f)
img_fs = np.log2(abs(img_f))


print(img_f.dtype)
print(img_fs.dtype)

# fig 3
fig = plt.figure(figsize=(10,5))
ax = [fig.add_subplot(1, 2, 1),
      fig.add_subplot(1, 2, 2)]
M = np.max(img_fs)
ax[0].imshow(img_fs/M)
ax[1].imshow(img_fs/M)
plt.show()

exit()

img01 = HightPass_Ps(img)

img1 = img[x0:x0+1000, y0:y0+1000]
img2 = img01[x0:x0+1000, y0:y0+1000]



fig = plt.figure(figsize=(10,5))
ax = [fig.add_subplot(1, 2, 1),
      fig.add_subplot(1, 2, 2)]
G1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
G2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#ax[0].imshow(cv2.merge((G1,G1,G1)))
#ax[1].imshow(cv2.merge((G2,G2,G2)))
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[0].axis('off')
ax[1].axis('off')
ax[0].sharey(ax[1])
ax[0].sharex(ax[1])


# fig 2
fig = plt.figure(figsize=(10,5))
ax = [fig.add_subplot(1, 2, 1),
      fig.add_subplot(1, 2, 2)]
ax[0].hist(img1.ravel(),256,[0,256])
ax[1].hist(img2.ravel(),256,[0,256])
plt.show()




exit()






