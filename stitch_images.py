import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import signal

num_images = 20
images = []
for i in range(11,num_images+11):
    images.append(cv2.cvtColor(cv2.imread("audio_images/img_{:04d}.png".format(i)),cv2.COLOR_BGR2GRAY).astype('float32'))


# window = 247
# y_shift = 3
# big_image = np.zeros((480+y_shift*num_images,640+window*num_images,3),dtype='float64')
# for i,image in enumerate(images[::-1]):
#     if image is None:
#         print(num_images-i,"image is not valid")
#         continue
#     big_image[i*y_shift:480+i*y_shift,i*window:i*window+640] += image/3
# plt.imshow(big_image.astype('uint8'))

windows = [0]
y_shifts = [0]


corr_norm = signal.correlate(np.ones((480,640)), np.ones((480,640)), mode='full',method='fft')
# plt.figure()
# plt.imshow(corr_norm)
# plt.show()

for i in range(num_images-1):
    corr = signal.correlate(images[i+1]-np.mean(images[i+1]), images[i]-np.mean(images[i]), mode='full',method='fft')/corr_norm
    corr = corr[200:-201,100:-101]
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    print(y,x)
    y_shifts.append(int(y-corr.shape[0]/2))
    windows.append(int(x-corr.shape[1]/2))
    # plt.figure()
    # plt.imshow(corr)
print(y_shifts)
print(windows)
# plt.show()

big_image = np.zeros((480+int(np.sum(y_shifts)),640+int(np.sum(windows))),dtype='float64')

total_y_shifts = 0
total_windows = 0

for i,image in enumerate(images[::-1]):
    if image is None:
        print(num_images-i,"image is not valid")
        continue

    big_image[total_y_shifts:480+total_y_shifts,total_windows:total_windows+640] += image/3
    total_y_shifts += y_shifts[num_images-i-1]
    total_windows += windows[num_images-i-1]
# cv2.imwrite("artifacts/stitched_image.png", big_image)
plt.imshow(big_image.astype('uint8'))
plt.show()