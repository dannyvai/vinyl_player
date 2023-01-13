import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import signal

num_images = 400
images = []
for i in range(11,num_images+11):
    images.append(cv2.cvtColor(cv2.imread("audio_images/img_{:04d}.png".format(i)),cv2.COLOR_BGR2GRAY).astype('float32'))


windows = [0]
y_shifts = [0]

corr_norm = signal.correlate(np.ones((480,640)), np.ones((480,640)), mode='full',method='fft')
# plt.figure()
# plt.imshow(corr_norm)
# plt.show()

base_image = 0
next_image = 1
images_to_pop = []
for i in range(num_images-1):
    corr = signal.correlate(images[next_image]-np.mean(images[next_image]), images[base_image]-np.mean(images[base_image]), mode='full',method='fft')/corr_norm
    corr = corr[420:-421,100:-101]
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    print(y,x)
    y_shift = int(y-corr.shape[0]/2)
    window = int(x-corr.shape[1]/2)


    if window < 0:
        next_image += 1
        images_to_pop.append(i)
        y_shifts.append(0)
        windows.append(0)
        continue
    y_shifts.append(y_shift)
    windows.append(window)
    base_image = next_image
    next_image += 1

    # plt.figure()
    # plt.imshow(corr)

for idx_to_pop in images_to_pop:
    images[idx_to_pop] = None
    y_shifts[idx_to_pop] = 0
    windows[idx_to_pop] = 0

print("y_shifts",y_shifts)
print("windows",windows)
print("images_to_pop",images_to_pop)
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
cv2.imwrite("artifacts/stitched_image.png", big_image)
# plt.imshow(big_image.astype('uint8'))
# plt.show()