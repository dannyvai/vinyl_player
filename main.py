import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywav
import serial
from pathlib import Path
from collections import defaultdict

use_camera = False
if use_camera:
    cam = cv2.VideoCapture(0)
    videogram_serial = serial.Serial('COM6', baudrate=9600)

else:
    img = cv2.imread("audio_images/img.png")


record_circumference_mm = 957.557
RPM = 100.0/3.0
RPS = RPM/60.0
camera_mm = 1.5
audio_t_sec = camera_mm/(record_circumference_mm*RPS)
freq_hz = 1/(audio_t_sec/640)
print(audio_t_sec)
kernel = np.ones((5, 5), np.uint8)

signals = defaultdict(list)

for i in range(2038//2):
    if use_camera:
        ret,img = cam.read()
        if not ret:
            continue
        # cv2.imwrite("img.png",img)
        cv2.imshow("img",img)
        cv2.waitKey(300)
        cv2.imwrite('img_{:04d}.png'.format(i),img)
        videogram_serial.write(b'A')
    else:
        if i > 100:
            break
        image_path = Path("audio_images")/"img_{:04d}.png".format(i)
        print(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Can't load {image_path}")
            continue
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,bin_img = cv2.threshold(gray_img,25,255,cv2.THRESH_BINARY_INV)
        # bin_img = cv2.erode(bin_img, kernel)

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(bin_img,
                                                    4,
                                                    cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

        # Initialize a new image to
        # store all the output components
        output = np.zeros(gray_img.shape, dtype="uint8")
        lines_count = 0
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]

            if (area > 5000) :

                # Labels stores all the IDs of the components on the each pixel
                # It has the same dimension as the threshold
                # So we'll check the component
                # then convert it to 255 value to mark it white
                componentMask = (label_ids == i).astype("uint8") * 255
                data = (np.argmax(componentMask,axis=0)+480-np.argmax(componentMask[::-1],axis=0))/2
                # Creating the Final output mask
                output = cv2.bitwise_or(output, componentMask)

                # plt.plot(data)
                signals[lines_count].append(data)
                lines_count += 1

        # plt.imshow(label_ids, cmap='gray')
        # plt.show()

window = 180
for i in signals.keys():
    full_data = np.zeros((len(signals[i]),640+window*len(signals[i])))+np.nan
    for j,data in enumerate(signals[i]):
        full_data[j,window*j:window*j+640] = data
    avg_data = np.nanmean(full_data,axis=0)
    plt.plot(avg_data)

    # first parameter is the file name to write the wave data
    # second parameter is the number of channels, the value can be 1 (mono) or 2 (stereo)
    # third parameter is the sample rate, 8000 samples per second
    # fourth paramaer is the bits per sample, 8 bits per sample
    # fifth parameter is the audio format, 1 means PCM with no compression.
    wave_write = pywav.WavWrite(f"line_{i}.wav", 1, int(freq_hz), 8, 1)
    # raw_data is the byte array. Write can be done only once for now.
    # Incremental write will be implemented later
    wave_write.write(avg_data)
    # close the file stream and save the file
    wave_write.close()

plt.show()

