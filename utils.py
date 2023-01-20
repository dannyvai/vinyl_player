import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import signal
import pywav


def convert_gray_binary_image(gray_img):
    kernel = np.ones((5, 5), np.uint8)
    ret, bin_img = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY_INV)
    bin_img = cv2.erode(bin_img, kernel)
    return bin_img

def get_connected_components(img: np.ndarray, debug:bool=False):
    signals = defaultdict(list)

    bin_img = convert_gray_binary_image(img)
    if debug:
        plt.imshow(bin_img)

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(bin_img,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to
    # store all the output components
    output = np.zeros(img.shape, dtype="uint8")
    lines_count = 0
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]

        if (area > 5000):
            # Labels stores all the IDs of the components on the each pixel
            # It has the same dimension as the threshold
            # So we'll check the component
            # then convert it to 255 value to mark it white
            componentMask = (label_ids == i).astype("uint8") * 255
            data = (np.argmax(componentMask, axis=0) + img.shape[0] - np.argmax(componentMask[::-1], axis=0)) / 2
            # Creating the Final output mask
            output = cv2.bitwise_or(output, componentMask)
            if debug:
                plt.plot(data)
            signals[lines_count].append(data)
            lines_count += 1
    if debug:
        plt.show()

    return signals


def load_flipped_image(image_path):
    img = np.fliplr(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2GRAY))
    return img


corr_norm = signal.correlate(np.ones((480,640)), np.ones((480,640)), mode='full',method='fft')


def get_image_shifts(img2, img1, debug=False):
    corr = signal.correlate(img2-np.mean(img2), img1-np.mean(img1), mode='full', method='fft')/corr_norm
    corr = corr[470:-471,50:-51]
    corr[:,:630] = 0
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    dy = int(y-corr.shape[0]/2)
    dx = int(x-corr.shape[1]/2)
    print("get_image_shifts dx",dx)
    print("get_image_shifts dy",dy)
    if debug:
        plt.figure()
        plt.title("img1")
        plt.imshow(img1)

        plt.figure()
        plt.title("img2")
        plt.imshow(img2)

        plt.figure()
        plt.title("correlation")
        plt.imshow(corr)

    return dx, dy


def combine_signals(selected_signal, new_signal, t_start_px, dx, dy):
    combined_signal_length = selected_signal.shape[0] + dx
    print("combine_signals")
    print(selected_signal.shape,new_signal.shape,dx,combined_signal_length)
    combined_signal = np.zeros((2, combined_signal_length)) + np.nan
    combined_signal[0, :selected_signal.shape[0]] = selected_signal - dy
    combined_signal[1, -new_signal.shape[0]:] = new_signal
    return np.nanmean(combined_signal, axis=0)

def save_signal_as_wave(signal,filename):

    # first parameter is the file name to write the wave data
    # second parameter is the number of channels, the value can be 1 (mono) or 2 (stereo)
    # third parameter is the sample rate, 8000 samples per second
    # fourth paramaer is the bits per sample, 8 bits per sample
    # fifth parameter is the audio format, 1 means PCM with no compression.
    wave_write = pywav.WavWrite(filename, 1, int(44100), 8, 1)
    # raw_data is the byte array. Write can be done only once for now.
    # Incremental write will be implemented later
    wave_write.write(signal)
    # close the file stream and save the file
    wave_write.close()