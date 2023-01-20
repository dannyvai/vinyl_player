import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import signal
from utils import (
    get_connected_components,
    load_flipped_image,
    get_image_shifts,
    combine_signals,
    convert_gray_binary_image,
    save_signal_as_wave
)
from scipy.signal import butter,filtfilt,resample


def track_image(right_image, selected_signal, dx, dy, debug=False):
    if debug:
        print("selected_signal size", selected_signal.shape)
    right_bin_image = convert_gray_binary_image(right_image)

    groove_radius_px = 20

    model = np.polyfit(np.arange(selected_signal.shape[0]), selected_signal, 1)
    print(model)

    y_start = model[0]*(selected_signal.shape[0]-(640-dx)) + model[1] - dy
    y_end = model[0]*right_image.shape[1] + y_start

    if debug:
        plt.figure()
        plt.imshow(right_bin_image)
        plt.plot(selected_signal[-(640-dx):]-dy,"b")
        plt.plot([0, right_image.shape[1]], [y_start, y_end],"r:")

    new_signal = np.zeros(640) + np.nan
    tops = np.zeros(640) + np.nan
    bottoms = np.zeros(640) + np.nan
    for i in range(right_image.shape[1]):
        y = int(model[0]*i + y_start)
        top_y = np.nan
        for j in range(y-groove_radius_px,y):
            if right_bin_image[j,i] == 0:
                top_y = j
            else:
                break

        tops[i] = top_y
        bottom_y = np.nan
        for j in range(y,y+groove_radius_px):
            if right_bin_image[j,i] == 0:
                bottom_y = j
                break
        bottoms[i] = bottom_y

        if np.isfinite(top_y) and np.isfinite(bottom_y):
            new_signal[i] = (top_y + bottom_y)/2.0
        else:
            new_signal[i] = y

    if debug:
        plt.plot(tops, label="top")
        plt.plot(bottoms, label="bottom")
        plt.plot(new_signal, label="signal")
        plt.legend()
    plt.show()

    return new_signal, tops, bottoms


def remove_slope(signal):
    model = np.polyfit(np.arange(signal.shape[0]), signal, 1)
    print(model)

    slope = model[0]*np.arange(signal.shape[0])+model[1]
    return signal - slope


def smooth_signal(signal):
    record_circumference_mm = 957.557
    RPM = 100.0 / 3.0
    RPS = RPM / 60.0
    camera_mm = 1.5
    audio_t_sec = camera_mm / (record_circumference_mm * RPS)
    freq_hz = 1 / (audio_t_sec / 640)
    order = 3  # sin wave can be approx represented as quadratic
    cutoff = 20000
    nyq = 0.5 * freq_hz
    normal_cutoff = cutoff / nyq

    ts = signal.shape[0]/freq_hz

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    smooth_signal = filtfilt(b, a, signal)
    resampled_smooth_signal = resample(smooth_signal, int(44100*ts))
    norm_signal = (resampled_smooth_signal/np.max(np.abs(resampled_smooth_signal)))*100

    return (norm_signal + 128)


selected_groove_idx = 4
num_images = 10
first_image_idx = 11
first_image = load_flipped_image("audio_images/img_{:04d}.png".format(first_image_idx))
signals = get_connected_components(first_image, debug=False)
selected_signal = signals[selected_groove_idx][0]
t_start_px = 0
dt_pixel = 640

# plt.figure()
# plt.imshow(first_image, cmap='gray')
# plt.plot(selected_signal, label="selected_signal")
# plt.legend()
# plt.show()

dxs = [0]
dys = [0]
tops_arr = []
bottoms_arr = []
left_image = first_image
for i in range(first_image_idx+1, first_image_idx + num_images):
    right_image = load_flipped_image("audio_images/img_{:04d}.png".format(i))
    print("audio_images/img_{:04d}.png".format(i))
    dx, dy = get_image_shifts(left_image, right_image, debug=False)
    new_signal, tops, bottoms = track_image(right_image, selected_signal, dx, dy, debug=False)
    selected_signal = combine_signals(selected_signal, new_signal, t_start_px, dx, dy)
    left_image = right_image
    t_start_px += dx
    dxs.append(dx)
    dys.append(dy)
    tops_arr.append(tops)
    bottoms_arr.append(bottoms)

    # plt.figure()
    # plt.imshow(right_image, cmap='gray')
    # plt.plot(tops, label="tops")
    # plt.plot(bottoms, label="bottom")
    # plt.plot(new_signal, label="new_signal")
    # plt.plot(selected_signal[-640:], label="selected_signal")
    # plt.legend()
    # plt.show()


"""
Stitch the images
"""
big_image = np.zeros((480+int(np.sum(np.abs(dys))),640+int(np.sum(dxs))), dtype='float64')
total_y_shifts = np.sum(np.abs(dys))
total_windows = 0
for i in range(first_image_idx, first_image_idx + num_images):
    print(i)
    print(dxs[i-first_image_idx], dys[i-first_image_idx])
    print(total_windows,total_y_shifts)
    image = load_flipped_image("audio_images/img_{:04d}.png".format(i))

    if image is None:
        print(num_images-i, "image is not valid")
        continue
    total_y_shifts += dys[i-first_image_idx]
    total_windows += dxs[i-first_image_idx]
    big_image[total_y_shifts:480+total_y_shifts,total_windows:total_windows+640] += image/3

"""
Stitch the tops and bottoms lines
"""
total_y_shifts = np.sum(np.abs(dys))
total_windows = 0
tops_ndarray = np.zeros((num_images-1, big_image.shape[1])) + np.nan
bottoms_ndarray = np.zeros((num_images-1, big_image.shape[1])) + np.nan
for i in range(first_image_idx+1, first_image_idx + num_images):
    total_y_shifts += dys[i-first_image_idx]
    total_windows += dxs[i-first_image_idx]
    tops_ndarray[i-first_image_idx-1, total_windows:total_windows+640] = tops_arr[i-first_image_idx-1] + total_y_shifts
    bottoms_ndarray[i-first_image_idx-1, total_windows:total_windows+640] = bottoms_arr[i-first_image_idx-1] + total_y_shifts

tops_ndarray = np.nanmean(tops_ndarray, axis=0)
bottoms_ndarray = np.nanmean(bottoms_ndarray, axis=0)

"""
Plot the stitched images with the top and bottom selected groove lines
"""
plt.figure(figsize=(8, 4))
plt.imshow(big_image)
plt.plot(tops_ndarray, label="top")
plt.plot(bottoms_ndarray, label="bottom")
plt.plot(selected_signal, label="signal")
plt.legend()
plt.show()

# cv2.imwrite("artifacts/stitched_image.png", big_image)
#
# selected_signal = remove_slope(selected_signal)
# fixed_signal = smooth_signal(selected_signal)
# save_signal_as_wave(fixed_signal[::-1].astype(np.uint8), f"line_{selected_groove_idx}.wav")
#
# plt.figure()
# plt.plot(selected_signal)
# plt.plot(fixed_signal)
#
# plt.show()
