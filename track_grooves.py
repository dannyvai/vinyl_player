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
from scipy.signal import butter, filtfilt, resample
from typing import Tuple

def track_image(
        right_image_: np.ndarray,
        selected_signal_: np.ndarray,
        dx_: float,
        dy_: float,
        debug:bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if debug:
        print("selected_signal size", selected_signal_.shape)
    right_bin_image = convert_gray_binary_image(right_image_)

    groove_radius_px = 20

    model = np.polyfit(np.arange(selected_signal_.shape[0]), selected_signal_, 1)
    if debug:
        print(model)

    y_start = model[0] * (selected_signal_.shape[0] - (right_image_.shape[1] - dx_)) + model[1] - dy_
    y_end = model[0] * right_image_.shape[1] + y_start

    if debug:
        plt.figure()
        plt.imshow(right_bin_image)
        plt.plot(selected_signal_[-(640 - dx_):] - dy_, "b")
        plt.plot([0, right_image_.shape[1]], [y_start, y_end], "r:")

    new_signal_ = np.zeros(640) + np.nan
    tops_ = np.zeros(640) + np.nan
    bottoms_ = np.zeros(640) + np.nan
    for i_ in range(right_image_.shape[1]):
        y = int(model[0] * i_ + y_start)
        top_y = np.nan
        for j in range(y-groove_radius_px,y):
            if right_bin_image[j, i_] == 0:
                top_y = j
            else:
                break

        tops_[i_] = top_y
        bottom_y = np.nan
        for j in range(y,y+groove_radius_px):
            if right_bin_image[j, i_] == 0:
                bottom_y = j
                break
        bottoms_[i_] = bottom_y

        if np.isfinite(top_y) and np.isfinite(bottom_y):
            new_signal_[i_] = (top_y + bottom_y) / 2.0
        else:
            new_signal_[i_] = y

    if debug:
        plt.plot(tops_, label="top")
        plt.plot(bottoms_, label="bottom")
        plt.plot(new_signal_, label="signal")
        plt.legend()
    plt.show()

    return new_signal_, tops_, bottoms_


def remove_slope(signal_: np.ndarray) -> np.ndarray:
    model = np.polyfit(np.arange(signal_.shape[0]), signal_, 1)
    print(model)

    slope = model[0] * np.arange(signal_.shape[0]) + model[1]
    return signal_ - slope


def smooth_signal(signal_: np.ndarray) -> np.ndarray:
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

    ts = signal_.shape[0] / freq_hz

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    smooth_signal_ = filtfilt(b, a, signal_)
    resampled_smooth_signal = resample(smooth_signal_, int(44100 * ts))
    norm_signal = (resampled_smooth_signal/np.max(np.abs(resampled_smooth_signal)))*100

    return norm_signal + 128


selected_groove_idx = 4
num_images = 589
first_image_idx = 11
first_image = load_flipped_image("audio_images/img_{:04d}.png".format(first_image_idx))
signals = get_connected_components(first_image, debug=False)
selected_signal = signals[selected_groove_idx][0]
t_start_px = 0
dt_pixel = 640
generate_debug_plots = False
debug = False

if debug:
    plt.figure()
    plt.imshow(first_image, cmap='gray')
    plt.plot(selected_signal, label="selected_signal")
    plt.legend()
    plt.show()


if generate_debug_plots:
    dxs = [0]
    dys = [0]
    tops_arr = []
    bottoms_arr = []
    signals = [selected_signal]
left_image = first_image
for i in range(first_image_idx+1, first_image_idx + num_images):
    right_image = load_flipped_image("audio_images/img_{:04d}.png".format(i))
    print("audio_images/img_{:04d}.png".format(i))
    dx, dy = get_image_shifts(left_image, right_image, debug=False)
    new_signal, tops, bottoms = track_image(right_image, selected_signal, dx, dy, debug=False)
    selected_signal = combine_signals(selected_signal, new_signal, t_start_px, dx, dy)
    left_image = right_image
    t_start_px += dx
    if generate_debug_plots:
        dxs.append(dx)
        dys.append(dy)
        tops_arr.append(tops)
        bottoms_arr.append(bottoms)
        #signals.append((tops+bottoms)/2)
        signals.append(new_signal)

    if debug:
        plt.figure()
        plt.imshow(right_image, cmap='gray')
        plt.plot(tops, label="tops")
        plt.plot(bottoms, label="bottom")
        plt.plot(new_signal, label="new_signal")
        plt.plot(selected_signal[-640:], label="selected_signal")
        plt.legend()
        plt.show()

if generate_debug_plots:
    """
    Stitch the images
    """
    big_image = np.zeros((480+int(np.sum(np.abs(dys))),640+int(np.sum(dxs))), dtype='float64')
    total_y_shifts = np.sum(np.abs(dys))
    total_windows = 0
    stitched_signal = np.zeros((num_images, 640+int(np.sum(dxs))), dtype='float64') + np.nan
    for i in range(first_image_idx, first_image_idx + num_images):
        print(i, dxs[i-first_image_idx], dys[i-first_image_idx],total_windows, total_y_shifts)
        image = load_flipped_image("audio_images/img_{:04d}.png".format(i))

        if image is None:
            print(num_images-i, "image is not valid")
            continue
        total_y_shifts += dys[i-first_image_idx]
        total_windows += dxs[i-first_image_idx]
        print(total_windows+640)
        big_image[total_y_shifts:480+total_y_shifts, total_windows:total_windows+640] += image/3

        stitched_signal[i-first_image_idx, total_windows:total_windows+640] = signals[i-first_image_idx] + total_y_shifts

    stitched_signal = np.nanmean(stitched_signal, axis=0)
    cv2.imwrite(f"artifacts/stitched_image_{first_image_idx}_{first_image_idx + num_images}.png", big_image)

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
    total_y_shifts = np.sum(np.abs(dys))
    fig = plt.figure(figsize=(150, 30), dpi=100)
    plt.imshow(big_image)
    plt.plot(tops_ndarray, label="top")
    plt.plot(bottoms_ndarray, label="bottom")
    plt.plot(stitched_signal, label="stitched_signal")
    plt.plot(selected_signal, label="selected_signal")
    plt.legend()
    plt.savefig(f"artifacts/groove_tracking_{first_image_idx}_{first_image_idx + num_images}.png")
    plt.close(fig)

    print("stitched_signal.shape",stitched_signal.shape)
    print("selected_signal.shape", selected_signal.shape)

    stitched_signal = remove_slope(stitched_signal)
    fixed_signal = smooth_signal(stitched_signal)
    save_signal_as_wave(fixed_signal[::-1].astype(np.uint8), f"line_{selected_groove_idx}_stitched.wav")

    plt.figure()
    plt.plot(stitched_signal)
    plt.plot(fixed_signal)

    plt.show()

selected_signal = remove_slope(selected_signal)
fixed_signal = smooth_signal(selected_signal)
save_signal_as_wave(fixed_signal[::-1].astype(np.uint8), f"line_{selected_groove_idx}.wav")

plt.figure()
plt.plot(selected_signal)
plt.plot(fixed_signal)

plt.show()
