import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywav
from pathlib import Path
from collections import defaultdict
from utils import get_connected_components

record_circumference_mm = 957.557
RPM = 100.0/3.0
RPS = RPM/60.0
camera_mm = 1.5
audio_t_sec = camera_mm/(record_circumference_mm*RPS)
freq_hz = 1/(audio_t_sec/640)
print(audio_t_sec)
kernel = np.ones((5, 5), np.uint8)



image_path = Path("artifacts")/"stitched_image.png"
img = cv2.imread(str(image_path))

signals = get_connected_components(img, True)

for i in signals.keys():
    full_data = signals[i][0]
    # plt.plot(full_data)



# plt.show()

