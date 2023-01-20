import cv2
import serial
import os

cam = cv2.VideoCapture(0)
videogram_serial = serial.Serial('COM6', baudrate=9600)

arduino_step_size = 2
vinyl_images_basedir = "vinyl_images"
os.makedirs(vinyl_images_basedir,exist_ok=True)

for i in range(3000//arduino_step_size):
    ret, img = cam.read()
    if not ret:
        continue
    if i % 50 == 0:
        print(i)
    cv2.imshow("img", img)
    cv2.waitKey(300)
    cv2.imwrite(os.path.join(vinyl_images_basedir,'img_{:04d}.png'.format(i)),img)
    videogram_serial.write(b'A')

