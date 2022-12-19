import os
import random
import signal
import sys
import time
from pprint import pprint
from glob import glob
import spidev
import numpy as np
import ws2812
import cv2

# width and height as total number of pixels per side from combined panels
WIDTH = 16
HEIGHT = 16
# segment order is determined by how the individual segments are wired together
# currently only works with 4 panels arranged as 2 x 2
# uses human indices starting at 1, not array indices!
SEGMENT_ORDER = [1, 2, 3, 4]
# how many times we increase saturation by. Power of 2 will make calculation more efficient
SATURATION_FACTOR = 1.5
SATURATION_OFFSET = 0
# how many times we decrease intensity by. Power of 2 will make calculation more efficient
INTENSITY_FACTOR = 1.5
PIXELS = WIDTH * HEIGHT
SPI_DEVICE = 0

spi = spidev.SpiDev()
spi.open(SPI_DEVICE, 0)


def split_into_quarters(array):
    """
    Split a matrix into sub-matrices.
    from https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy
    """

    upper_half = np.hsplit(np.vsplit(array, 2)[0], 2)
    lower_half = np.hsplit(np.vsplit(array, 2)[1], 2)

    upper_left = upper_half[0]
    upper_right = upper_half[1]
    lower_left = lower_half[0]
    lower_right = lower_half[1]

    return [upper_left, upper_right, lower_left, lower_right]


def convert_matrix_to_bit_array(data: np.ndarray):
    segments = split_into_quarters(data)
    segment_list = [np.reshape(segments[index - 1], (WIDTH * HEIGHT // 4, 3)) for index in SEGMENT_ORDER]

    return np.concatenate(segment_list)


def signal_handler(sig, frame):
    print('Terminating on CTRL+C')
    data = np.zeros((PIXELS, 3), dtype=np.uint8)
    ws2812.write2812(spi, data)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def process_file(image_file):
    cache_filename = image_file + '.npy'
    if os.path.isfile(cache_filename):
        # use cached image for speed
        flattened_matrix = np.load(cache_filename)
    else:
        # no cache, so perform full preprocessing
        img = cv2.imread(image_file)

        # check if image needs to be resized to WIDTH x HEIGHT
        w, h, d = img.shape
        if w != WIDTH or h != HEIGHT:
            img = cv2.resize(img, dsize=(WIDTH, HEIGHT))

        if SATURATION_FACTOR > 1 or INTENSITY_FACTOR > 1 or SATURATION_OFFSET != 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            # raise saturation value
            if SATURATION_FACTOR > 1:
                s = np.multiply(s.astype(np.int16), SATURATION_FACTOR)
            if SATURATION_OFFSET != 0:
                s = np.add(s.astype(np.int16), SATURATION_OFFSET)
            s = np.clip(s, 0, 255).astype(np.uint8)
            # decrease intensity
            if INTENSITY_FACTOR > 1:
                v = np.divide(v.astype(np.int16), INTENSITY_FACTOR).astype(np.uint8)
            hsv_img = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # ws2812b takes colour channels as G R B,
        # no native conversion, so we merge them manually
        blue, green, red = cv2.split(img)
        img = cv2.merge([green, red, blue])
        #np.save(cache_filename, img)

        flattened_matrix = convert_matrix_to_bit_array(img)
        np.save(cache_filename, flattened_matrix)

    target_matrix = flattened_matrix
    ws2812.write2812(spi, target_matrix)


if __name__ == '__main__':
    # delete old caches to allow regenerating altered saturation and intensity
    cached_arrays = glob('images/*.npy')
    for cached in cached_arrays:
        os.remove(cached)

    files = glob('images/*.png')
    files.extend(glob('images/*.jpg'))
    print("Press CTRL+C to shut down.")

    while True:
        random.shuffle(files)
        for file in files:
            process_file(file)
            time.sleep(3)
