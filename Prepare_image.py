import cv2
import numpy as np


CHANNELS = 3
IMAGE = "Lena64.jpg"
STEGO_IMAGE = "stego_image.png"
raw_cover_image = cv2.imread(IMAGE, flags=cv2.IMREAD_COLOR)


def split_image_into_8x8_blocks(image):  # разбиваем на блоки 8х8
    blocks = []
    for vert_slice in np.vsplit(image, int(image.shape[0] / 8)):
        for horiz_slice in np.hsplit(vert_slice, int(image.shape[1] / 8)):
            blocks.append(horiz_slice)
    return blocks


class YCC_Image(object):
    def __init__(self, cover_image):
        self.height, self.width = cover_image.shape[:2]
        self.channels = [
                         split_image_into_8x8_blocks(cover_image[:, :, 0]),
                         split_image_into_8x8_blocks(cover_image[:, :, 1]),
                         split_image_into_8x8_blocks(cover_image[:, :, 2]),
                        ]


raw_cover_image = cv2.imread(IMAGE, flags=cv2.IMREAD_COLOR)
height, width = raw_cover_image.shape[:2]
while(height % 8): height += 1 # Rows
while(width % 8): width += 1 # Cols
valid_dim = (width, height)
padded_image = cv2.resize(raw_cover_image, valid_dim)
cover_image_f32 = np.float32(padded_image)
cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

a = []
for chan_index in range(CHANNELS):
    dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]
    for i in dct_blocks:
        a.append(i[0][0])

a = list(map(lambda i: round(i), a))
print(a)
