import cv2
import numpy as np
from Prepare_image import cover_image_f32
from Stego_fully import extracted_data

CHANNELS = 3
IMAGE = "Lena64.jpg"
STEGO_IMAGE = "stego_image.png"
PIXELS = 64
STEGO_IMAGE2 = "new_stego.png"
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


stego_image = np.empty_like(cover_image_f32)
stego_image = stego_image * 0
print(stego_image)
stego_image_YCC = YCC_Image(cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb))
stego = []
for chan_index in range(CHANNELS):
    temp = np.array([[], [], [], [], [], [], [], []])
    for block1 in stego_image_YCC.channels[chan_index]:
        block1[0][0] = extracted_data[0]
        extracted_data.pop(0)
        idct_blocks = cv2.idct(block1)
        temp = np.hstack([temp, idct_blocks])
    temp_1 = np.zeros(PIXELS)
    for line in np.hsplit(temp, PIXELS/8):  # vsplit
        temp_1 = np.vstack([temp_1, line])
    temp_1 = np.delete(temp_1, 0, axis=0)
    stego.append(temp_1)
image = []
for y in range(PIXELS):
    for x in range(PIXELS):
        hui = []
        hui.append(stego[0][y][x])
        hui.append(stego[1][y][x])
        hui.append(stego[2][y][x])
        image.append(hui)
image = np.array(image)
image = np.reshape(image, (PIXELS,PIXELS,3))
image = np.float32(image)
stego_image_BGR = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))
cv2.imwrite(STEGO_IMAGE2, final_stego_image)