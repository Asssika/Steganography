import cv2
import numpy as np
from Prepare_image3 import a

CHANNELS = 3
IMAGE = "frameFHD.jpg"  # Choose your cover image (PNG)
STEGO_IMAGE = "stego_image.png"  # Choose name for your saving stego file
SECRET_MESSAGE = str(a)
# ============================================================================= #
# ============================================================================= #
# ============================= IMAGE PREPARATION ============================= #
# ============================================================================= #
# ============================================================================= #

# Numpy Macros
HORIZ_AXIS = 1
VERT_AXIS = 0


class YCC_Image(object):
    def __init__(self, cover_image):
        self.height, self.width = cover_image.shape[:2]
        self.channels = [
                         split_image_into_8x8_blocks(cover_image[:, :, 0]),
                         split_image_into_8x8_blocks(cover_image[:, :, 1]),
                         split_image_into_8x8_blocks(cover_image[:, :, 2]),
                        ]


def stitch_8x8_blocks_back_together(Nc, block_segments):  # собираем блоки в строку
    image_rows = []
    temp = []
    for i in range(len(block_segments)):
        if i > 0 and not(i % int(Nc / 8)):
            image_rows.append(temp)
            temp = [block_segments[i]]
        else:
            temp.append(block_segments[i])
    image_rows.append(temp)

    return np.block(image_rows)


def split_image_into_8x8_blocks(image):  # разбиваем на блоки 8х8
    blocks = []
    for vert_slice in np.vsplit(image, int(image.shape[0] / 8)):
        for horiz_slice in np.hsplit(vert_slice, int(image.shape[1] / 8)):
            blocks.append(horiz_slice)
    return blocks

# ============================================================================= #
# ============================================================================= #
# ========================== START STEGO EMBEDDING ============================ #
# ============================================================================= #
# ============================================================================= #


raw_cover_image = cv2.imread(IMAGE, flags=cv2.IMREAD_COLOR)
height, width = raw_cover_image.shape[:2]
while(height % 8): height += 1
while(width % 8): width += 1
valid_dim = (width, height)
padded_image = cv2.resize(raw_cover_image, valid_dim)
cover_image_f32 = np.float32(padded_image)
cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))
stego_image = np.empty_like(cover_image_f32)

k = []
k2 = []
length = 0
count = 0
for chan_index in range(CHANNELS):

    dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

    if chan_index == 0:
        for i in dct_blocks:
            i = str('{0:08b}'.format(round(i[0][0]/16)))
            k.append(i)
    while len(a) != 0:
        k[length + 0] = k[length + 0][:6] + a[0][0:2]
        k[length + 1] = k[length + 1][:6] + a[0][2:4]
        k[length + 2] = k[length + 2][:6] + a[0][4:6]
        k[length + 3] = k[length + 3][:6] + a[0][6:8]
        # k[length + 0] = k[length + 0][:7] + a[0][0]
        # k[length + 1] = k[length + 1][:7] + a[0][1]
        # k[length + 2] = k[length + 2][:7] + a[0][2]
        # k[length + 3] = k[length + 3][:7] + a[0][3]
        # k[length + 4] = k[length + 4][:7] + a[0][4]
        # k[length + 5] = k[length + 5][:7] + a[0][5]
        # k[length + 6] = k[length + 6][:7] + a[0][6]
        # k[length + 7] = k[length + 7][:7] + a[0][7]
        a.pop(0)
        length = length + 4

    if chan_index == 0:
        for i in k:
            i = int(str(i), 2) * 16
            k2.append(i)
        while count != len(k2):
            dct_blocks[count][0][0] = k2[count]
            count += 1

    idct_blocks = [cv2.idct(block) for block in dct_blocks]

    stego_image[:, :, chan_index] = np.asarray(stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))  # установка макс и мин размера значений матрицы(clip)

cv2.imwrite(STEGO_IMAGE, final_stego_image)

