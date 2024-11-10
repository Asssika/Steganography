import cv2
import struct
import bitstring  # библиотека для работы с битами
import numpy as np
import zigzag as zz
from PIL import Image
from Prepare_image import a

# ============================================================================= #
# ============================================================================= #
# =============================== USING DATA ================================== #
# ============================================================================= #
# ============================================================================= #

# https://scask.ru/a_lect_cod.php?id=16
# https://github.com/MasonEdgar/DCT-Image-Steganography

# ============================================================================= #
# ============================================================================= #
# ============================== INITIAL DATA ================================= #
# ============================================================================= #
# ============================================================================= #

CHANNELS = 3
IMAGE = "Brunette.jpg"  # Choose your cover image (PNG)
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

# Стандартная таблица квантования для светимости определенная форматом JPEG
JPEG_STD_LUM_QUANT_TABLE = np.asarray([
                                        [16, 11, 10, 16,  24, 40,   51,  61],
                                        [12, 12, 14, 19,  26, 58,   60,  55],
                                        [14, 13, 16, 24,  40, 57,   69,  56],
                                        [14, 17, 22, 29,  51, 87,   80,  62],
                                        [18, 22, 37, 56,  68, 109, 103,  77],
                                        [24, 36, 55, 64,  81, 104, 113,  92],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103,  99]
                                      ],
                                      dtype=np.float32)
# Image container class
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


def embed_encoded_data_into_DCT(encoded_bits, dct_blocks):
    data_complete = False; encoded_bits.pos = 0  # Pos отыскивает в строке первое вхождение подстроки и возвращает в качестве значения номер элемента, с которого начинается вхождение
    encoded_data_len = bitstring.pack('uint:32', len(encoded_bits))
    converted_blocks = []
    for current_dct_block in dct_blocks:
        for i in range(1, len(current_dct_block)):
            curr_coeff = np.int32(current_dct_block[i])
            if (curr_coeff > 1):
                curr_coeff = np.uint8(current_dct_block[i])
                if encoded_bits.pos == (len(encoded_bits) - 1):
                    data_complete = True; break
                pack_coeff = bitstring.pack('uint:8', curr_coeff)
                if encoded_data_len.pos <= len(encoded_data_len) - 1:
                    pack_coeff[-1] = encoded_data_len.read(1)
                else: pack_coeff[-1] = encoded_bits.read(1)
                # Replace converted coefficient
                current_dct_block[i] = np.float32(pack_coeff.read('uint:8'))
        converted_blocks.append(current_dct_block)


    if not(data_complete): raise ValueError("Data didn't fully embed into cover image!")

    return converted_blocks


raw_cover_image = cv2.imread(IMAGE, flags=cv2.IMREAD_COLOR)
height, width = raw_cover_image.shape[:2]
# Force Image Dimensions to be 8x8 compliant
while(height % 8): height += 1 # Rows
while(width % 8): width += 1 # Cols
valid_dim = (width, height)
padded_image = cv2.resize(raw_cover_image, valid_dim)
cover_image_f32 = np.float32(padded_image)
cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb)) # cvtColor она принимает всего два аргумента и отдает итоговое изображение. Первый аргумент – исходное изображение, второй – направление трансформации

# Placeholder for holding stego image data
stego_image = np.empty_like(cover_image_f32) # empty_like () используется для создания нового массива с той же формой и типом, что и у данного массива

for chan_index in range(CHANNELS):
    # FORWARD DCT STAGE
    dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

    # QUANTIZATION STAGE
    dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks] # divide() выполняет поэлементное истинное деление массивов x1 и x2

    # Sort DCT coefficients by frequency
    sorted_coefficients = [zz.zigzag(block) for block in dct_quants] # зигзаг преобразование всех блоков 8х8

    # Embed data in Luminance layer
    if (chan_index == 0):
        # DATA INSERTION STAGE
        secret_data = ""
        for char in SECRET_MESSAGE.encode('ascii'): secret_data += bitstring.pack('uint:8', char) # преобразвание сообщения в двоичный код
        embedded_dct_blocks = embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
        desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks] # reverse zigzag transform
    else:
        # Reorder coefficients to how they originally were
        desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in sorted_coefficients]

    # DEQUANTIZATION STAGE
    dct_dequants = [np.multiply(data, JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]

    # Inverse DCT Stage
    idct_blocks = [cv2.idct(block) for block in dct_dequants]

    # Rebuild full image channel
    stego_image[:, :, chan_index] = np.asarray(stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

# Convert back to RGB (BGR) Colorspace
stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

# Clamp Pixel Values to [0 - 255]
final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255)) # установка макс и мин размера значений матрицы(clip)

# Write stego image
cv2.imwrite(STEGO_IMAGE, final_stego_image)

# ============================================================================= #
# ============================================================================= #
# ======================== EXTRACTION STEGO IMAGE ============================= #
# ============================================================================= #
# ============================================================================= #


def extract_encoded_data_from_DCT(dct_blocks):
    extracted_data = ""
    for current_dct_block in dct_blocks:
        for i in range(1, len(current_dct_block)):
            curr_coeff = np.int32(current_dct_block[i])
            if (curr_coeff > 1):
                extracted_data += bitstring.pack('uint:1', np.uint8(current_dct_block[i]) & 0x01)
    return extracted_data


stego_image = cv2.imread(STEGO_IMAGE, flags=cv2.IMREAD_COLOR)
stego_image_f32 = np.float32(stego_image)
stego_image_YCC = YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

# FORWARD DCT STAGE
dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

# QUANTIZATION STAGE
dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

# Sort DCT coefficients by frequency
sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

# DATA EXTRACTION STAGE
recovered_data = extract_encoded_data_from_DCT(sorted_coefficients)

# Determine length of secret message
data_len = int(recovered_data.read('uint:32') / 8)

# Extract secret message from DCT coefficients
extracted_data = bytes()
for _ in range(data_len): extracted_data += struct.pack('>B', recovered_data.read('uint:8'))

# Print secret message back to the user
# print((extracted_data.decode('ascii')))
extracted_data = extracted_data.decode('ascii')
extracted_data = extracted_data[1:-1]
extracted_data = extracted_data.replace(',','')
extracted_data = list(map(int, extracted_data.split(' ')))
print(extracted_data)