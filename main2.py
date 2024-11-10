
from PyQt5 import  QtWidgets, uic
import sys
import cv2
import numpy as np


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('untitled.ui', self)
        self.pushButton.clicked.connect(lambda: self.vnes())
        self.pushButton_2.clicked.connect(lambda: self.izvl())
        self.pushButton_3.clicked.connect(lambda: self.fail1())
        self.pushButton_4.clicked.connect(lambda: self.fail2())
        self.pushButton_5.clicked.connect(lambda: self.fail3())

    def vnes(self):
        global f1, f2, f3
        CHANNELS = 3
        IMAGE = f2[0]

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

        def format_bin(n):
            n = str('{0:08b}'.format(n))
            if n[0] == '-':
                n = '1' + '0' * (8 - len(n)) + n[1:]  # 1 at the beginning is equal -
                return n
            return '0' * (8 - len(n)) + n  # 0 is equal +

        a = []
        raw_cover_image = cv2.imread(IMAGE, flags=cv2.IMREAD_COLOR)
        height, width = raw_cover_image.shape[:2]
        while height % 8: height += 1  # Rows
        while width % 8: width += 1  # Cols
        valid_dim = (width, height)
        padded_image = cv2.resize(raw_cover_image, valid_dim)
        cover_image_f32 = np.float32(padded_image)
        cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

        for chan_index in range(CHANNELS):
            dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]
            for i in dct_blocks:
                a.append(i[0][0])

        print((list(map(lambda i: round(i), a)))[7450:7500])
        a = list(map(lambda i: round(i / 16), a))
        a = list(map(format_bin, a))

        CHANNELS = 3
        IMAGE = f1[0]
        STEGO_IMAGE = "stego_image.png"
        SECRET_MESSAGE = str(a)

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
                if i > 0 and not (i % int(Nc / 8)):
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
        while (height % 8): height += 1
        while (width % 8): width += 1
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
                    i = str('{0:08b}'.format(round(i[0][0] / 16)))
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

            stego_image[:, :, chan_index] = np.asarray(
                stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

        stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

        final_stego_image = np.uint8(
            np.clip(stego_image_BGR, 0, 255))  # установка макс и мин размера значений матрицы(clip)

        cv2.imwrite(STEGO_IMAGE, final_stego_image)
        self.label_3.setStyleSheet(f'border-image: url(stego_image.png) 0 0 0 0 ; background-repeat: no-repeat;')

    def izvl(self):
        global f3

        STEGO_IMAGE=f3[0]
        CHANNELS = 3
        ZERO_IMAGE = "White.jpg"
        PIXELS = 400
        EXTRACTED_IMAGE = "extracted_image.png"
        raw_cover_image = cv2.imread(ZERO_IMAGE, flags=cv2.IMREAD_COLOR)

        def split_image_into_8x8_blocks(image):  # разбиваем на блоки 8х8
            blocks = []
            for vert_slice in np.vsplit(image, int(image.shape[0] / 8)):
                for horiz_slice in np.hsplit(vert_slice, int(image.shape[1] / 8)):
                    blocks.append(horiz_slice)
            return blocks

        def stitch_8x8_blocks_back_together(Nc, block_segments):  # собираем блоки в строку
            image_rows = []
            temp = []
            for i in range(len(block_segments)):
                if i > 0 and not (i % int(Nc / 8)):
                    image_rows.append(temp)
                    temp = [block_segments[i]]
                else:
                    temp.append(block_segments[i])
            image_rows.append(temp)

            return np.block(image_rows)

        class YCC_Image(object):
            def __init__(self, cover_image):
                self.height, self.width = cover_image.shape[:2]
                self.channels = [
                    split_image_into_8x8_blocks(cover_image[:, :, 0]),
                    split_image_into_8x8_blocks(cover_image[:, :, 1]),
                    split_image_into_8x8_blocks(cover_image[:, :, 2]),
                ]

        matrix_2 = []
        i = 0
        x = 2500
        y = 5000
        number = ''
        recovered_matrix = []
        matrix_10 = []
        stego_image = cv2.imread(STEGO_IMAGE, flags=cv2.IMREAD_COLOR)
        stego_image_f32 = np.float32(stego_image)
        stego_image_YCC = YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

        # FORWARD DCT STAGE
        dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]

        for chan_index in range(CHANNELS):
            if chan_index == 0:
                for index in dct_blocks:
                    index = str('{0:08b}'.format(round(index[0][0] / 16)))
                    matrix_2.append(index)

        while len(matrix_10) != 7500:
            number = matrix_2[0][6] + matrix_2[0][7]
            number = number + matrix_2[1][6] + matrix_2[1][7]
            number = number + matrix_2[2][6] + matrix_2[2][7]
            number = number + matrix_2[3][6] + matrix_2[3][7]
            matrix_2.pop(3)
            matrix_2.pop(2)
            matrix_2.pop(1)
            matrix_2.pop(0)
            if number[0] == '1':
                number = '-' + number[1:]
            matrix_10.append(int(str(number), 2) * 16)
            number = ''

        raw_cover_image = cv2.imread(ZERO_IMAGE, flags=cv2.IMREAD_COLOR)
        height, width = raw_cover_image.shape[:2]
        while (height % 8): height += 1
        while (width % 8): width += 1
        valid_dim = (width, height)
        padded_image = cv2.resize(raw_cover_image, valid_dim)
        cover_image_f32 = np.float32(padded_image)
        cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))
        stego_image = np.empty_like(cover_image_f32)

        for chan_index in range(CHANNELS):

            dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

            if chan_index == 0:
                while i != 2500:
                    dct_blocks[i][0][0] = matrix_10[i]
                    i += 1
                i = 0
            if chan_index == 1:
                while i != 2500:
                    dct_blocks[i][0][0] = matrix_10[x + i]
                    i += 1
                i = 0
            if chan_index == 2:
                while i != 2500:
                    dct_blocks[i][0][0] = matrix_10[y + i]
                    i += 1

            idct_blocks = [cv2.idct(block) for block in dct_blocks]

            stego_image[:, :, chan_index] = np.asarray(
                stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

        stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)
        final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))
        cv2.imwrite(EXTRACTED_IMAGE, final_stego_image)

        self.label_6.setStyleSheet(f'border-image: url(extracted_image.png) 0 0 0 0 ; background-repeat: no-repeat;')

    def fail1(self):
        global f1
        f1 = QtWidgets.QFileDialog.getOpenFileName(parent=self)
        print(f1)
        self.label_2.setStyleSheet(f'border-image: url({f1[0]}) 0 0 0 0 ; background-repeat: no-repeat;')

    def fail2(self):
        global f2
        f2 = QtWidgets.QFileDialog.getOpenFileName(parent=self)
        self.label.setStyleSheet(f'border-image: url({f2[0]}) 0 0 0 0 ; background-repeat: no-repeat;')

    def fail3(self):
        global f3
        f3 = QtWidgets.QFileDialog.getOpenFileName(parent=self)
        self.label_5.setStyleSheet(f'border-image: url({f3[0]}) 0 0 0 0 ; background-repeat: no-repeat;')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())