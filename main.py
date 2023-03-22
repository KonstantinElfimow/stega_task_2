from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os


encoding: str = 'utf-8'


class Generator:
    def __init__(self, base: int, key: int):
        # основание (m > 0)
        self._base = base
        # подразумевает под собой n-19
        self._first_index = 39
        # подразумевает под собой n-58
        self._second_index = 0
        # инициализируем буфер
        self._buffer = [i for i in range(abs(key), abs(key) + 58)]

    # Находим следующий {Xi}
    def next(self) -> int:
        value = (self._buffer[self._second_index] + self._buffer[self._first_index]) % self._base
        del self._buffer[self._second_index]
        self._buffer.append(value)
        return value

    @property
    def base(self) -> int:
        return self._base

    @base.setter
    def base(self, value: int) -> None:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError('base > 0!')
            self._base = value


class KutterMethod:
    @staticmethod
    def _str_to_bits(message: str) -> list:
        return list(map(int, ''.join(['{0:08b}'.format(num) for num in list(message.encode(encoding))])))

    @staticmethod
    def _bits_to_str(bits: list) -> str:
        return ''.join(chr(int(''.join(map(str, bits[i: i + 8])), 2)) for i in range(0, len(bits), 8))

    def __init__(self, old_image_path: str, new_image_path: str) -> None:
        self._empty_image_path = old_image_path
        self._full_image_path = new_image_path
        self._occupancy = 0

    def encode(self, message: str, key_generator: int):
        img = Image.open(self._empty_image_path).convert('RGB')
        image = np.asarray(img, dtype='uint8')
        img.close()

        height, width, depth = image.shape[0], image.shape[1], image.shape[2]

        message_bits = KutterMethod._str_to_bits(message)
        if len(message_bits) > height * width:
            raise ValueError('Размер сообщения превышает размер контейнера!')
        # print(message_bits)
        # использованные пиксели
        keys = []
        generator = Generator(base=height * width, key=key_generator)
        for bit in message_bits:

            coordinate = generator.next()
            while coordinate in keys:
                coordinate = generator.next()
            keys.append(coordinate)

            i, j = divmod(coordinate, width)
            pixel = image[i, j]

            lam = 2
            L = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            if bit == 0:
                pixel[2] = np.uint8(min(255, pixel[2] + lam * L))
            elif bit == 1:
                pixel[2] = np.uint8(max(0, pixel[2] - lam * L))
        # print(sorted(keys))
        # print(len(keys) - len(set(keys)))

        self._occupancy = len(keys)
        Image.fromarray(image).save(self._full_image_path, 'PNG')

    def decode(self, key_generator: int) -> str:
        img = Image.open(self._full_image_path).convert('RGB')
        image = np.asarray(img, dtype='uint8')
        img.close()

        height, width, depth = image.shape[0], image.shape[1], image.shape[2]

        keys = []
        generator = Generator(base=height * width, key=key_generator)
        while len(keys) < self._occupancy:
            coordinate = generator.next()
            while coordinate in keys:
                coordinate = generator.next()
            keys.append(coordinate)
        # print(sorted(keys))
        # print(len(keys) - len(set(keys)))

        message_bits = []
        for coordinate in keys:
            i, j = divmod(coordinate, width)
            sigma = 1
            summary = 0
            for n in range(1, sigma + 1):
                if 0 <= i - n < height and 0 <= j < width:
                    summary += image[i - n, j, 2]
                if 0 <= i + n < height and 0 <= j < width:
                    summary += image[i + n, j, 2]
                if 0 <= i < height and 0 <= j - n < width:
                    summary += image[i, j - n, 2]
                if 0 <= i < height and 0 <= j + n < width:
                    summary += image[i, j + n, 2]

            if image[i, j, 2] > (summary / (4 * sigma)):
                message_bits.append(0)
            else:
                message_bits.append(1)
        # print(message_bits)
        message = KutterMethod._bits_to_str(message_bits)
        return message


def metrics(empty_image: str, full_image: str) -> None:
    img = Image.open(empty_image).convert('RGB')
    empty = np.asarray(img, dtype='uint8')
    img.close()

    img = Image.open(full_image).convert('RGB')
    full = np.asarray(img, dtype='uint8')
    img.close()

    max_d = np.max(np.abs(empty.astype(int) - full.astype(int)))
    print('Максимальное абсолютное отклонение:\n{}'.format(max_d))

    SNR_res = np.sum((empty * empty)) / np.sum((empty - full) * (empty - full))
    print('Отношение сигнал-шум:\n{}'.format(SNR_res))

    H, W = empty.shape[0], empty.shape[1]
    MSE = (1 / (W * H)) * (np.sum((empty - full) * (empty - full)))
    print('Среднее квадратичное отклонение:\n{}'.format(MSE))


def main():
    load_dotenv('.env')
    KEY: int = int(os.getenv('KEY'))

    old_image = 'input/old_image.png'
    new_image = 'output/new_image.png'
    message = 'Its the secret'

    t = KutterMethod(old_image, new_image)
    t.encode(message, KEY)
    m = t.decode(KEY)
    print(m)

    metrics(old_image, new_image)


if __name__ == '__main__':
    main()
