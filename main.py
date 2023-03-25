from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

encoding: str = 'utf-8'


class Generator:
    def __init__(self, base: int, key: int):
        # основание (m > 0)
        self.base = base
        # подразумевает под собой n-19
        self._first_index = 39
        # подразумевает под собой n-58
        self._second_index = 0
        # инициализируем буфер
        self._buffer = [i for i in range(abs(key), abs(key) + 58)]

    # Находим следующий {Xi}
    def next(self) -> int:
        value = (self._buffer[self._second_index] + self._buffer[self._first_index]) % self.base
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
    def __init__(self, old_image_path: str, new_image_path: str):
        self._empty_image_path: str = old_image_path
        self._full_image_path: str = new_image_path
        self._lam: float = 1
        self._sigma: int = 1
        self._occupancy: int = 0

    @staticmethod
    def str_to_bits(message: str) -> list:
        result = []
        for num in list(message.encode(encoding=encoding)):
            result.extend([(num >> x) & 1 for x in range(7, -1, -1)])
        return result

    @staticmethod
    def bits_to_str(bits: list) -> str:
        chars = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
        return ''.join(chars)

    def embed(self, message: str, key_generator: int):
        img = Image.open(self._empty_image_path).convert('RGB')
        image = np.asarray(img, dtype='uint8')
        img.close()

        height, width, depth = image.shape[0], image.shape[1], image.shape[2]

        message_bits = KutterMethod.str_to_bits(message)
        if len(message_bits) > height * width:
            raise ValueError('Размер сообщения превышает размер контейнера!')
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

            lam = self.lam
            L = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            if bit == 1:
                pixel[2] = np.uint8(min(255, pixel[2] + lam * L))
            elif bit == 0:
                pixel[2] = np.uint8(max(0, pixel[2] - lam * L))

        self._occupancy = len(message_bits)
        Image.fromarray(image).save(self._full_image_path, 'PNG')

    def recover(self, key_generator: int) -> str:
        img = Image.open(self._full_image_path).convert('RGB')
        image = np.asarray(img, dtype='uint8')
        img.close()

        height, width, depth = image.shape[0], image.shape[1], image.shape[2]

        keys = []
        generator = Generator(base=height * width, key=key_generator)
        while len(keys) < self.occupancy:
            coordinate = generator.next()
            while coordinate in keys:
                coordinate = generator.next()
            keys.append(coordinate)

        message_bits = []
        for coordinate in keys:
            i, j = divmod(coordinate, width)
            sigma = self.sigma
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
                message_bits.append(1)
            else:
                message_bits.append(0)
        decoded_message = KutterMethod.bits_to_str(message_bits)
        return decoded_message

    @property
    def sigma(self) -> int:
        return self._sigma

    @sigma.setter
    def sigma(self, value: int) -> None:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError('sigma > 0!')
            self._sigma = value

    @property
    def lam(self) -> float:
        return self._lam

    @lam.setter
    def lam(self, value: float) -> None:
        if isinstance(value, float):
            if value < 1E-14:
                raise ValueError('lambda > 0!')
            self._lam = value

    @property
    def occupancy(self) -> int:
        return self._occupancy


def metrics(empty_image: str, full_image: str) -> None:
    img = Image.open(empty_image).convert('RGB')
    empty = np.asarray(img, dtype='uint8')
    img.close()

    img = Image.open(full_image).convert('RGB')
    full = np.asarray(img, dtype='uint8')
    img.close()

    max_d = np.max(np.abs(empty.astype(int) - full.astype(int)))
    print('Максимальное абсолютное отклонение:\n{}'.format(max_d))

    SNR = np.sum(empty * empty) / np.sum((empty - full) ** 2)
    print('Отношение сигнал-шум:\n{}'.format(SNR))

    H, W = empty.shape[0], empty.shape[1]
    MSE = np.sum((empty - full) ** 2) / (W * H)
    print('Среднее квадратичное отклонение:\n{}\n'.format(MSE))

    sigma = np.sum((empty - np.mean(empty)) * (full - np.mean(full))) / (H * W)
    # С помощью данной метрики оцениваются
    # коррелированность, изменение динамического диапазона, а также изменение
    # среднего значения одного изображения относительно другого.
    # -1 <= UQI <= 1
    # минимальному искажению изображения соответствуют
    # значения UQI ~ 1
    UQI = (4 * sigma * np.mean(empty) * np.mean(full)) / \
          ((np.var(empty) ** 2 + np.var(full) ** 2) * (np.mean(empty) ** 2 + np.mean(full) ** 2))
    print(f'Универсальный индекс качества (УИК):\n{UQI}\n')


def dependence(key: int, old_image: str, new_image: str, message: str):
    # Готовим данные
    d = dict()

    message_bits = np.array(KutterMethod.str_to_bits(message))
    for lam in (0.5, 1, 1.5, 2, 2.5, 3):
        kutter = KutterMethod(old_image, new_image)
        kutter.lam = lam
        kutter.embed(message, key)
        for sigma in (1, 2, 3):
            kutter.sigma = sigma
            decoded_message = kutter.recover(key)
            decoded_message_bits = np.array(KutterMethod.str_to_bits(decoded_message))

            d.setdefault('lambda', []).append(lam)
            d.setdefault('sigma', []).append(sigma)
            d.setdefault('e_probability', []).append(
                np.mean(np.abs(message_bits - decoded_message_bits[:message_bits.shape[0]])) * 100)

    df = np.round(pd.DataFrame(d), decimals=5)
    df.to_csv('log.csv', sep='\t', encoding='utf-8')
    print('Таблица:')
    print(df)
    print('Корреляция:')
    print(np.round(df.corr(), decimals=2))

    df.groupby('lambda')['e_probability'].mean().plot(kind='bar', grid=True, ylim=0)
    plt.show()


def main():
    load_dotenv('.env')
    key: int = int(os.getenv('KEY'))

    old_image = 'input/old_image.png'
    new_image = 'output/new_image.png'

    with open('message.txt', mode='r', encoding=encoding) as file:
        message = file.read()

    kutter = KutterMethod(old_image, new_image)
    kutter.embed(message, key)
    decoded_message = kutter.recover(key)
    print('Ваше сообщение:\n{}'.format(decoded_message))

    metrics(old_image, new_image)
    dependence(key, old_image, 'output/image.png', message)


if __name__ == '__main__':
    main()
