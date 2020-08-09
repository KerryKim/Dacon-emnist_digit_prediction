## import libraries

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

## define dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        data = pd.read_csv(self.data_dir)

        del data['id']
        del data['letter']

        label = data['digit']
        input = []

        for i in range(2048):
            temp = data.iloc[i, 1:].values.reshape(28, 28)
            input.append(temp)

        # list 자료형으로는 img를 나타낼 수 없으므로 numpy 자료형으로 바꾸어 준다.
        # 데이터 타입 에러 발생으로 np.float32를 추가해준다. (하기 정규화시 필요)
        label = (np.array(label)).astype(np.float32)
        input = (np.array(input)).astype(np.float32)

        input = input / 255.0

        # pytorch에 들어가는 dimension은 input이 3개의 축을 가져야 한다.
        if input.ndim == 3:
            input = input[:, :, :, np.newaxis]

        # 참고> input.shape : (2048, 28, 28 ,1) / label.shape : (2048, )

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## define transform

# nnConv2D 는 nSamples x nChannels x Height x Width 의 4차원 Tensor를 입력을 기본으로 하고
# 샘플 수에 대한 차원이 없을땐 채널, 세로, 가로만 입력해도 되는듯 하다.

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # (2048, 28, 28, 1) → (2048, 1, 28, 28)
        # (샘플수, 가로, 세로, 채널) → (2048, 채널, 가로, 세로)
        # 값을 확인해봐도 자리가 바뀌는 게 아니라 순서가 바뀌어 있다.
        input = np.moveaxis(input[:, :, :, :], -1, 1)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.fliqud(label)
            input = np.fliqud(input)

        data = {'label': label, 'input': input}

        return data