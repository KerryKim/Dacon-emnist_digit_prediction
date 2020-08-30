## import libraries
import numpy as np
import torch

from torchvision import transforms

## define dataset
class Dataset(torch.utils.data.Dataset):
    # dataset preprocessing
    def __init__(self, data, mode, transform=None):
        self.data = data
        self.transform = transform
        self.mode = mode

        # train에는 input/label이 있지만 test에는 input만 있으므로 구분해 준다.
        if mode == 'train':
            # __getitem__()에 넣기 위해 list label/input 생성
            label = self.data['digit']
            lst_label = list(label.values)

            df_input = self.data.iloc[:, 3:]
        else:
            lst_label = []
            df_input = self.data.iloc[:, 2:]

        lst_input = []
        for i in range(len(df_input)):
            temp = df_input.iloc[i, :].values.reshape(28, 28)
            temp = np.where(temp >= 4, temp, 0)
            lst_input.append(temp)

        self.lst_label = lst_label
        self.lst_input = lst_input

    # dataset length
    def __len__(self):
        return len(self.data)

    # dataset에서 특정 1개의 샘플을 가져옴
    def __getitem__(self, index):
        mode = self.mode
        if mode == 'train':
            # 인덱스를 통해 그에 맵핑되는 입출력 데이터를 리턴 (즉, 1개 sample을 가져옴)
            label = self.lst_label[index]
            input = self.lst_input[index]
        else:
            label = self.lst_label
            input = self.lst_input[index]

        # list 자료형으로는 img를 나타낼 수 없으므로 numpy 자료형으로 바꾸어 준다.
        # 데이터 타입 에러 발생, 정규화를 위해 np.float32를 추가해준다.
        # label은 Net output과 연산을 위해 float형으로 바꾸지 않는다. (바꿀시 데이터형 에러 발생)
        label = np.array(label)
        input = (np.array(input)).astype(np.float32)

        input = input / 255.0

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## define transform

# nnConv2D 는 nSamples x nChannels x Height x Width 의 4차원 Tensor를 입력을 기본으로 하고
# 샘플 수에 대한 차원이 없을땐 채널, 세로, 가로만 입력해도 되는듯 하다.

class ToPILImage(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        ToPIL = transforms.ToPILImage()

        data = {'label': label, 'input': ToPIL(input)}

        return data


class RandomRotation(object):
    def __init__(self, degree=5):
        self.degree = degree

    def __call__(self, data):
        label, input = data['label'], data['input']

        R_r = transforms.RandomRotation(self.degree)

        data = {'label': label, 'input': R_r(input)}

        return data


class RandomAffine(object):
    def __init__(self, degree=5):
        self.degree = degree

    def __call__(self, data):
        label, input = data['label'], data['input']

        R_a = transforms.RandomAffine(self.degree)

        data = {'label': label, 'input': R_a(input)}

        return data


class ToNumpy(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        data = {'label': label, 'input': np.array(input)}

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        return data


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # PIL -> Numpy로 변환하면서 축이 하나 사라졌다.
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # (28, 28, 1) → (1, 28, 28)
        # (가로, 세로, 채널) → (채널, 가로, 세로)
        # 값을 확인해봐도 자리가 바뀌는 게 아니라 순서가 바뀌어 있다.
        input = np.moveaxis(input[:, :, :], -1, 0)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data