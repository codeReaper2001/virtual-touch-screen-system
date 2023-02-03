import numpy as np
from enum import Enum
from typing import List
import math
import random


class RotateDir(Enum):
    X = 0
    Y = 1
    Z = 2


def get_rotate_mat(rotate_dir: RotateDir, theta: float):
    theta = math.radians(theta)

    def x():
        return np.array([
            [1, 0, 0, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1],
        ])

    def y():
        return np.array([
            [math.cos(theta), 0, math.sin(theta), 0],
            [0, 1, 0, 0],
            [-math.sin(theta), 0, math.cos(theta), 0],
            [0, 0, 0, 1],
        ])

    def z():
        return np.array([
            [math.cos(theta), -math.sin(theta), 0, 0],
            [math.sin(theta), math.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
    switch_map = {
        RotateDir.X: x,
        RotateDir.Y: y,
        RotateDir.Z: z,
    }
    func = switch_map[rotate_dir]
    rotate = func()
    return rotate


class LmDataGenerator():
    def __init__(self, rotate_range: float) -> None:
        self.rotate_range = rotate_range
        self.rotate_dirs = [RotateDir.X, RotateDir.Y, RotateDir.Z]

    def __get_enhanced_data_from_dataline(self, dataline: List[float], flip=False) -> List[List[float]]:
        res = []
        line = np.array(dataline)
        cxyz = line.reshape(-1, 3)  # [x, y, z]
        if flip:
            flip_vec = np.array([-1, 1, 1])
            cxyz = cxyz * flip_vec
        # [x
        #  y
        #  z]
        lxyz = cxyz.T.copy()
        lxyz = np.row_stack((lxyz, np.full([21], fill_value=1)))

        # 每个原始数据生成20个增强数据
        for i in range(20):
            # 随机一个方向和一个旋转角度
            rotate_dir = self.rotate_dirs[random.randint(
                0, len(self.rotate_dirs) - 1)]
            rotate_theta = random.uniform(-self.rotate_range,
                                        self.rotate_range)
            print(i, rotate_dir, rotate_theta)
            rotate_mat = get_rotate_mat(rotate_dir, rotate_theta)
            # 旋转
            rotated_record = np.dot(rotate_mat, lxyz)

            # 整理成需要的格式返回
            ret_line = rotated_record[:-1, :].T
            ret_line = ret_line.reshape(21 * 3)
            ret_line = ret_line.tolist()
            res.append(ret_line)
        return res

    def get_enhanced_data(self, dataset: List[List[float]], add_flip=False) -> List[List[float]]:
        res = []
        for dataline in dataset:
            line_enhanced_data = self.__get_enhanced_data_from_dataline(dataline)
            res.extend(line_enhanced_data)
            if not add_flip:
                continue
            line_flip_enhanced_data = self.__get_enhanced_data_from_dataline(dataline, True)
            res.extend(line_flip_enhanced_data)
        return res


if __name__ == "__main__":
    DATA_LEN = 21 * 3
    dataset = '../model/sign_classifier/sign.csv'
    data:List[List[float]] = []
    import csv
    with open(dataset) as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(list(map(float, line[1:])))
    # data = np.loadtxt(dataset, delimiter=',', dtype='float32',
    #                   usecols=list(range(1, DATA_LEN + 1)))
    gen = LmDataGenerator(rotate_range=30)
    idx = 170
    enhanced_data = gen.get_enhanced_data([data[idx]])
    # enhanced_data = gen.__get_enhanced_data_from_dataline(data[idx], True)
    # print(enhanced_data)

    import data_visualization
    data_visualization.show_dataline_img([data[idx], enhanced_data[10], enhanced_data[11]])