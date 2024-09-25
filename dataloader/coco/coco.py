import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class COCODataset(Dataset):

    def __init__(self, root='', train=True, index=None, base_sess=None, autoaug=False):
        self.root = os.path.expanduser(root)
        self.train = train  # 是否为训练集
        self._pre_operate(self.root)

        if autoaug is False:
            # 不使用自动增强
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            # 使用自动增强
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    # 自动增强策略（如果需要）
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def _pre_operate(self, root):
        """遍历训练集或验证集文件夹，收集图像路径和标签"""
        if self.train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'val')

        self.data = []
        self.targets = []
        self.data2label = {}

        classes = sorted(os.listdir(data_path))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_folder = os.path.join(data_path, cls_name)
            img_files = os.listdir(cls_folder)
            for img_file in img_files:
                img_path = os.path.join(cls_folder, img_file)
                self.data.append(img_path)
                self.targets.append(class_to_idx[cls_name])
                self.data2label[img_path] = class_to_idx[cls_name]

    def SelectfromTxt(self, data2label, index_path):
        """从给定索引路径中选择图像"""
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        """根据给定的类索引从数据集中选择图像"""
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, target = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, target


if __name__ == '__main__':
    # 示例：加载训练集和测试集数据
    txt_path = "../../data/index_list/coco/session_1.txt"
    base_class = 60  # 基础类数量
    class_index = np.arange(base_class)
    dataroot = '~/datasets/coco'
    batch_size_base = 128

    trainset = COCODataset(root=dataroot, train=True, index=class_index, base_sess=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8, pin_memory=True)

    testset = COCODataset(root=dataroot, train=False, index=class_index, base_sess=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size_base, shuffle=False, num_workers=8, pin_memory=True)

    cls = np.unique(trainset.targets)
    print(f"训练集类别数: {len(cls)}")
