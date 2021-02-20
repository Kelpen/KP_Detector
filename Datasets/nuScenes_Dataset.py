"""
Image sequence only.

Good for unsupervised tasks.

Each image in the sequence is transformed independently.
"""

from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms as trans

import json
from PIL import Image


def get_seq_list(data_root):
    with open(data_root + 'my_anns/SeqList.txt') as f:
        seq_list = f.read().split()
    return seq_list


class ImgSubSeq(Dataset):
    def __init__(self, data_root, seqs=(), devices=(), subseq_len=0, transform=trans.ToTensor(), time_first_dim=True):
        self.subseq_len = subseq_len
        self.transform = transform
        self.time_first_dim = time_first_dim

        self.img_root = data_root + 'v1.0-trainval/'
        self.list_root = data_root + 'my_anns/ImgSeq_Lists/'

        if not seqs:  # all sequences
            seqs = get_seq_list(data_root)
        if not devices:  # all devices
            devices = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        self.all_data_list = []
        self.index_arr = []
        max_index = 0
        for seq in seqs:
            for device in devices:
                with open(self.list_root + seq + '/' + device + '.json') as seq_file:
                    data = json.load(seq_file)
                    self.all_data_list += data

                    data_len = len(data)
                    self.index_arr += list(range(max_index, max_index + data_len - subseq_len + 1))
                    max_index += data_len
        self.subseq_num = len(self.index_arr)

    def __getitem__(self, idx):
        start_frame = self.index_arr[idx]
        img_seq = []
        for i in range(self.subseq_len):
            data: dict = self.all_data_list[start_frame + i]
            img = Image.open(self.img_root + data['filename'])
            if self.time_first_dim:
                img = self.transform(img)[None]
            else:
                img = self.transform(img)[:, None]
            img_seq.append(img)
        if self.time_first_dim:
            return torch.cat(img_seq)
        else:
            return torch.cat(img_seq, dim=1)

    def __len__(self):
        return self.subseq_num


'''
if __name__ == '__main__':
    from torchvision import transforms as T

    transform = T.Compose([T.Resize((90, 160)), T.ToTensor()])
    a = ImgSubSeq([], ['CAM_FRONT'], 5, transform)
    a[5].shape

    import matplotlib.pyplot as plt

    img_seq = a[5]
    for img in img_seq:
        print(img.shape)
        plt.imshow(img.numpy().transpose(1, 2, 0))
        plt.show()
'''