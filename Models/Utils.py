"""
Some util models

"""

from torch import nn
import torch


class CenterOfMass(nn.Module):
    def __init__(self, img_shape):
        # This model only accepts images with constant shape.
        # Can be extended to any shape.
        # (-1, -1) is the "center" of the upper left pixel
        super(CenterOfMass, self).__init__()
        h, w = img_shape
        self.grid_x, self.grid_y = torch.meshgrid(torch.linspace(-1, 1, steps=h), torch.linspace(-1, 1, steps=w))
        self.grid_x = self.grid_x.cuda()
        self.grid_y = self.grid_y.cuda()

    def forward(self, feature_map):
        """

        :param feature_map:
            shape: (Batch, n, h, w)
            all value should >= 0
        :return:
        """
        all_weight_sum = torch.sum(feature_map, dim=(2, 3))
        x_sum = torch.sum(feature_map * self.grid_x, dim=(2, 3))
        y_sum = torch.sum(feature_map * self.grid_y, dim=(2, 3))
        return torch.cat([(x_sum/all_weight_sum).unsqueeze(-1), (y_sum/all_weight_sum).unsqueeze(-1)], dim=-1)


class Point2Map(nn.Module):
    def __init__(self, img_shape, point_size=10):
        super(Point2Map, self).__init__()
        self.h, self.w = img_shape
        self.y_coord = torch.linspace(-1, 1, steps=self.h).reshape(1, 1, self.h, 1).cuda()
        self.x_coord = torch.linspace(-1, 1, steps=self.w).reshape(1, 1, 1, self.w).cuda()
        self.point_size = point_size
        self.sigma_x = (self.point_size / self.w) ** 2
        self.sigma_y = (self.point_size / self.h) ** 2

    def forward(self, kp_list):
        batch, kp_num, _ = kp_list.shape
        x, y = torch.split(kp_list, 1, dim=-1)
        x = x.view(batch, kp_num, 1, 1)
        y = y.view(batch, kp_num, 1, 1)
        x = torch.exp(-torch.square(x - self.x_coord) / self.sigma_x)
        y = torch.exp(-torch.square(y - self.y_coord) / self.sigma_y)
        return x*y


if __name__ == '__main__':
    kp = torch.Tensor([[[0.3, 0.3], [0.7, 0.5], [0.2, -0.2]]])
    print(kp.shape)
    mapper = Point2Map((600, 900))
    f = mapper(kp)
    import matplotlib.pyplot as plt
    plt.imshow(f[0, 0] + f[0, 1] + f[0, 2])
    plt.show()
