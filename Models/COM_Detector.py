"""
Center of Mass Keypoint Detector
The position is determined by the center of mass.
"""

from torch import nn
import torch

from Models import Utils


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.enc_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(3, 96, 6, padding=2, stride=2)),
            nn.ReLU(inplace=True),
        )
        self.enc_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(96, 192, 6, padding=2, stride=2, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(192, 192, 3, padding=1, stride=1, groups=3)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(192, 192, 3, padding=1, stride=1, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(192, 192, 3, padding=1, stride=1, groups=3)),
            nn.ReLU(inplace=True),
        )
        self.enc_3 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(192, 384, 6, padding=2, stride=2, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(384, 384, 3, padding=1, stride=1, groups=3)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(384, 384, 3, padding=1, stride=1, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(384, 384, 3, padding=1, stride=1, groups=3)),
        )

    def forward(self, img):
        code1 = self.enc_1(img)
        code2 = self.enc_2(code1)
        code3 = self.enc_3(code2)
        return code3


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.dec_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(384+96+96, 384, 3, padding=1, stride=1, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(384, 384, 3, padding=1, stride=1, groups=3)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(384, 384, 3, padding=1, stride=1, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(384, 384, 3, padding=1, stride=1, groups=3)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, padding=1, stride=2, groups=4),
            nn.ReLU(inplace=True),
        )
        self.dec_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(192, 192, 3, padding=1, stride=1, groups=3)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(192, 192, 3, padding=1, stride=1, groups=4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(192, 192, 3, padding=1, stride=1, groups=3)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, padding=1, stride=2, groups=4),
            nn.ReLU(inplace=True),
        )
        self.dec_3 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(96, 96, 3, padding=1, stride=1, groups=3)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(96, 96, 3, padding=1, stride=1, groups=2)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(96, 96, 3, padding=1, stride=1, groups=3)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, padding=1, stride=2, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1, padding=0, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        code1 = self.dec_1(img)
        code2 = self.dec_2(code1)
        code3 = self.dec_3(code2)
        return code3


class KPDetector(nn.Module):
    def __init__(self, cfg):
        super(KPDetector, self).__init__()
        self.image_shape = cfg['Dataset']['image_size']
        self.kp_map_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8]
        self.seq_len = cfg['Dataset']['seq_len']

        self.img_encoder = ImageEncoder()
        self.keypoint_detector = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, 3, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, groups=2),
            nn.Sigmoid(),
        )
        self.com_det = Utils.CenterOfMass(self.kp_map_shape)

    def forward(self, img_seq):
        # img_seq shape = (batch, seq_len, 3, h, w)
        img_seq = torch.split(img_seq, 1, dim=1)
        kp_seq = []
        first_feature = self.img_encoder(img_seq[0].squeeze(1))
        first_kp = self.com_det(self.keypoint_detector(first_feature))
        for img in img_seq[1:]:
            # img shape = (batch, 1, 3, h, w)
            code = self.img_encoder(img.squeeze(1))
            kp_heatmap = self.keypoint_detector(code)
            kp_list = self.com_det(kp_heatmap)
            # kp_list shape = (batch, 96, 2)
            kp_seq.append(kp_list)
        return first_feature, first_kp, kp_seq


class KPDecoder(nn.Module):
    def __init__(self, cfg):
        super(KPDecoder, self).__init__()

        self.image_shape = cfg['Dataset']['image_size']
        self.kp_map_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8]
        self.seq_len = cfg['Dataset']['seq_len']

        self.kp_mapper = Utils.Point2Map(img_shape=self.kp_map_shape)
        self.img_decoder = ImageDecoder()

    def forward(self, first_frame, first_keypoint, kp_seq):
        """

        :param first_frame:
            Single frame feature map from ImageEncoder
            shape = (batch, 384, h, w)

        key-point shape = (batch, h, w, 2), note xy-coordination is at the last dimension
        :param first_keypoint: keypoint
        :param kp_seq: List of key-points
        :return:
        """
        recovered_img_list = []

        first_frame_kp_map = self.kp_mapper(first_keypoint)
        for kp in kp_seq:
            current_kp_map = self.kp_mapper(kp)
            all_feature = torch.cat([first_frame, first_frame_kp_map, current_kp_map], dim=1)
            recovered_img = self.img_decoder(all_feature)
            recovered_img_list.append(recovered_img.unsqueeze(1))

        return torch.cat(recovered_img_list, dim=1)
