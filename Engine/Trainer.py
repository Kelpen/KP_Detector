from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as trans
from torch import optim
from torch.nn import MSELoss
import torch

from Models import COM_Detector as Com
from Configs import Configs as Cfg
from Datasets import nuScenes_Dataset as Data
from torchvision.utils import save_image

class COMTrainer:
    def __init__(self, cfg):
        self.kp_detector = Com.KPDetector(cfg=cfg).cuda()
        self.kp_decoder = Com.KPDecoder(cfg=cfg).cuda()

        self.img_size = cfg['Dataset']['image_size']
        self.seq_len = cfg['Dataset']['seq_len']
        self.batch = cfg['Training']['batch']
        transform = trans.Compose([trans.Resize(self.img_size), trans.ToTensor()])
        dataset = Data.ImgSubSeq(cfg['Dataset']['nuScenes_root'], transform=transform, subseq_len=self.seq_len)
        self.data_loader = DataLoader(dataset, batch_size=self.batch, num_workers=4, shuffle=True, drop_last=True)

        self.optimizer = optim.Adam([{'params': self.kp_detector.parameters(), 'lr': 1e-4},
                                     {'params': self.kp_decoder.parameters(), 'lr': 1e-4}])

        self._train_epoch = cfg['Training']['epoch']
        self.loss_func = MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        counter = 0
        for epoch in range(self._train_epoch):
            for data_cpu in self.data_loader:
                data = data_cpu.cuda()
                with torch.cuda.amp.autocast():
                    first_feature, first_kp, kps = self.kp_detector(data)
                    predicted_images = self.kp_decoder(first_feature, first_kp, kps)
                    loss = torch.sum(self.loss_func(data[:, 1:], predicted_images))

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                '''
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                '''
                print(counter, loss.detach().cpu())
                if counter % 10 == 0:
                    # batch, len, 3, h, w
                    img_data = data_cpu[0].permute(1, 2, 0, 3)
                    img_pred = predicted_images[0].detach().cpu().permute(1, 2, 0, 3)
                    img_data = img_data.contiguous().view(3, self.img_size[0], self.img_size[1]*self.seq_len)
                    img_pred = img_pred.contiguous().view(3, self.img_size[0], self.img_size[1]*(self.seq_len-1))
                    final_image = torch.cat([img_data,
                                             torch.cat([torch.zeros((3, *self.img_size)), img_pred], dim=2)], dim=1)
                    save_image(final_image, 'results/com/img_results/%06d.png' % counter)
                if counter % 500 == 0:
                    torch.save(self.kp_detector, 'results/com/weights/%06d_det.pth' % counter)
                    torch.save(self.kp_decoder, 'results/com/weights/%06d_dec.pth' % counter)
                counter += 1

