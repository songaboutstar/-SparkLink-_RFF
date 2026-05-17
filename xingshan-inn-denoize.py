import glob
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_iq(iq):
    max_val = np.max(np.abs(iq)) + 1e-8
    return iq / max_val


def plot_iq(gt, noisy, denoised, epoch=0, save_only=True):
    # 创建 plots 目录
    os.makedirs('./plots', exist_ok=True)

    # I 通道
    plt.figure(figsize=(12, 4))
    plt.plot(gt[0], label='GT (I)')
    plt.plot(noisy[0], label='Noisy (I)')
    plt.plot(denoised[0], label='Denoised (I)')
    plt.legend();
    plt.title(f'I Channel Denoising - Epoch {epoch}')
    if save_only:
        plt.savefig(f'./plots/epoch_{epoch}_I.png')
    else:
        plt.show()
    plt.close()

    # Q 通道
    plt.figure(figsize=(12, 4))
    plt.plot(gt[1], label='GT (Q)')
    plt.plot(noisy[1], label='Noisy (Q)')
    plt.plot(denoised[1], label='Denoised (Q)')
    plt.legend();
    plt.title(f'Q Channel Denoising - Epoch {epoch}')
    if save_only:
        plt.savefig(f'./plots/epoch_{epoch}_Q.png')
    else:
        plt.show()
    plt.close()


# ===========================
# BaseModel 保持不变（添加 resume 支持）
# ===========================
class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] and torch.cuda.is_available() else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        if cur_iter < warmup_iter:
            init_lr_g_l = self._get_init_lr()
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=self.device)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)


# ===========================
# 损失函数优化（不变）
# ===========================
class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l1', eps=1e-3):
        super().__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(x.unsqueeze(-1), size=(1, target.shape[-1]), mode='linear').squeeze(-1)
        diff = x - target
        if self.losstype == 'l2':
            loss = torch.mean(diff ** 2)
        elif self.losstype == 'l1':
            loss = torch.mean(torch.abs(diff))
        else:
            raise ValueError("Unknown loss type")
        return loss


class Gradient_Loss(nn.Module):
    def __init__(self, losstype='l1'):
        super().__init__()
        kernel = torch.tensor([[-1, 2, -1]], dtype=torch.float32).unsqueeze(0)
        self.conv = nn.Conv1d(2, 2, 3, padding=1, groups=2, bias=False)
        self.conv.weight.data = kernel.repeat(2, 1, 1)
        self.conv.weight.requires_grad = False
        self.Loss_criterion = nn.L1Loss()

    def forward(self, x, y):
        x_g = self.conv(x)
        y_g = self.conv(y)
        return self.Loss_criterion(x_g, y_g)


class SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.window = nn.AvgPool1d(11, 1, padding=5)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        mu_x = self.window(x)
        mu_y = self.window(y)
        sigma_x_sq = self.window(x ** 2) - mu_x ** 2
        sigma_y_sq = self.window(y ** 2) - mu_y ** 2
        sigma_xy = self.window(x * y) - mu_x * mu_y
        sigma_x_sq = torch.clamp(sigma_x_sq, min=1e-6)  # 关键：clamp 防止负值
        sigma_y_sq = torch.clamp(sigma_y_sq, min=1e-6)
        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        ssim = torch.mean(ssim_n / ssim_d, dim=-1)
        return 1 - ssim.mean()


# ===========================
# Glow1D 子模块优化（不变）
# ===========================
class ResBlock1D(nn.Module):
    def __init__(self, channel_in, channel_out, feature=128):  # feature 增加
        super().__init__()
        self.conv1 = nn.Conv1d(channel_in, feature, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(feature, feature, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv1d(channel_in + feature, channel_out, 3, padding=1)

    def forward(self, x):
        residual = self.act1(self.conv1(x))
        residual = self.act2(self.conv2(residual))
        out = self.conv3(torch.cat([x, residual], dim=1))
        return out


def subnet1D(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'Resnet':
            return ResBlock1D(channel_in, channel_out)
        else:
            return None

    return constructor


class ActNorm1D(nn.Module):
    def __init__(self, num_channels, logdet=False):
        super().__init__()
        self.num_channels = num_channels
        self.logdet = logdet
        self.mean = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.initialized = False

    def forward(self, x, reverse=False):
        batch_size = x.shape[0]
        if not self.initialized:
            self.initialize(x)
        if not reverse:
            z = x * torch.exp(self.log_scale) + self.mean
            log_det = torch.sum(self.log_scale) * x.shape[2]
        else:
            z = (x - self.mean) * torch.exp(-self.log_scale)
            log_det = -torch.sum(self.log_scale) * x.shape[2]

        log_det = log_det * torch.ones(batch_size, device=x.device)
        return z, log_det

    def initialize(self, x):
        with torch.no_grad():
            self.mean.data.copy_(-x.mean(dim=(0, 2), keepdim=True))
            self.log_scale.data.copy_(torch.log1p(x.std(dim=(0, 2), keepdim=True)))
            self.initialized = True


class InvConv1D(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        # QR 初始化
        w = torch.linalg.qr(torch.randn(num_channels, num_channels), mode='reduced')[0]

        self.weight = nn.Parameter(w)
        self.logdet = torch.slogdet(self.weight)[1] * num_channels

    def forward(self, x, reverse=False):
        batch_size, channels, length = x.shape
        weight = self.weight.view(channels, channels, 1)
        if not reverse:
            z = F.conv1d(x, weight)
            log_det = self.logdet * length
        else:
            try:
                w_inv = torch.inverse(self.weight + 1e-6 * torch.eye(channels, device=x.device)).view(channels,
                                                                                                      channels, 1)
            except:
                w_inv = torch.eye(channels, device=x.device).view(channels, channels, 1)
            z = F.conv1d(x, w_inv)
            log_det = -self.logdet * length
        log_det = log_det * torch.ones(batch_size, device=x.device)
        return z, log_det


class AffineCoupling1D(nn.Module):
    def __init__(self, num_channels, subnet_constructor):
        super().__init__()
        self.split_len1 = num_channels // 2
        self.split_len2 = num_channels - self.split_len1
        self.nn = subnet_constructor(self.split_len1, self.split_len2 * 2)

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split_len1, :], x[:, self.split_len1:, :]
        s_t = self.nn(x1)
        s, t = s_t[:, :self.split_len2, :], s_t[:, self.split_len2:, :]
        # (3) 改AffineCoupling中的scale函数
        s = torch.tanh(s)  # 代替 s = 0.5 * torch.tanh(s)
        if not reverse:
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=(1, 2))
        else:
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -torch.sum(s, dim=(1, 2))
        z = torch.cat([y1, y2], dim=1)
        return z, log_det


class Glow1DBlock(nn.Module):
    def __init__(self, num_channels, subnet_constructor):
        super().__init__()
        self.actnorm = ActNorm1D(num_channels)
        self.invconv = InvConv1D(num_channels)
        self.affine = AffineCoupling1D(num_channels, subnet_constructor)

    def forward(self, x, reverse=False):
        log_det_total = 0
        if not reverse:
            z, log_det = self.actnorm(x)
            if torch.isnan(z).any():
                print("NaN in ActNorm forward")
                z = torch.nan_to_num(z, nan=0.0)
            log_det_total += log_det
            z, log_det = self.invconv(z)
            if torch.isnan(z).any():
                print("NaN in InvConv forward")
                z = torch.nan_to_num(z, nan=0.0)
            log_det_total += log_det
            z, log_det = self.affine(z)
            if torch.isnan(z).any():
                print("NaN in Affine forward")
                z = torch.nan_to_num(z, nan=0.0)
            log_det_total += log_det
        else:
            # 类似添加 reverse 检查
            z, log_det = self.affine(x, reverse=True)
            if torch.isnan(z).any():
                print("NaN in Affine reverse")
                z = torch.nan_to_num(z, nan=0.0)
            log_det_total += log_det
            z, log_det = self.invconv(z, reverse=True)
            if torch.isnan(z).any():
                print("NaN in InvConv reverse")
                z = torch.nan_to_num(z, nan=0.0)
            log_det_total += log_det
            z, log_det = self.actnorm(z, reverse=True)
            if torch.isnan(z).any():
                print("NaN in ActNorm reverse")
                z = torch.nan_to_num(z, nan=0.0)
            log_det_total += log_det
        return z, log_det_total


class Glow1DNet(nn.Module):
    def __init__(self, channel_in=2, num_blocks=24, hidden_channels=128, subnet_type='Resnet'):
        super().__init__()
        subnet_constructor = subnet1D(subnet_type)
        self.blocks = nn.ModuleList([
            Glow1DBlock(channel_in, subnet_constructor) for _ in range(num_blocks)
        ])

    def forward(self, x, reverse=False):
        log_det_total = 0
        z = x
        if not reverse:
            for block in self.blocks:
                z, log_det = block(z)
                log_det_total += log_det
        else:
            for block in reversed(self.blocks):
                z, log_det = block(z, reverse=True)
                log_det_total += log_det
        return z, log_det_total


# ===========================
# 数据集（不变）
# ===========================
class IQDenoiseDataset(Dataset):
    def __getitem__(self, idx):
        item = self.samples[idx]

        iq = np.fromfile(item["path"], dtype=np.complex64)
        iq = iq[:self.seq_length]

        i = np.real(iq)
        q = np.imag(iq)
        clean = np.stack([i, q], axis=0)

        # 统一幅度归一化（必须保留）
        clean = clean / (np.max(np.abs(clean)) + 1e-8)

        return {
            "GT": torch.tensor(clean, dtype=torch.float32),
            "Noisy": torch.tensor(clean, dtype=torch.float32),  # ★等于 GT
            "LQ": torch.tensor(clean, dtype=torch.float32),
            "device": item["device"],
            "file_name": item["file"]
        }

# ===========================
# 模型（不变）
# ===========================
class InvDN_Model(BaseModel):
    def __init__(self, opt):
        self.log_dict = OrderedDict()
        self.lambda_fp = 0.01 # RF fingerprint preservation weight（0.01~0.1）
        super().__init__(opt)
        self.netG = Glow1DNet(
            channel_in=opt['network_G']['in_nc'],
            num_blocks=opt['network_G']['block_num']
        ).to(self.device)

        if self.is_train:
            self.Reconstruction_forw = ReconstructionLoss('l1').to(self.device)
            self.Reconstruction_back = ReconstructionLoss('l1').to(self.device)
            self.Rec_Grad = Gradient_Loss().to(self.device)
            self.Rec_SSIM = SSIM_Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt['lr_G'],
                                                betas=(opt.get('beta1', 0.9), opt.get('beta2', 0.999)),
                                                weight_decay=opt.get('weight_decay_G', 0))
            self.optimizers.append(self.optimizer_G)
            self.schedulers.append(
                torch.optim.lr_scheduler.StepLR(self.optimizer_G,
                                                step_size=opt.get('lr_steps', 10000),
                                                gamma=opt.get('lr_gamma', 0.1))
            )
        # 新增：加载 checkpoint
        if opt.get('resume_epoch', None):
            self.load_from_epoch(opt['resume_epoch'])
        self.load()
        if self.is_train:
            self.print_network()
        self.log_dict = OrderedDict()

    def load_from_epoch(self, epoch):
        load_path = os.path.join(self.opt['path']['models'], f'{epoch}_G.pth')
        if os.path.exists(load_path):
            self.load_network(load_path, self.netG)
            state_path = os.path.join(self.opt['path']['training_state'], f'{epoch}.state')
            if os.path.exists(state_path):
                resume_state = torch.load(state_path, map_location=self.device)
                self.resume_training(resume_state)
            print(f"Resumed from epoch {epoch}")
        else:
            print(f"No checkpoint found for epoch {epoch}")

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # shape -> [batch, 2, seq_length]
        self.real_H = data['GT'].to(self.device)
        self.noisy_H = data['Noisy'].to(self.device)

    def loss_forward(self, out, y):
        return self.Reconstruction_forw(out, y)

    def loss_backward(self, x, y):
        x_samples, _ = self.netG(x=y, reverse=True)
        l_back = self.Reconstruction_back(x, x_samples)
        l_grad = 0.5 * self.Rec_Grad(x, x_samples)  # 升到 0.5，强调边缘平滑
        l_ssim = 0.5 * self.Rec_SSIM(x, x_samples)  # 同，强调结构相似
        return l_back + l_grad + l_ssim

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.output, _ = self.netG(self.noisy_H)
        self.fake_H = self.output  # 保证 fake_H 是最新 forward 输出
        epoch = self.opt.get('current_epoch', 0)  # 需要在训练循环中给 opt['current_epoch'] 赋值
        lambda_fp = min(self.lambda_fp, 0.01 + epoch * 0.001)
        if torch.isnan(self.output).any():
            print("Warning: NaN in forward output")
            return  # 跳过此步
        l_forw = self.loss_forward(self.output, self.ref_L)

        l_back = self.loss_backward(self.real_H, self.output)
        if torch.isnan(l_back).any():
            print("Warning: NaN in backward loss")
            return

        alpha_forw = 1.0  # 弱化 l_forw (从 1.0 降到 0.01)
        self.output = torch.tanh(self.output)  # 限制 latent 到 [-1,1]，防爆炸
        l_forw = alpha_forw * l_forw

        # ===== 指纹保护：Residual Energy Preservation =====
        self.fake_H = self.output  # 添加在 optimize_parameters() 开头
        residual = self.noisy_H - self.fake_H
        fp_loss = -torch.mean(residual ** 2)

        loss = l_forw + l_back + self.lambda_fp * fp_loss

        if torch.isnan(loss).any():
            print("Warning: NaN in total loss")
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)
        self.optimizer_G.step()
        self.log_dict['l_fwd'] = l_forw.item()
        self.log_dict['l_bwd'] = l_back.item()
        self.log_dict['l_fp'] = fp_loss.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            output, _ = self.netG(self.noisy_H)
            gaussian_scale = self.opt.get('gaussian_scale', 0.01)
            y_ = output + gaussian_scale * torch.randn_like(output)
            y_ = torch.clamp(y_, min=-3.0, max=3.0)
            fake_H_test, _ = self.netG(y_, reverse=True)  # 用独立变量，不覆盖训练阶段 self.fake_H
        self.netG.train()
        return fake_H_test

    def get_current_visuals(self):
        return OrderedDict({
            'LR_ref': self.ref_L.detach()[0].cpu(),
            'Denoised': self.fake_H.detach()[0].cpu(),
            'GT': self.real_H.detach()[0].cpu(),
            'Noisy': self.noisy_H.detach()[0].cpu()

        })

    def get_current_losses(self):
        return self.log_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        print(f"Model G structure: {s}")
        print(f"Number of parameters: {n}")


# ===========================
# 训练/测试示例（添加可视化 + checkpoint）
# ===========================
if __name__ == "__main__":
    # 配置示例
    data_dir = "/home/custdev/wangguanglong/xingshan-data/yuanshi"  # 原始IQ数据目录
    opt = {
        'gpu_ids': [0],
        'is_train': True,
        'lr_G': 0.0001,
        'network_G': {'in_nc': 2, 'block_num': 16},
        'path': {
            'models': '/home/custdev/wangguanglong/kd2-model',         # 模型保存路径
            'training_state': '/home/custdev/wangguanglong/kd2-model', # 训练状态保存路径
            'results': '/home/custdev/wangguanglong/xingshan-data/yuanshi-add-denoize/'        # 去噪后的IQ保存路径
        },
        'gaussian_scale': 0.005,
        'lr_steps': 1600,
        'lr_gamma': 0.1,
        'resume_epoch': 500,
    }

    # ===========================
    # 创建目录
    # ===========================
    os.makedirs(opt['path']['models'], exist_ok=True)
    os.makedirs(opt['path']['training_state'], exist_ok=True)
    os.makedirs(opt['path']['results'], exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    # ===========================
    # Dataset & DataLoader
    # ===========================
    dataset = IQDenoiseDataset(
        root_dir=data_dir,
        seq_length=1024,
        snr_db=20
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # 一次处理一个文件，确保 file_name 匹配

    # ===========================
    # 打印输入 SNR
    # ========================
    print("Dataset noise setting: SNR = 20 dB (Gaussian white noise)")
    # ===========================
    # 模型初始化
    # ===========================
    model = InvDN_Model(opt)
    start_epoch = opt.get('resume_epoch', 0) if opt.get('resume_epoch') else 0

    best_psnr = -float('inf')
    best_epoch = -1

    # ===========================
    # 主训练循环
    # ===========================
    for epoch in range(start_epoch, 500):
        opt['current_epoch'] = epoch

        epoch_psnr_list = []
        epoch_snr_list = []
        epoch_mse_list = []

        for i, data in enumerate(loader):

            device = data['device'][0]
            file_name = data['file_name'][0]
            model.feed_data(data)

            # ======== 前向 + 反向训练 ========
            model.optimize_parameters()
            losses = model.get_current_losses()
            loss_fwd = losses.get('l_fwd', 0.0)
            loss_bwd = losses.get('l_bwd', 0.0)
            loss_total = loss_fwd + loss_bwd

            # ======== 测试阶段（计算去噪指标）========
            denoised = model.test()  # 输出 GPU Tensor
            visuals = model.get_current_visuals()
            denoised = visuals['Denoised'].to(model.noisy_H.device)
            gt = visuals['GT'].to(model.noisy_H.device)

            mse = torch.mean((denoised - gt) ** 2)
            psnr = 20 * torch.log10(torch.tensor(1.0, device=denoised.device) / torch.sqrt(mse))
            signal_power = torch.mean(gt ** 2)
            noise_power = torch.mean((denoised - gt) ** 2)
            snr = 10 * torch.log10(signal_power / noise_power)

            residual_energy = torch.mean((denoised - model.noisy_H) ** 2)

            epoch_psnr_list.append(psnr.item())
            epoch_snr_list.append(snr.item())
            epoch_mse_list.append(mse.item())
            # ===========================
            # 每个 epoch 结束时计算平均指标
            # ===========================
            epoch_psnr_avg = sum(epoch_psnr_list) / len(epoch_psnr_list)
            epoch_snr_avg = sum(epoch_snr_list) / len(epoch_snr_list)
            epoch_mse_avg = sum(epoch_mse_list) / len(epoch_mse_list)

            # 计算残差能量 Residual Energy（对每个 batch）
            residual_energy_list = []
            for data in loader:
                model.feed_data(data)
                denoised = model.test()  # GPU Tensor
                residual = denoised - data['Noisy'].to(model.noisy_H.device)
                residual_energy_list.append(torch.mean(residual ** 2).item())
            epoch_residual_avg = sum(residual_energy_list) / len(residual_energy_list)

            print(f"[Epoch {epoch}] Avg PSNR: {epoch_psnr_avg:.2f} dB, "
                  f"Avg SNR: {epoch_snr_avg:.2f} dB, "
                  f"Avg MSE: {epoch_mse_avg:.6f}, "
                  f"Avg Residual Energy: {epoch_residual_avg:.6f}")

            if i % 10 == 0:
                print(f"[Epoch {epoch}][Iter {i}] "
                      f"Loss_fwd: {loss_fwd:.4f}, Loss_bwd: {loss_bwd:.4f}, Loss_total: {loss_total:.4f} | "
                      f"PSNR: {psnr.item():.2f} dB, SNR: {snr.item():.2f} dB, MSE: {mse.item():.6f}, "
                      f"Residual Energy: {residual_energy.item():.6f}")

            # ======== 🔹 验证阶段（计算 PSNR / SNR / MSE）========
            # ======== 🔹 验证阶段（计算 PSNR / SNR / MSE）========
            model.test()
            visuals = model.get_current_visuals()
            denoised = visuals['Denoised'].to(model.noisy_H.device)  # 保持在 GPU
            gt = visuals['GT'].to(model.noisy_H.device)  # 保持在 GPU

            mse = torch.mean((denoised - gt) ** 2)
            psnr = 20 * torch.log10(torch.tensor(1.0, device=denoised.device) / torch.sqrt(mse))

            signal_power = torch.mean(gt ** 2)
            noise_power = torch.mean((denoised - gt) ** 2)
            snr = 10 * torch.log10(signal_power / noise_power)

            # 可选：残差能量观察指纹
            residual_energy = torch.mean((denoised - model.noisy_H) ** 2)

            # ======== 🔹 保存带噪声 + 去噪数据 ========
            # ======== 🔹 固定保存到 20 dB / 设备名 目录 ========
            save_dir = os.path.join(opt['path']['results'], "20", device)
            os.makedirs(save_dir, exist_ok=True)

            # ======== 保存去噪 IQ（保持原文件名）========
            denoised_save_path = os.path.join(save_dir, file_name)
            denoised_np = denoised.detach().cpu().numpy()  # 转 CPU 再保存
            complex_iq = denoised_np[0] + 1j * denoised_np[1]
            complex_iq.astype(np.complex64).tofile(denoised_save_path)

            # print(f"💾 Denoised saved: {denoised_save_path}\n")

            # 更新最佳模型
            if psnr > best_psnr:
                best_psnr = psnr
                best_epoch = epoch
                model.save_network(model.netG, 'G_best', epoch)

                # 🔴 只在这里保存 IQ
                save_dir = os.path.join(opt['path']['results'], "20", device)
                os.makedirs(save_dir, exist_ok=True)

                denoised_save_path = os.path.join(save_dir, file_name)
                denoised = visuals['Denoised'].numpy()
                complex_iq = denoised[0] + 1j * denoised[1]
                complex_iq.astype(np.complex64).tofile(denoised_save_path)

    print(f"✅ Training finished. Best PSNR = {best_psnr:.2f} dB at epoch {best_epoch}")

    # ===========================
    # 最终评估与绘图
    # ===========================
    print("\n🔍 Loading best model for final evaluation...")
    best_model_path = os.path.join(opt['path']['models'], f'{best_epoch}_G_best.pth')
    model.load_network(best_model_path, model.netG)

    batch = next(iter(loader))
    model.feed_data(batch)
    model.test()

    visuals = model.get_current_visuals()
    gt = visuals['GT'].numpy()
    noisy = visuals['Noisy'].numpy()
    denoised = visuals['Denoised'].numpy()
    plot_iq(gt, noisy, denoised, epoch=best_epoch, save_only=True)

    print(f"💾 Final model tested (epoch {best_epoch})")

