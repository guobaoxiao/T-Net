import torch
import torch.nn as nn
from loss import batch_episym
from torch.nn import functional as F
import numpy as np
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU


class AFF(nn.Module):
    def __init__(self, channels, r=4):
        nn.Module.__init__(self)
        inter_channels = int(channels // r)
        
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(1)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.global_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        #x = x1 + x2        
        xl = self.local_att(x)
        
        xga = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        xga = self.global_att1(xga)
        
        xgm = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        xgm = self.global_att2(xgm)
        
        xg = xga + self.weight*xgm    
        #xg = self.global_att(xg)
        xg = xg.expand_as(xl)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)
        #one = torch.ones(wei.size()).cuda()
        #out = x + x*wei
        out = x*wei
        
        return out

class SE(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.fc3 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc4 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight1.data.fill_(1)

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        x = self.conv0(input_tensor)
        x = self.in0(x)
        x = self.bn0(x)
        input_tensor = self.relu(x)
        
        squeeze_tensor1 = F.avg_pool2d(input_tensor, (H, W), stride=(H, W)).squeeze()
        squeeze_tensor2 = F.max_pool2d(input_tensor, (H, W), stride=(H, W)).squeeze()
             
        # channel excitation
        fc_out_mean = self.relu(self.fc1(squeeze_tensor1))
        fc_out_mean = self.fc2(fc_out_mean)
        
        fc_out_max = self.relu(self.fc3(squeeze_tensor2))
        fc_out_max = self.fc4(fc_out_max)
        
        fc_out = self.sigmoid(fc_out_mean + self.weight1*fc_out_max )
                   
        output_tensor = torch.mul(input_tensor, fc_out.view(batch_size, num_channels, 1, 1))

        fs_out_1 = torch.mean(output_tensor, 1).unsqueeze(1)
        fs_out = self.sigmoid(fs_out_1)
        output_tensor = fs_out * output_tensor

        return output_tensor


class PCSE(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, bottleneck_width=64):
        super(PCSE, self).__init__()
        SE_channel = int(planes * (bottleneck_width / 64.))
        self.shot_cut = None
        if planes * 2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes * 2, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, SE_channel, kernel_size=1),
            nn.InstanceNorm2d(SE_channel, eps=1e-3),
            nn.BatchNorm2d(SE_channel),
            nn.ReLU()
        )
        self.sce = SE(SE_channel)

        self.conv2 = nn.Sequential(
            nn.Conv2d(SE_channel, planes * 2, kernel_size=1),
            nn.InstanceNorm2d(planes * 2, eps=1e-3),
            nn.BatchNorm2d(planes * 2)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.sce(out)
        out = self.conv2(out)
        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)

        return out
        
        
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points1),
                nn.ReLU(),
                nn.Conv2d(points1, points2, kernel_size=1),
                nn.BatchNorm2d(points2),
                nn.ReLU(),
                nn.Conv2d(points2, points1, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)# b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):

        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out



class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PCSE(channels, channels // 2, 1))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PCSE(2 * channels, channels // 2, 1))
        for _ in range(self.layer_num // 2):
            self.l1_2.append(PCSE(channels, channels // 2, 1))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs):
        #data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2( torch.cat([x1_1,x_up], dim=1))

        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual, out


class T_Net_plus(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num + 1
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = OANBlock(config.net_channels, 4+self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [OANBlock(config.net_channels, 6+self.side_channel, depth_each_stage, config.clusters) for _ in range(self.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        self.weights_he = []
        self.weights_he.append(AFF(3*config.net_channels))
        self.weights_he.append(PCSE(3* config.net_channels, config.net_channels // 2, 1))        
        for _ in range(3):
            self.weights_he.append(PCSE(config.net_channels, config.net_channels // 2, 1))
            
        self.weights_he = nn.Sequential(*self.weights_he)

        self.out = nn.Conv2d(config.net_channels, 1, kernel_size=1)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        #data: b*1*n*c
        input = data['xs'].transpose(1,3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual,out = self.weights_init(input, data['xs'])
        More_weight = out
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual,out = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs'])
            More_weight = torch.cat([More_weight, out], dim=1)    
            res_logits.append(logits), res_e_hat.append(e_hat)
        out = self.weights_he(More_weight)
        logits = torch.squeeze(torch.squeeze(self.out(out), 3), 1)
        e_hat = weighted_8points(data['xs'], logits)
        res_logits.append(logits), res_e_hat.append(e_hat)
        
        return res_logits, res_e_hat  


        
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    
    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

