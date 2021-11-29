import torch
import torch.nn as nn
from loss import batch_episym
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU

class SE(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):        
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = Conv2d(num_channels, num_channels, kernel_size=1, stride=1,bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_tensor):        
        batch_size, num_channels, H, W = input_tensor.size()       
        x = self.conv0(input_tensor) 
        x = self.in0(x) 
        x = self.bn0(x)
        input_tensor = self.relu(x)
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
    


class PCSE(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, bottleneck_width=64):
        super(PCSE, self).__init__()
        SE_channel = int(planes * (bottleneck_width / 64.))
        self.shot_cut = None
        if planes*2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes*2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, SE_channel, kernel_size=1, bias=True)
        self.in1 = nn.InstanceNorm2d(SE_channel, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(SE_channel)
        self.conv2 = SE(SE_channel)
        self.conv3 = nn.Conv2d(SE_channel, planes*2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(planes*2, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.in3(out)
        out = self.bn3(out)
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
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
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
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
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

    def forward(self, x_geo, x_down):
        embed = self.conv(x_geo)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class DP_OA_DOP_block(nn.Module):
    def __init__(self, channels, l2_nums, num):
        nn.Module.__init__(self)
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(num):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
    def forward(self, pre):
        x_down = self.down1(pre)
        x2 = self.l2(x_down)
        x_up = self.up1(pre, x2)
        return x_up


class sub_T(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)
        #self.in1 = nn.InstanceNorm2d(channels, eps=1e-5)
        #self.bn1 = nn.BatchNorm2d(channels)
        #self.Re = nn.ReLU(inplace=True)
        l2_nums = clusters
        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PCSE(channels,channels//2,1))
        self.geo = DP_OA_DOP_block(channels, clusters, 3)
        self.l1_2 = []
        self.l1_2.append(PCSE(2*channels, channels//2,1))
        for _ in range(self.layer_num//2):
            self.l1_2.append(PCSE(channels,channels//2,1))
        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
    def forward(self, data, xs):
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x2 = self.geo(x1_1)
        out = self.l1_2( torch.cat([x1_1,x2], dim=1))
        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, logits)
        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)
        return logits, e_hat, residual,out


class T_Net(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = 2
        depth_each_stage = config.net_depth//(config.iter_num+1)      
        self.subnetwork_init = sub_T(config.net_channels, 4, depth_each_stage
                                     , config.clusters)
        self.subnetwork = [sub_T(config.net_channels, 6, depth_each_stage , config.clusters) for _ in range(self.iter_num)]
        self.subnetwork = nn.Sequential(*self.subnetwork)
        self.l = []
        self.l.append(PCSE((self.iter_num+1)* config.net_channels, config.net_channels // 2, 1))
        for _ in range(3):
            self.l.append(PCSE(config.net_channels, config.net_channels // 2, 1))
        self.l = nn.Sequential(*self.l)
        self.covn = nn.Conv2d(128, 1, kernel_size=1)
        
    def forward(self, data):  
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        #data: b*1*n*c
        input = data['xs'].transpose(1,3)
        res_weights, res_e_hat = [], []
        logits, e_hat, residual,out = self.subnetwork_init(input, data['xs'])
        More_weight = out
        res_weights.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):           
            logits, e_hat, residual,out = self.subnetwork[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1), data['xs'])
            More_weight = torch.cat([More_weight, out], dim=1)
            res_weights.append(logits), res_e_hat.append(e_hat)
        More_weight = self.l(More_weight)
        logits = torch.squeeze(torch.squeeze(self.covn(More_weight), 3), 1)
        e_hat = weighted_8points(data['xs'], logits)
        res_weights.append(logits), res_e_hat.append(e_hat)
        return res_weights, res_e_hat 


        
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

