import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
import torch.nn.init as init
from torch_geometric.nn import GCNConv, SAGPooling, ASAPooling

from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer, STN3d, STNkd
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:0')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class DGCNN(nn.Module):
    def __init__(self, args, cls = -1):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k1
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        if cls != -1:
            self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=args.dropout)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=args.dropout)
            self.linear3 = nn.Linear(256, output_channels)
        
        self.cls = cls
        
        self.inv_head = nn.Sequential(
                            nn.Linear(args.emb_dims * 2, args.emb_dims),
                            nn.BatchNorm1d(args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.emb_dims, 256)
                            )

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        feat = x
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)
        
        inv_feat = self.inv_head(feat)
        
        return x, inv_feat, feat


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class MERIT(nn.Module):
    def __init__(self,args,moving_average_decay):
        super().__init__()

        self.online_encoder = pointnet_seg(args)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self):
        # model1 = self.online_encoder(args)
        model2 = self.target_encoder
        return model2







class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class pointnet(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(pointnet, self).__init__()
        if normal_channel:
            channel = 3
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return  x, trans_feat

class pointnet_seg(nn.Module):
    def __init__(self,args, seg_num_all = None, pretrain = True,normal_channel=True):
        super(pointnet_seg, self).__init__()
        if normal_channel:
            channel = 3
        else:
            channel = 3
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k1
        self.pretrain = pretrain
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv7 = self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)


        self.fstn = STNkd(k=128)
        # self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs1 = torch.nn.Conv1d(2896, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        # self.convs4 = nn.Conv1d(128, self.seg_num_all, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        # self.conv = nn.Conv1d(2048,1024,1,bias=False)
        self.l = nn.Linear(2048,256)
        self.inv_head = nn.Sequential(
            nn.Linear(args.emb_dims, args.emb_dims),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(inplace=True),
            nn.Linear(args.emb_dims, 256)
        )
        if not self.pretrain:
            self.convs4 = nn.Conv1d(128, self.seg_num_all, 1)
            self.convs5 = nn.Conv1d(1088,3088,1)
            self.bns5 = nn.BatchNorm1d(3088)
            self.l5 = nn.Linear(1088, 3088)
    def forward(self, point_cloud, l=None):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))# （B,64,N）
        out2 = F.relu(self.bn2(self.conv2(out1)))# （B,128,N）
        out3 = F.relu(self.bn3(self.conv3(out2)))# （B,128,N）

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))#（B,512,N）
        out5 = self.bn5(self.conv5(out4))#（B,1024,N）
        out_max = torch.max(out5, 2, keepdim=True)[0]#（B,1024,1）
        if self.pretrain:
            # print("Pretrain")
            out_max = out_max.squeeze()#（B,1024）
            inv_feat = self.inv_head(out_max)#（B,256）

            return out_max, inv_feat, out_max
        #
        # out_max = out_max.view(-1, 2048)
        # # out_max = self.conv(out_max)
        # out_max = self.l(out_max)
        # if self.pretrain:
        #     return out5,out_max,out5
        else:
            # l = l.view(B, -1, 1) # (batch_size, num_categoties, 1)
            # l = self.conv7(l)
            out_max = out_max.view(-1, 1024)
            out_max = torch.cat([out_max,l.squeeze(1)],1)
            # out_max = torch.cat([out_max, l], 1)
            # out_max = torch.cat([out_max], 1)
            expand = out_max.view(-1, 1024+16, 1).repeat(1, 1, N)
            # expand = out_max.repeat(1,1,N)
            # expand = F.relu(self.bns5(self.convs5(expand)))
            # expand = expand.transpose(1,2)
            # expand = self.l5(expand)
            # expand = expand.transpose(1,2)
            concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
            net = F.relu(self.bns1(self.convs1(concat)))
            net = F.relu(self.bns2(self.convs2(net)))
            net = F.relu(self.bns3(self.convs3(net)))
            net = self.convs4(net)
            net = net.transpose(2, 1).contiguous()
            net = F.log_softmax(net.view(-1, self.seg_num_all), dim=-1)
            net = net.view(B, N, self.seg_num_all) # [B, N, 50]
            net = net.transpose(1,2)
            return net

class pointnetplus(nn.Module):
    def __init__(self,num_class=40,normal_channel=True):
        super(pointnetplus, self).__init__()
        in_channel = 3 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        return x, l3_points


class pointnetplus_seg(nn.Module):
    def __init__(self,args, seg_num_all=None, pretrain = True,normal_channel=False):
        super(pointnetplus_seg, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k1
        self.pretrain = pretrain
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, seg_num_all, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points

class GCN(nn.Module):
    def __init__(self, pool='SAG', ratio=0.5, class_num=10):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 16, add_self_loops = True, normalize = True)
        self.conv2 = GCNConv(16, 32, add_self_loops = True, normalize = True)
        self.conv3 = GCNConv(32, 64, add_self_loops = True, normalize = True)
        if pool == 'SAG':
            self.pool1 = SAGPooling(in_channels=32, ratio=ratio)
            self.pool2 = SAGPooling(in_channels=64, ratio=ratio)
        else:
            self.pool1 = ASAPooling(in_channels=32, ratio=ratio)
            self.pool2 = ASAPooling(in_channels=64, ratio=ratio)
        self.conv4 = GCNConv(64, 32, add_self_loops = True, normalize = True)
        self.conv5 = GCNConv(32, 32, add_self_loops = True, normalize = True)
        self.conv6 = GCNConv(32, class_num, add_self_loops = True, normalize = True)

    def forward(self, x, edge_index, edge_attr):
        x = x.view(-1, 3)
        edge_index = edge_index.view(2, -1)
        edge_attr = edge_attr.view(-1)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        temp = self.pool1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = temp[0], temp[1], temp[2]

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        temp = self.pool2(x, edge_index, edge_attr)
        x, edge_index, edge_attr = temp[0], temp[1], temp[2]

        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.conv5(x, edge_index, edge_attr)
        x = F.relu(x)

        # x = self.conv6(x, edge_index, edge_attr)
        # x = F.relu(x)

        # x = torch.max(x, dim=0, keepdim=True)[0]
        return x
        # return F.log_softmax(x, dim=1)


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all=None, pretrain=True):
    # def __init__(self, args):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k1
        self.pretrain = pretrain
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.inv_head = nn.Sequential(
                            nn.Linear(args.emb_dims, args.emb_dims),
                            nn.BatchNorm1d(args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.emb_dims, 256)
                            )
        
        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=args.dropout)
            self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       self.bn9,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp2 = nn.Dropout(p=args.dropout)
            self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                       self.bn10,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l = None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        
        if self.pretrain:
            # print("Pretrain")
            x = x.squeeze()
            inv_feat = self.inv_head(x)
            
            return x, inv_feat, x
        
        else:
            l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
            l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

            x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
            x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

            x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

            x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
            x = self.dp1(x)
            x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
            x = self.dp2(x)
            x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
            x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
            
            return x
        