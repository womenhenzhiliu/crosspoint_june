from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from lightly.loss.ntx_ent_loss import NTXentLoss
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from train_partseg import train_seg,test_seg
import wandb
from datasets.data import ShapeNet, ModelNet40SVM
from models.dgcnn import DGCNN, MERIT, DGCNN_partseg,pointnetplus,pointnet,GCN,pointnet_seg,pointnetplus_seg
from util import IOStream, AverageMeter

# make 'checkpoints/exp_name/models' directory
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def train(args, io):
    
    wandb.init(project="CrossPoint", name=args.exp_name)

    train_loader = DataLoader(ShapeNet(), num_workers=0,
                              batch_size=args.batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        point_model1 = DGCNN(args).to(device)
        merit = MERIT(args,0.8).to(device)
        point_model2 = merit().to(device)
    elif args.model == 'dgcnn_seg':
        point_model1 = DGCNN_partseg(args).to(device)
        merit = MERIT(args, 0.8).to(device)
        point_model2 = merit().to(device)
    elif args.model == 'pointnet':
        point_model1 = pointnet().to(device)
        merit = MERIT(args, 0.8).to(device)
        point_model2 = merit().to(device)
    elif args.model == 'pointnet_seg':
        point_model1 = pointnet_seg(args).to(device)
        merit = MERIT(args,0.8).to(device)
        point_model2 = merit().to(device)
    elif args.model == 'pointnetplus':
        point_model1 = pointnetplus().to(device)
        merit = MERIT(args, 0.8).to(device)
        point_model2 = merit().to(device)
    elif args.model == 'pointnetplus_seg':
        point_model1 = pointnetplus_seg(args).to(device)
        merit = MERIT(args,0.8).to(device)
        point_model2 = merit().to(device)
    else:
        raise Exception("Not implemented")
        

        
    wandb.watch(point_model1,point_model2)
    
    if args.resume:
        point_model1.load_state_dict(torch.load(args.model_path))
        print("Model Loaded !!")
        
    parameters = list(point_model1.parameters())
    # choose the Optimization algorithm
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)
    # 余弦退火学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature = 0.1).to(device)
    
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        # lr_scheduler.step()

        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        train_1_losses = AverageMeter()
        train_2_losses = AverageMeter()
        train_3_losses = AverageMeter()
        train_4_losses = AverageMeter()
        train_5_losses = AverageMeter()
        # train_6_losses = AverageMeter()
        point_model1.train()
        point_model2.train()

        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, ((data_t1, data_t2)) in enumerate(train_loader):
            data_t1, data_t2 = data_t1.to(device), data_t2.to(device)
            batch_size = data_t1.size()[0] #data_t1.shape =  torch.Size([20, 2048, 3])
            # print('data_t1.shape = ',data_t1.shape)
            
            opt.zero_grad()
            data = torch.cat((data_t1, data_t2))
            # print('data.shape',data.shape) #data.shape torch.Size([40, 2048, 3])
            data = data.transpose(2, 1).contiguous()
            _,point_feats1,_  = point_model1(data) #point_feats1.shape torch.Size([40, 256])
            # print('point_feats1.shape',point_feats1.shape)
            with torch.no_grad():
                _,point_feats2, _ = point_model2(data)
            
            point_t1_n1feats = point_feats1[:batch_size, :]
            point_t2_n1feats = point_feats1[batch_size: , :]
            point_t1_n2feats = point_feats2[:batch_size, :]
            point_t2_n2feats = point_feats2[batch_size:, :]

            loss_1 = criterion(point_t1_n1feats, point_t1_n2feats)
            loss_2 = criterion(point_t1_n1feats, point_t2_n1feats)
            loss_3 = criterion(point_t1_n1feats, point_t2_n2feats)
            loss_4 = criterion(point_t2_n1feats, point_t1_n2feats)
            loss_5 = criterion(point_t2_n1feats, point_t2_n2feats)
            # loss_6 = criterion(point_t1_n2feats, point_t2_n2feats)
                
            # total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 +loss_6
            # total_loss = 0.2*( loss_1 + loss_5 ) + 0.5*loss_2 + 0.3*( loss_3 + loss_4 )#first tiaozheng
            total_loss = 0.1 * (loss_1 + loss_5) + 0.7 * loss_2 + 0.2 * (loss_3 + loss_4) #sec tiaozheng
            total_loss.backward()
            opt.step()
            merit.update_ma()
            
            train_losses.update(total_loss.item(), batch_size)
            train_1_losses.update(loss_1.item(), batch_size)
            train_2_losses.update(loss_2.item(), batch_size)
            train_3_losses.update(loss_3.item(), batch_size)
            train_4_losses.update(loss_4.item(), batch_size)
            train_5_losses.update(loss_5.item(), batch_size)
            # train_6_losses.update(loss_6.item(), batch_size)

            
            
            if i % args.print_freq == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f ' % (epoch, i, len(train_loader), train_losses.avg))
        lr_scheduler.step()
        wandb_log['Train Loss'] = train_losses.avg
        wandb_log['Train  Loss1'] = train_1_losses.avg
        wandb_log['Train  Loss2'] = train_2_losses.avg
        wandb_log['Train  Loss3'] = train_3_losses.avg
        wandb_log['Train  Loss4'] = train_4_losses.avg
        wandb_log['Train  Loss5'] = train_5_losses.avg
        # wandb_log['Train  Loss6'] = train_6_losses.avg

        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)  
        
        # Testing
        
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=128, shuffle=True)

        feats_train = []
        labels_train = []
        point_model1.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model1(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model1(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        
        model_tl = SVC(C = 0.1, kernel ='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")
        
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save(point_model1.state_dict(), save_file)
            
            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_model_best.pth')
            torch.save(point_model2.state_dict(), save_img_model_file)
  
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model1.state_dict(), save_file)

        wandb.log(wandb_log)
    
    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(point_model1.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_model_last.pth')
    torch.save(point_model2.state_dict(), save_img_model_file)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg','pointnetplus','pointnet','pointnet_seg','pointnetplus_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k1', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    # parser.add_argument('--model_path', type=str, default='', metavar='N',
    #                     help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

