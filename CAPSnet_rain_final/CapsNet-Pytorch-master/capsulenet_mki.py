"""
Pytorch implementation of CapsNet in paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       Launch `python CapsNet.py -h` for usage help

Result:
    Validation accuracy > 99.6% after 50 epochs.
    Speed: About 73s/epoch on a single GTX1070 GPU card and 43s/epoch on a GTX1080Ti GPU.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import os
from scipy import stats
import pandas as pd

class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        print(input_size)
        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=2, stride=1, padding=1)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=2, stride=2, padding=1)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=512, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.)) #.cuda())
        # print("forward torch.tensor(x).shape")
        # print(torch.tensor(x).shape)
        # print(torch.tensor(y.shape))
        # print(torch.tensor(y[:,:, None]).shape)
        # print(torch.tensor(y[:, None]).shape)
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        #reconstruction = self.decoder((x * y[:, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon


def show_reconstruction(model, test_loader, n_images, args):
    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image
    import numpy as np

    model.eval()
    for x, _ in test_loader:
        x = Variable(x[:min(n_images, x.size(0))], volatile=True) #.cuda(), volatile=True)
        _, x_recon = model(x)
        data = np.concatenate([x.data, x_recon.data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        print('-' * 70)
        plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
        plt.show()
        break


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        #y = torch.zeros(y.size(0), 3).scatter_(1, y.view(-1, 1).long(), 1.)
        #x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).item() * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, test_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.
    for epoch in range(args.epochs):
        print(epoch)
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            #y = torch.zeros(y.size(0), 3).scatter_(1, y.view(-1, 1).long(), 1.)  # change to one-hot coding
            #x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            a = loss.item()
            b = x.size(0)
            training_loss += a * b  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model


def load(input_size, path, batch_size, location):
    
    rain_bins = np.array([0,0.1,4,16,32,64,100])

    fieldname = '_6hr_1.25_1979-2019.pkl'
    mki_arr = pd.read_pickle(path+'mki12'+fieldname) # geopotential height data (5*5 resolution)
    mki_arr = mki_arr[mki_arr.index.month.isin([1,2,3,11,12])]
    mki_arr = mki_arr[~((mki_arr.index.month == 2) & (mki_arr.index.day == 29))]
    mki_arr_norm = stats.zscore(mki_arr).reshape([mki_arr.shape[0]//4,4,5,5]) # normalize
    y_arr = np.load(path+'rain_basin_NDJFM_19790101-20190331.npy') # rain data
    y_arr = np.digitize(y_arr[:,location],rain_bins)-1

    mki_arr_norm = mki_arr_norm[:-61] # remove last NOV-DEC

    tensor_x = torch.Tensor(mki_arr_norm)
    tensor_y = torch.Tensor(y_arr).long()

    forecast_dataset = TensorDataset(tensor_x,tensor_y) # creates a dataset based on tensors
    forecast_dataset2 = []

    for j in forecast_dataset:
        x1 = j[0]
        y1 = j[1]
        if(x1 != 'nan' and y1 != 'nan'):
            classification = torch.zeros(6)
            classification[y1] = 1
            forecast_dataset2.append( ( j[0].reshape(input_size) , classification ) )
    length = int(len(forecast_dataset2)/2)
    training_ds, validation_ds = torch.utils.data.random_split(forecast_dataset2, [length,length])
    training_dataloader = DataLoader(training_ds,batch_size=batch_size,shuffle=True)
    valid_dataloader = DataLoader(validation_ds,batch_size=batch_size)

    return training_dataloader, valid_dataloader

if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_loader, test_loader = load_mnist(args.data_dir, download=False, batch_size=args.batch_size)
    print(train_loader[1])
    # define model
    model = CapsuleNet(input_size=[4, 5, 5], classes=3, routings=3)
    #model.cuda()
    print(model)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))
    if not args.testing:
        train(model, train_loader, test_loader, args)
    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
        print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
        show_reconstruction(model, test_loader, 50, args)