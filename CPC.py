import random
import time
import logging
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from timeit import default_timer as timer


class CDCK2(nn.Module):
    def __init__(self, timestep, obs_dim, latent_dim, c_dim):

        super(CDCK2, self).__init__()
        self.timestep = timestep
        self.c_dim = c_dim
        self.latent_dim = latent_dim
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True)
        # )
        self.encoder = nn.GRU(obs_dim, latent_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.gru = nn.GRU(latent_dim, c_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(c_dim, latent_dim) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, feature_dim, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, feature_dim).cuda()
        else:
            return torch.zeros(1, batch_size, feature_dim)

    def forward(self, x, encoder_hidden, c_hidden):
        batch = x.size()[0]
        seq_len = x.size()[1]
        # no Down sampling in RL
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long()  # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z, encoder_hidden = self.encoder(x, encoder_hidden)
        # Do not need transpose as the input shape is N*L*C
        # z = z.transpose(1, 2)
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.latent_dim)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.latent_dim)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512
        output, c_hidden = self.gru(forward_seq, c_hidden)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, self.c_dim)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.latent_dim)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch

        return accuracy, nce, c_hidden


def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)  # add channel dimension
        optimizer.zero_grad()
        encoder_hidden = model.init_hidden(len(data), args.latent_dim, use_gpu=True)
        c_hidden = model.init_hidden(len(data), args.c_dim, use_gpu=True)
        acc, loss, hidden = model(data,encoder_hidden,c_hidden)

        loss.backward()
        # add gradient clipping
        nn.utils.clip_grad_norm(model.parameters(), 20)
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, acc, loss.item()))


def validation(args, model, device, data_loader, batch_size):
    logging.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device)  # add channel dimension
            hidden = model.init_hidden(len(data), args.c_dim,use_gpu=True)
            acc, loss, hidden = model(data, hidden)
            total_loss += len(data) * loss
            total_acc += len(data) * acc

    total_loss /= len(data_loader.dataset)  # average loss
    total_acc /= len(data_loader.dataset)  # average acc

    logging.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))

    return total_acc, total_loss


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-raw', required=True)
    parser.add_argument('--validation-raw', required=True)
    parser.add_argument('--eval-raw')
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--validation-list', required=True)
    parser.add_argument('--eval-list')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--timestep', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='batch size')
    parser.add_argument('--c_dim', type=int, default=32,
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=20480,
                        help='window length to sample from each utterance')

    parser.add_argument('--masked-frames', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CDCK2(args.timestep, args.batch_size, args.audio_window).to(device)
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logging.info('===> loading train, validation and eval dataset')

    # nanxin optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('### Model summary below###\n {}\n'.format(str(model)))
    logging.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        # trainXXreverse(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        # val_acc, val_loss = validationXXreverse(args, model, device, validation_loader, args.batch_size)
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)

        # Save
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            # TODO add save functions
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        logging.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))

    ## end
    end_global_timer = timer()
    logging.info("################## Success #########################")
    logging.info("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == '__main__':
    ############ Control Center and Hyperparameter ###############
    run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
    print(run_name)
    main()
