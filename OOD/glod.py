import torch
import torch.nn as nn
import numpy as np
import os


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.centers = nn.Parameter(0.5*torch.randn(n_classes, input_dim))
        self.covs = nn.Parameter(0.2+torch.tensor(np.random.exponential(scale=0.3, size=(n_classes, input_dim))))

    def forward(self, x):
        covs = self.covs.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        centers = self.centers.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        diff = x.unsqueeze(1).repeat(1, self.n_classes, 1) - centers

        Z_log = (-0.5*torch.sum(torch.log(self.covs +
                                          np.finfo(np.float32).eps), axis=-1)
                 - 0.5*self.input_dim*np.log(2*np.pi))
        exp_log = -0.5 * \
            torch.sum(diff*(1/(covs+np.finfo(np.float32).eps))*diff, axis=-1)
        likelihood = Z_log+exp_log
        return likelihood

    def clip_convs(self):
        '''
        Cliping the convariance matricies to be alaways positive. \n
        Use: call after optimizer.step()
        '''
        with torch.no_grad():
            self.covs.clamp_(np.finfo(np.float32).eps)

    def cov_regulaizer(self, beta=0.01):
        '''
        Covarianvce regulzer \n
        Use: add to the loss if used for OOD detection
        '''
        return beta*(torch.norm(self.covs, p=2))


# class GlodLoss(nn.Module):
#     def __init__(self, lambd=0.003):
#         super(GlodLoss, self).__init__()
#         self.lambd = lambd
#         self.cross_entropy = nn.CrossEntropyLoss()

#     def forward(self, x, y):
#         ce = self.cross_entropy(x, y)
#         likelihood = -x.gather(1, y.unsqueeze(1))
#         return ce+(self.lambd/x.size(0))*likelihood.sum()


def retrieve_scores(model, loader, device, k, saved_in_scores, saved_out_scores, out=False):
    # saved_out_scores = './OOD_scores/{}_out_scores_{}_preds.pt'.format(
    #     model_type, data_type)
    # saved_in_scores = './OOD_scores/{}_in_scores_{}_preds.pt'.format(
    #     model_type, data_type)
    if out:
        if os.path.exists(saved_out_scores):
            preds = torch.load(saved_out_scores)
        else:
            preds = predict(model, loader, device)
        # preds = predict(model, loader, device)
    else:
        if os.path.exists(saved_in_scores):
            preds = torch.load(saved_in_scores)
        else:
            preds = predict(model, loader, device)
        # preds = predict(model, loader, device)
    if out and not os.path.exists(saved_out_scores):
        torch.save(preds, saved_out_scores)
        print('Save to: '+saved_out_scores)
    elif not out and not os.path.exists(saved_in_scores):
        torch.save(preds, saved_in_scores)
        print('Save to: '+saved_in_scores)
    top_k = preds.topk(k).values.squeeze()
    avg_ll = np.mean(top_k[:, 1:k].cpu().detach().numpy())
    llr = top_k[:, 0].cpu()-avg_ll
    return llr


class ConvertToGlod(nn.Module):
    def __init__(self, net, num_classes=100, input_dim=2048):
        super(ConvertToGlod, self).__init__()
        self.gaussian_layer = GaussianLayer(input_dim=input_dim, n_classes=num_classes)
        # self.net = nn.Sequential(*list(net.children())[:-1])
        self.net = nn.Sequential(*list(*net.children())[:-2])

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        # print('out before gaussian: ',out)
        out = self.gaussian_layer(out)
        # print('\nout after gaussian: ',out)
        return out

    def penultimate_forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


def calc_gaussian_params(model, loader, device, n_classes):
    outputs_list = []
    target_list = []
    with torch.no_grad():
        # for (inputs, targets) in tqdm(loader):
        (inputs, targets) = loader
        inputs, targets = torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
        outputs = model.penultimate_forward(inputs)
        # print(outputs.size())
        outputs_list.append(outputs)
        target_list.append(targets)
        outputs = torch.cat(outputs_list, axis=0)
        target_list = torch.cat(target_list)
        x_dim = outputs.size(1)
        centers = torch.zeros(n_classes, x_dim).to(device)
        covs = 0.01*torch.ones(n_classes, x_dim).to(device)
        for c in range(n_classes):
            class_points = outputs[c == target_list].clone()
            if class_points.size(0) <= 1:
                continue
            centers[c] = torch.mean(class_points, axis=0)
            covs[c] = torch.var(class_points, axis=0)
        return covs, centers


def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        # for batch_idx, (inputs, _) in enumerate(loader):
        # for inputs,_ in tqdm(loader):
        inputs = loader
        inputs = torch.tensor(inputs,device=device)
        outputs = model(inputs)
        predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions
