import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from OOD.glod import ConvertToGlod, calc_gaussian_params,retrieve_scores
from SAC_evaluation import convert_to_glod,ood_scores
from OppModeling.utils import colors
from OppModeling.SAC import MLPActorCritic
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.hid = 256
args.l = 2
args.cuda = False
obs_dim = 143
act_dim = 56
ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
global_ac.load_state_dict(torch.load("experiments/ReiwaThunder/ReiwaThunder_1.torch"))
device = torch.device("cuda") if args.cuda else torch.device("cpu")

(glod_input, glod_target) = torch.load("experiments/ReiwaThunder/evaluation/GLOD_SCORES")
(in_p2, p2_list) = torch.load("experiments/ReiwaThunder/evaluation/OPP_INFO")
uncertainties = torch.load("experiments/ReiwaThunder/evaluation/SOFTMAX_SCORE")


plt.hist(uncertainties, bins=200, histtype='bar', color=colors[:len(uncertainties)], label=p2_list, alpha=0.5,
         rwidth=0.8)
plt.legend(prop={'size': 10})
plt.title(in_p2 + " softmax")
plt.ylabel("counts")
plt.xlim(right=0.1, left=0)
plt.xlabel("1 - MaxSoftmax")
plt.savefig(os.path.join("experiments/ReiwaThunder/evaluation", '{}.pdf'.format("Softmax")))
plt.show()
plt.clf()

# Draw the GLOD plot
train_input = glod_input[in_p2]
train_target = glod_target[in_p2]
glod_train = (train_input, train_target)
model = convert_to_glod(model=global_ac.pi, hidden_dim=args.hid, act_dim=act_dim, train_loader=glod_train,device=device)
for k in range(2, 56):
    scores = [retrieve_scores(model, glod_input[name], device, k).detach().cpu().numpy() for name in p2_list]
    plt.hist(scores, bins=500, histtype='bar', color=colors[:len(scores)], label=p2_list, alpha=0.5, rwidth=0.8)
    plt.legend(prop={'size': 10})
    plt.title(in_p2 + " GLOD")
    plt.ylabel("counts")
    plt.xlabel("GLOD SCORE")
    plt.savefig(os.path.join("experiments/ReiwaThunder/evaluation", '{}_{}.pdf'.format("GLOD",k)))
    plt.clf()
# plt.show()


# To Draw the GLOD-MaxSoftmax

k = 15
for index, p2 in enumerate(glod_input.keys()):
    temp = glod_input[p2]
    temp = torch.tensor(temp)
    a_prob, log_a_prob, sample_a, max_a, = global_ac.get_actions_info(global_ac.pi(temp))
    uncertainty = ood_scores(a_prob)
    scores= retrieve_scores(model, glod_input[p2], device, k).detach().cpu().numpy()
    plt.scatter(x=scores, y=uncertainty.detach().numpy(), alpha=0.5, c=colors[index],label= str(p2))
plt.savefig(os.path.join("experiments/ReiwaThunder/evaluation", '{}.pdf'.format("GLOD_SOFT")))

