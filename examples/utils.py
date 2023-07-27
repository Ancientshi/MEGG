'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import numpy as np
import scipy.stats as stats
import os
import sys
import time
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func



def init_forget_stats(num_train_examples):
  forget_stats = SimpleNamespace()
  forget_stats.prev_accs = np.zeros(num_train_examples, dtype=np.int32)
  forget_stats.num_forgets = np.zeros(num_train_examples, dtype=float)
  forget_stats.never_correct = np.arange(num_train_examples, dtype=np.int32)
  return forget_stats


def update_forget_stats(forget_stats, idxs, accs):
  forget_stats.num_forgets[idxs[forget_stats.prev_accs[idxs] > accs]] += 1
  forget_stats.prev_accs[idxs] = accs
  forget_stats.never_correct = np.setdiff1d(forget_stats.never_correct, idxs[accs.astype(bool)], True)
  return forget_stats


def save_forget_scores(save_dir, dirName,epoch,forget_stats):
  forget_scores = forget_stats.num_forgets.copy()
  forget_scores[forget_stats.never_correct] = 1000
  if not os.path.exists(os.path.join(save_dir, dirName)):
    os.makedirs(os.path.join(save_dir, dirName))
  forget_scores_dict=dict(zip(list(range(1,len(forget_scores)+1)),forget_scores))
  np.save(os.path.join(save_dir, dirName,'ForgettingScore_%s.npy'%epoch), forget_scores_dict)


def load_forget_scores(load_dir, ckpt):
  return np.load(load_dir + f'/forget_scores/ckpt_{ckpt}.npy')

class EL2N(nn.Module):
    def __init__(self,reduction='mean',label_num=10):
        super(EL2N, self).__init__()
        self.reduction=reduction
        self.label_num=label_num
        return

    def forward(self, outputs,targets):
        onehot_targets=torch.nn.functional.one_hot(targets, num_classes=self.label_num) 
        outputs=torch.softmax(outputs, dim=1)
        loss=torch.norm(outputs-onehot_targets, p=2, dim=1)
        if  self.reduction=='mean':
            loss=torch.mean(loss,dim =0)
        elif self.reduction=='none':
            pass
        return loss

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            try:
                init.constant_(m.bias, 0)

            except:
                pass
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            try:
                init.constant_(m.bias, 0)
            except:
                pass
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            try:
                init.constant_(m.bias, 0)
            except:
                pass


term_width = 600

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f








