import os,sys,math
from unittest import TestLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import adapter
import saver
from dataset import TurbDataset
from DfpNet import UNet_, weights_init
import time 

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

suffix = "" # customize loading & output if necessary
prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

expo = 7
dataset = TurbDataset(mode=TurbDataset.TEST, dataDirTest="./files/")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

targets = torch.FloatTensor(1, 4, 100, 100)
targets = Variable(targets)
targets = targets.to(device)
inputs = torch.FloatTensor(1, 4, 100, 100)
inputs = Variable(inputs)
inputs = inputs.to(device)

targets_dn = torch.FloatTensor(1, 4, 100, 100)
targets_dn = Variable(targets_dn)
targets_dn = targets_dn.to(device)
outputs_dn = torch.FloatTensor(1, 4, 100, 100)
outputs_dn = Variable(outputs_dn)
outputs_dn = targets_dn.to(device)

netG = UNet_(channelExponent=expo)

# loop over different trained models
avgLoss = 0.
losses = []
models = []

print("Запускаю тестирование прогнозирования...")
pred_max_time = 0

for si in range(25):
    s = chr(96+si)
    if(si==0): 
        s = "" # check modelG, and modelG + char
    modelFn = "./" + prefix + "modelG{}{}".format(suffix,s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    netG.load_state_dict(  torch.load(modelFn, map_location=device) )
    #netG.cuda()

    criterionL1 = nn.L1Loss()
    criterionL1.to(device)
    L1val_accum = 0.0
    L1val_dn_accum = 0.0
    lossPer_s_accum = 0
    lossPer_v_accum = 0
    lossPer_p_accum = 0
    lossPer_accum = 0

    netG.eval()

    for i, data in enumerate(testLoader, 0):
        start_ = time.time()
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float().to(device), inputs_cpu.float().to(device)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]
        end_ = time.time()

        if pred_max_time < (end_ - start_):
            pred_max_time = end_ - start_
        # precentage loss by ratio of means which is same as the ratio of the sum

        # denormalized error 
        progress(i+1, len(testLoader))


tmp = ("pressure", "vel_x", "vel_y", "smoke")
errors = [[],[],[],[],[]]
print("\nГотово!")

print("Запускаю тестирование аппроксимации...")

approx_max_time = 0
for j in range(5):
    start_ = time.time()

    for i in range(4):       
        to_count = saver.read_np_f("./test_files/noised_files/" + tmp[i] + str(j) + "_noised")
        adapter.count_p(to_count)
        progress(j*4 + i+1, 20)

    end_ = time.time()

    if approx_max_time < (end_ - start_):
            approx_max_time = end_ - start_


print("\nГотово!")

            
print("Максимальное время прогнозирования составило %.1f секунд(ы)" %(pred_max_time))
print("Максимальное время аппроксимации составило %.1f секунд(ы)" %(approx_max_time))