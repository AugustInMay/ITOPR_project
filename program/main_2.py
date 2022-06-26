import os,sys,math
from unittest import TestLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
from DfpNet import UNet_, weights_init
import saver 

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
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float().to(device), inputs_cpu.float().to(device)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        # precentage loss by ratio of means which is same as the ratio of the sum
        lossPer_s = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0]))/np.sum(np.abs(targets_cpu[0]))
        lossPer_v_x = np.sum(np.abs(outputs_cpu[1] - targets_cpu[1]))/np.sum(np.abs(targets_cpu[1]))
        lossPer_v_y = np.sum(np.abs(outputs_cpu[2] - targets_cpu[2]))/np.sum(np.abs(targets_cpu[2]))
        #lossPer_v = ( np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) ) / ( np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])) )
        lossPer_p = np.sum(np.abs(outputs_cpu[3] - targets_cpu[3]))/np.sum(np.abs(targets_cpu[3]))

        lossPer = np.sum(np.abs(outputs_cpu - targets_cpu))/np.sum(np.abs(targets_cpu))
        lossPer_s_accum += lossPer_s.item()
        #lossPer_v_accum += lossPer_v.item()
        lossPer_p_accum += lossPer_p.item()
        lossPer_accum += lossPer.item()
        
        print(str(i), "s", lossPer_s.item()*100, "v_x", lossPer_v_x.item()*100, "v_y", lossPer_v_y.item()*100,  "p", lossPer_p.item()*100, "overall", lossPer.item()*100)

        tmp = ("smoke", "vel_x", "vel_y", "pres")
        for j in range(4):
            saver.save_np_scaled_img(inputs_cpu[i], "./pics/" + tmp[j] + str(i) + "_orig")
            saver.save_np_scaled_img(targets_cpu[i], "./pics/" + tmp[j] + str(i) + "_target")
            saver.save_np_scaled_img(outputs_cpu[i], "./pics/" + tmp[j] + str(i) + "_next")

        # denormalized error 
        outputs_tmp = np.array([outputs_cpu])
        outputs_tmp = torch.from_numpy(outputs_tmp)
        targets_tmp = np.array([targets_cpu])
        targets_tmp = torch.from_numpy(targets_tmp)
        
        outputs_dn.data.resize_as_(outputs_tmp).copy_(outputs_tmp)
        targets_dn.data.resize_as_(targets_tmp).copy_(targets_tmp)

        lossL1_dn = criterionL1(outputs_dn, targets_dn)
        L1val_dn_accum += lossL1_dn.item()
        progress(i+1, len(testLoader))