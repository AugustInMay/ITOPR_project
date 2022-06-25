import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from DfpNet import UNet_, DefNet_, weights_init
import dataset
import utils
from utils import log
######## Settings ########

# number of training iterations
iterations = 10000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.001
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 7
# data set config
prop=None # by default, use all from "../data/train"
# save txt files with per epoch loss?
saveL1 = True

##########################

prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout    = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad     = ""      # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True # warning, slower

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(dataDir = "../data_3_2/", ratio=0.1)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=1, shuffle=False, drop_last=True) 
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations/len(trainLoader) + 0.5)
netG = UNet_(channelExponent=expo, dropout=dropout)
print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad)>0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lrG, weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 4, 100, 100))
inputs  = Variable(torch.FloatTensor(batch_size, 4, 100, 100))
targets = targets.cuda()
inputs  = inputs.cuda()

##########################

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))

    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()

        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i==len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)


    # validation
    # netG.eval()
    # L1val_accum = 0.0
    # for i, validata in enumerate(valiLoader, 0):
    #     inputs_cpu, targets_cpu = validata
    #     targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
    #     inputs.resize_as_(inputs_cpu).copy_(inputs_cpu)
    #     targets.resize_as_(targets_cpu).copy_(targets_cpu)

    #     outputs = netG(inputs)
    #     outputs_cpu = outputs.data.cpu().numpy()

    #     lossL1 = criterionL1(outputs, targets)
    #     L1val_accum += lossL1.item()

    #     if i==0:
    #         input_ndarray = inputs_cpu.cpu().numpy()[0]
    #         v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5
            
    #         utils.makeDirs(["results_train"])
    #         utils.imageOut("results_train/epoch{}_{}".format(epoch, i), outputs_cpu[0], targets_cpu.cpu().numpy()[0], saveTargets=True)

    # # data for graph plotting
    # L1_accum    /= len(trainLoader)
    # L1val_accum /= len(valiLoader)
    # if saveL1:
    #     if epoch==0: 
    #         utils.resetLog(prefix + "L1.txt"   )
    #         utils.resetLog(prefix + "L1val.txt")
    #     utils.log(prefix + "L1.txt"   , "{} ".format(L1_accum), False)
    #     utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)

suffix = "" # customize loading & output if necessary
prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

torch.save(netG.state_dict(), prefix + "modelG" )

testLoader = valiLoader
targets = torch.FloatTensor(1, 4, 100, 100)
targets = Variable(targets)
targets = targets.cuda()
inputs = torch.FloatTensor(1, 4, 100, 100)
inputs = Variable(inputs)
inputs = inputs.cuda()

targets_dn = torch.FloatTensor(1, 4, 100, 100)
targets_dn = Variable(targets_dn)
targets_dn = targets_dn.cuda()
outputs_dn = torch.FloatTensor(1, 4, 100, 100)
outputs_dn = Variable(outputs_dn)
outputs_dn = targets_dn.cuda()

lf = "./" + prefix + "testout{}.txt".format(suffix) 
utils.makeDirs(["results_test"])

# loop over different trained models
avgLoss = 0.
losses = []
models = []

for si in range(25):
    s = chr(96+si)
    if(si==0): 
        s = "" # check modelG, and modelG + char
    modelFn = "./" + prefix + "modelG{}{}".format(suffix,s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    log(lf, "Loading " + modelFn )
    netG.load_state_dict( torch.load(modelFn) )
    log(lf, "Loaded " + modelFn )
    netG.cuda()

    criterionL1 = nn.L1Loss()
    criterionL1.cuda()
    L1val_accum = 0.0
    L1val_dn_accum = 0.0
    lossPer_s_accum = 0
    lossPer_v_accum = 0
    lossPer_p_accum = 0
    lossPer_accum = 0

    netG.eval()

    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        with torch.no_grad():
            inputs.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        # precentage loss by ratio of means which is same as the ratio of the sum
        lossPer_s = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0]))/np.sum(np.abs(targets_cpu[0]))
        lossPer_v = ( np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) ) / ( np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])) )
        lossPer_p = np.sum(np.abs(outputs_cpu[3] - targets_cpu[3]))/np.sum(np.abs(targets_cpu[3]))

        lossPer = np.sum(np.abs(outputs_cpu - targets_cpu))/np.sum(np.abs(targets_cpu))
        lossPer_s_accum += lossPer_s.item()
        lossPer_v_accum += lossPer_v.item()
        lossPer_p_accum += lossPer_p.item()
        lossPer_accum += lossPer.item()

        log(lf, "Test sample %d"% i )
        log(lf, "    smoke:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_s.item()) )
        log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) , lossPer_v.item() ) )
        log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[3] - targets_cpu[3])), lossPer_p.item()) )
        log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu    - targets_cpu   )), lossPer.item()) )

        # denormalized error 
        outputs_tmp = np.array([outputs_cpu])
        outputs_tmp = torch.from_numpy(outputs_tmp)
        targets_tmp = np.array([targets_cpu])
        targets_tmp = torch.from_numpy(targets_tmp)
        
        outputs_dn.data.resize_as_(outputs_tmp).copy_(outputs_tmp)
        targets_dn.data.resize_as_(targets_tmp).copy_(targets_tmp)

        lossL1_dn = criterionL1(outputs_dn, targets_dn)
        L1val_dn_accum += lossL1_dn.item()

        # write output image, note - this is currently overwritten for multiple models
        os.chdir("./results_test/")
        utils.imageOut("%04d"%(i), outputs_cpu, targets_cpu, normalize=False, saveMontage=True) # write normalized with error
        os.chdir("../")

    log(lf, "\n") 
    L1val_accum     /= len(testLoader)
    lossPer_p_accum /= len(testLoader)
    lossPer_v_accum /= len(testLoader)
    lossPer_s_accum /= len(testLoader)
    lossPer_accum   /= len(testLoader)
    L1val_dn_accum  /= len(testLoader)
    log(lf, "Loss percentage (p, v, combined): %f %%    %f %%    %f %%    %f %% " % (lossPer_p_accum*100, lossPer_v_accum*100, lossPer_s_accum*100, lossPer_accum*100 ) )
    log(lf, "L1 error: %f" % (L1val_accum) )
    log(lf, "Denormalized error: %f" % (L1val_dn_accum) )
    log(lf, "\n") 

    avgLoss += lossPer_accum
    losses.append(lossPer_accum)

if len(losses)>1:
	avgLoss /= len(losses)
	lossStdErr = np.std(losses) / math.sqrt(len(losses))
	log(lf, "Averaged relative error and std dev across models:   %f , %f " % (avgLoss,lossStdErr) )
    