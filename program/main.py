from phi.torch.flow import *
import pylab
import saver
import adapter
import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from DfpNet import UNet_, weights_init

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def file_choice():
    f_names = ["smoke", "vel_x", "vel_y", "pressure"]

    print("Would you like to continue with default file names? Otherwise enter your custom ones: y/n")
    key = input().lower()
    while key != 'y' and key != 'n':
        print("Wrong input. Enter y or n to choose the option")
        key = input()

    if key == 'n':
        print("Enter smoke density file name:")
        name = input()
        f_names[0] = name

        print("Enter velocity by x file name:")
        name = input()
        f_names[1] = name

        print("Enter velocity by y file name:")
        name = input()
        f_names[2] = name

        print("Enter pressure file name:")
        name = input()
        f_names[3] = name

    return f_names


if __name__ == '__main__':
    files_n = file_choice()
    
    fields = list()
    nan_val = 0

    for el in files_n:
        if not os.path.isfile(el + ".npy"):
            print("'" + el + ".npy' file doesn't exist! Please check your filename again:")
            break

        fields.append(saver.read_np_f(el))
        nan_val += np.count_nonzero(np.isnan(fields[-1]))


    if int(nan_val) == 0:
        print("No noise points were discovered. Continuing with step forward...")

        tmp = ["smoke", "vel_x", "vel_y", "pressure"]

        for i in range(4):
            print("Beginning to approximate...")

            orig = saver.read_np_f(files_n[i])
            adapter.count_p(fields[i])
            print(np.sum(np.abs(orig - fields[i]))/np.sum(np.abs(fields[i])))
            saver.save_np_f(fields[i], files_n[i] + "_app")
            saver.save_np_scaled_img(fields[i], files_n[i] + "_app")

            print("Approximated ", tmp[i], end = '.')

        
        print("Done!")

    else:
        print(str(nan_val) + " noise points were discovered, proceeding with approximation")

        inputs = torch.FloatTensor(1, 4, 100, 100)
        inputs = Variable(inputs)
        
        for i in range(4):
            inputs[0][i] = torch.from_numpy(fields[i])

        netG = UNet_(channelExponent=5)
        modelFn = "./" + "modelG"
        netG.load_state_dict( torch.load(modelFn, map_location=torch.device('cpu')) )

        netG.eval()

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]

        for i in range(4):
            saver.save_np_f(outputs_cpu[i], files_n[i] + "_next")
            saver.save_np_scaled_img(outputs_cpu[i], files_n[i] + "_next")

        print("Done!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
