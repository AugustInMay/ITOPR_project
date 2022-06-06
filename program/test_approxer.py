from phi.torch.flow import *
import pylab
import saver
import adapter
import os
import numpy as np

from DfpNet import UNet_, weights_init

os.environ['KMP_DUPLICATE_LIB_OK']='True'

tmp = ["smoke", "vel_x", "vel_y", "pressure"]

users = [(0, 25), (25, 50), (50, 75), (75, 100)]
print("Enter your number: ", end='')
n = int(input())

for j in range(users[n][0], users[n][1]):
    for i in range(4):
        print("Beginning to approximate...")

        orig = saver.read_np_f("../data_noised/" + tmp[i] + str(j) + "_orig")
        to_count = saver.read_np_f("../data_noised/" + tmp[i] + str(j) + "_noised")
        adapter.count_p(to_count)

        f = open("testout.txt", "a+")
        f.write(str(np.sum(np.abs(orig - to_count))/np.sum(np.abs(to_count))) + "\n")
        f.close()

        print("Approximated ", tmp[i], end = '.\n')
