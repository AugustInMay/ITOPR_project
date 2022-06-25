from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random

# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = True
# global switch, make data dimensionless?
makeDimLess = True
# global switch, remove constant offsets from pressure channel?
removePOffset = True

## helper - compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


######################################## DATA LOADER #########################################
#         also normalizes data with max , and optionally makes it dimensionless              #

def LoaderNormalizer(data, isTest = False, shuffle = 0, ratio = 0.8):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    # dataProp: proportions for loading & mixing 3 different data directories "reg", "shear", "sup"
    #           should be array with [total-length, fraction-regular, fraction-superimposed, fraction-sheared],
    #           passing None means off, then loads from single directory
    """

    # load single directory
    # if ratio==0.8:
    #     data.inputs  = np.empty((400, 4, 100, 100))
    #     data.targets = np.empty((400, 4, 100, 100))
        
    #     count = 0

    #     for i in range(1,5):
    #         for j in range(100):
    #             sm = np.load(data.dataDir + "smoke" + str(i) + "_" + str(j) + ".npy")
    #             vel_x = np.load(data.dataDir + "vel_x" + str(i)+ "_" + str(j) + ".npy")
    #             vel_y = np.load(data.dataDir + "vel_y" + str(i)+ "_" + str(j) + ".npy")
    #             pres = np.load(data.dataDir + "pressure" + str(i)+ "_" + str(j) + ".npy")
    #             data.inputs[count] = np.array([sm, vel_x, vel_y, pres])

    #             sm = np.load(data.dataDir + "smoke" + str(i)+ "_" + str(j) + "_target.npy")
    #             vel_x = np.load(data.dataDir + "vel_x" + str(i)+ "_" + str(j) + "_target.npy")
    #             vel_y = np.load(data.dataDir + "vel_y" + str(i)+ "_" + str(j) + "_target.npy")
    #             pres = np.load(data.dataDir + "pressure" + str(i)+ "_" + str(j) + "_target.npy")
    #             data.targets[count] = np.array([sm, vel_x, vel_y, pres])

    #             count += 1

    # else:
    data.inputs  = np.empty((500, 4, 100, 100))
    data.targets = np.empty((500, 4, 100, 100))
    count = 0

    first_range = random.sample(range(1,6), k=5)
    second_range = random.sample(range(100), k=100)

    for i in first_range:
        second_range = random.sample(range(100), k=100)
        for j in second_range:
            sm = np.load(data.dataDir + "smoke" + str(i) + "_" + str(j) + ".npy")
            vel_x = np.load(data.dataDir + "vel_x" + str(i)+ "_" + str(j) + ".npy")
            vel_y = np.load(data.dataDir + "vel_y" + str(i)+ "_" + str(j) + ".npy")
            pres = np.load(data.dataDir + "pressure" + str(i)+ "_" + str(j) + ".npy")
            data.inputs[count] = np.array([sm, vel_x, vel_y, pres])
            
            if data.dataDir == "../data_3_1/":
                sm = np.load(data.dataDir + "smoke_target" + str(i)+ "_" + str(j) + ".npy")
                vel_x = np.load(data.dataDir + "vel_x_target" + str(i)+ "_" + str(j) + ".npy")
                vel_y = np.load(data.dataDir + "vel_y_target" + str(i)+ "_" + str(j) + ".npy")
                pres = np.load(data.dataDir + "pressure_target" + str(i)+ "_" + str(j) + ".npy")
            else:
                sm = np.load(data.dataDir + "smoke" + str(i)+ "_" + str(j) + "_target.npy")
                vel_x = np.load(data.dataDir + "vel_x" + str(i)+ "_" + str(j) + "_target.npy")
                vel_y = np.load(data.dataDir + "vel_y" + str(i)+ "_" + str(j) + "_target.npy")
                pres = np.load(data.dataDir + "pressure" + str(i)+ "_" + str(j) + "_target.npy")

            data.targets[count] = np.array([sm, vel_x, vel_y, pres])

            count += 1

            
    ###################################### NORMALIZATION  OF TEST DATA #############################################

    if isTest:
        data.inputs  = np.empty((100, 4, 100, 100))
        data.targets = np.empty((100, 4, 100, 100))

        count = 0

        for i in range(5,6):
            for j in range(100):
                sm = np.load(data.dataDirTest + "smoke" + str(i) + "_" + str(j) + ".npy")
                vel_x = np.load(data.dataDirTest + "vel_x" + str(i)+ "_" + str(j) + ".npy")
                vel_y = np.load(data.dataDirTest + "vel_y" + str(i)+ "_" + str(j) + ".npy")
                pres = np.load(data.dataDirTest + "pressure" + str(i)+ "_" + str(j) + ".npy")
                data.inputs[count] = np.array([sm, vel_x, vel_y, pres])

                sm = np.load(data.dataDirTest + "smoke" + str(i)+ "_" + str(j) + "_target.npy")
                vel_x = np.load(data.dataDirTest + "vel_x" + str(i)+ "_" + str(j) + "_target.npy")
                vel_y = np.load(data.dataDirTest + "vel_y" + str(i)+ "_" + str(j) + "_target.npy")
                pres = np.load(data.dataDirTest + "pressure" + str(i)+ "_" + str(j) + "_target.npy")
                data.targets[count] = np.array([sm, vel_x, vel_y, pres])

                count += 1

    print("Number of data loaded:", count)
    print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % ( 
      np.mean(np.abs(data.targets), keepdims=False), np.max(np.abs(data.targets), keepdims=False) , 
      np.mean(np.abs(data.inputs), keepdims=False) , np.max(np.abs(data.inputs), keepdims=False) ) ) 

    return data

######################################## DATA SET CLASS #########################################

class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, mode=TRAIN, dataDir="../data_2/train/", dataDirTest="../data_2/test/", normMode=0, ratio=0.8):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1:	
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2:	
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST

        # load & normalize data
        self = LoaderNormalizer(self, isTest=(mode==self.TEST), ratio=ratio)
        
        self.totalLength = 500

        if mode == self.TEST:
            self.totalLength = int(500 * (1 - ratio))

        if not self.mode==self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = int(500 * (1 - ratio))

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# simplified validation data set (main one is TurbDataset above)

class ValiDataset(TurbDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

