#from phi.torch.flow import *

import saver
import adapter
import os
import numpy as np

import time 
import torch
from torch.autograd import Variable
from DfpNet import UNet_

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def file_choice():
    f_names = ["smoke", "vel_x", "vel_y", "pressure"]

    print("Вы желаете продолжить работу с именами файлов по-умолчанию? Иначе введите произвольные: y/n")
    key = input().lower()
    while key != 'y' and key != 'n':
        print("Неправильный ввод. Введите латинские буквы 'y' или 'n', означающие yes (да) и no (нет) соответсвенно: ")
        key = input()

    if key == 'n':
        print("Введите имя файла с концентрацией жидкости:")
        name = input()
        f_names[0] = name

        print("Введите имя файла со скоростью жидкости по координате X:")
        name = input()
        f_names[1] = name

        print("Введите имя файла со скоростью жидкости по координате Y:")
        name = input()
        f_names[2] = name

        print("Введите имя файла с давлением среды:")
        name = input()
        f_names[3] = name

    return f_names


if __name__ == '__main__':
    files_n = file_choice()
    
    fields = list()
    nan_val = 0
    
    go_out = False
    for el in files_n:
        if not os.path.isfile(el + ".npy"):
            print("Файл '" + el + ".npy' не существует! Пожалуйста, проверьте его наличие или измените имя:")
            go_out = True
        else:
            print("Файл '" + el + ".npy' успешно найден и прочитан!")
            fields.append(saver.read_np_f(el))
            nan_val += np.count_nonzero(np.isnan(fields[-1]))

    if not go_out:
        if int(nan_val) != 0:
            start_ = time.time()

            print(str(nan_val) + " неизвестных точек обнаружено. Начинаю процесс аппроксимации")
            
            tmp = ["smoke", "vel_x", "vel_y", "pressure"]

            for i in range(4):
                print("Начало аппроксимации...")

                orig = saver.read_np_f(files_n[i])
                adapter.count_p(fields[i])
                saver.save_np_f(fields[i], files_n[i] + "_app")
                saver.save_np_scaled_img(fields[i], files_n[i] + "_app")

                print("Аппроксимировано  ", tmp[i], end = '.\n')

            end_ = time.time()

            print("Готово! Процесс занял %.1f секунд(ы)" %(end_-start_))

        else:
            print("Не было обнаружено неизвестных точек. Начинаю процесс прогнозирования...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            start_ = time.time()
            
            inputs = torch.FloatTensor(1, 4, 100, 100)
            inputs = Variable(inputs)
            inputs=inputs.to(device)

            for i in range(4):
                inputs[0][i] = torch.from_numpy(fields[i])

            netG = UNet_(channelExponent=7)
            modelFn = "./" + "modelG"
            netG.load_state_dict( torch.load(modelFn, map_location=device) )

            netG.eval()

            outputs = netG(inputs)
            outputs_cpu = outputs.data.cpu().numpy()[0]

            for i in range(4):
                saver.save_np_f(outputs_cpu[i], files_n[i] + "_next")
                saver.save_np_scaled_img(outputs_cpu[i], files_n[i] + "_next")
            
            end_ = time.time()
            
            print("Готово! Процесс занял %.1f секунд(ы)" %(end_-start_))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
