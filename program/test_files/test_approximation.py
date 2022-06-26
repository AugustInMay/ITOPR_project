import adapter
import numpy as np
import saver
import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

tmp = ("pressure", "vel_x", "vel_y", "smoke")
errors = [[],[],[],[],[]]

print("Запускаю тестирование аппроксимации...")

for j in range(5):
    original = []
    counted = []

    for i in range(4):
        orig = saver.read_np_f("./noised_files/" + tmp[i] + str(j) + "_orig")
        original.append(orig)
        
        to_count = saver.read_np_f("./noised_files/" + tmp[i] + str(j) + "_noised")
        adapter.count_p(to_count)
        
        counted.append(to_count)

        errors[i].append(np.sum(np.abs(orig - to_count))/np.sum(np.abs(orig)))
        progress(j*4 + i + 1, 20)

    original = np.array(original)
    counted = np.array(counted)
    errors[4].append(np.sum(np.abs(original - counted))/np.sum(np.abs(original)))

print("\nГотово!")
print("Средняя ошибка по концентрации: %2f %%" % (np.mean(errors[0])*100), sep='')
print("Средняя ошибка по скорости по координате X: %2f %%" % (np.mean(errors[1])*100), sep='')
print("Средняя ошибка по скорости по координате Y: %2f %%" % (np.mean(errors[2])*100), sep='')
print("Средняя ошибка по давлению: %2f %%" % (np.mean(errors[3])*100), sep='')
print("Средняя общая ошибка: %2f %%" % (np.mean(errors[4])*100), sep='')
