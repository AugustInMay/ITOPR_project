import random
from phi.torch.flow import *
import pylab
import saver
from pathlib import Path


def step(velocity, smoke, pressure, dt=1.0, buoyancy_factor=1.0, INFLOW = None):
    NU = 0.01

    smoke = advect.semi_lagrangian(smoke, velocity, dt)

    if INFLOW != None:
        smoke += INFLOW

    buoyancy_force = (smoke * (0, buoyancy_factor)).at(velocity)
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + dt * buoyancy_force
    velocity = diffuse.explicit(velocity, NU, dt)
    velocity, pressure = fluid.make_incompressible(velocity)
    return velocity, smoke, pressure

def multi_step(t, bf, INFLOW = None):
    pressure = None
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100])
    velocity = StaggeredGrid(0, extrapolation.ZERO, x=100, y=100, bounds=Box[0:100, 0:100])

    for i in range(t):
        velocity, smoke, pressure = step(velocity, smoke, pressure, buoyancy_factor=bf, INFLOW=INFLOW)

    return velocity, smoke, pressure

#def next_step(vel_x, vel_y, sm, pres):


#input - np.arr
def scatter_noise(data):
    mask = np.zeros(data.shape[0] * data.shape[1], dtype=bool)
    c = random.randint(2000, 5000)
    mask[:c] = True
    np.random.shuffle(mask)
    mask = mask.reshape(data.shape[0], data.shape[1])

    new_data = data
    new_data[mask] = np.nan

    return new_data

#input - np.arr
def save_noise_img(data, name:str):
    new_data = data
    new_data[np.isnan(new_data)] = 0
    saver.save_scaled_img(new_data, name)


def grid_to_val(velocity, smoke, pressure):
    return velocity.values.vector[0], velocity.values.vector[1], smoke.values, pressure.values

def grid_to_np(velocity, smoke, pressure):
    return [velocity.values.vector[0].numpy('y,x'), velocity.values.vector[1].numpy('y,x'), smoke.values.numpy('y,x'), pressure.values.numpy('y,x')]


def generate_data(plot = False, specific_name = ''):
    t = random.randint(50, 100)
    INFLOW_ = CenteredGrid(Sphere(center=(random.randint(10, 90), random.randint(10, 40)), radius=random.randint(5, 10)), extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100]) * round(random.uniform(0.2,1.5), 2)
    b_f = round(random.uniform(1,2), 2)
    velocity, smoke, pressure = multi_step(t, b_f, INFLOW_)

    if plot:
        Path("../pics/").mkdir(parents=True, exist_ok=True)
        vals = grid_to_val(velocity, smoke, pressure)
        names = ["vel_x", "vel_y", "smoke", 'pressure']
        for i in range(4):
            saver.plot_grid_val(vals[i], "../pics/"+names[i] + specific_name)

        del vals

    return velocity, smoke, pressure

def generate_simple_data(plot = False, specific_name = ''):
    t = random.randint(25, 100)
    INFLOW_ = CenteredGrid(Sphere(center=(random.randint(10, 90), random.randint(10, 40)), radius=random.randint(3, 10)), extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100]) * 0.5
    velocity, smoke, pressure = multi_step(t, 1, INFLOW_)

    if plot:
        Path("../pics_2/").mkdir(parents=True, exist_ok=True)
        vals = grid_to_val(velocity, smoke, pressure)
        names = ["vel_x", "vel_y", "smoke", 'pressure']
        for i in range(4):
            saver.plot_grid_val(vals[i], "../pics_2/"+names[i] + specific_name)

        del vals
    
    velocity_2, smoke_2, pressure_2 = step(velocity, smoke, pressure, buoyancy_factor=1, INFLOW=INFLOW_)

    if plot:
        Path("../pics_2/").mkdir(parents=True, exist_ok=True)
        vals = grid_to_val(velocity_2, smoke_2, pressure_2)
        names = ["vel_x", "vel_y", "smoke", 'pressure']
        for i in range(4):
            saver.plot_grid_val(vals[i], "../pics_2/"+names[i] + specific_name + "_target")

        del vals

    return velocity, smoke, pressure, velocity_2, smoke_2, pressure_2

def save_simple_data(plot=False, specific_name = ''):
    Path("../data_2/test").mkdir(parents=True, exist_ok=True)
    generated_data = generate_simple_data(plot, specific_name)
    lst_inp = generated_data[:3]
    lst_targ = generated_data[3:]

    lst_inp_np = grid_to_np(lst_inp[0], lst_inp[1], lst_inp[2])
    lst_targ_np = grid_to_np(lst_targ[0], lst_targ[1], lst_targ[2])
    names = ["vel_x", "vel_y", "smoke", 'pressure']
    
    lst_inp_np[0] = np.transpose(lst_inp_np[0])
    lst_inp_np[0] = np.append(lst_inp_np[0], [lst_inp_np[0][-1]], axis=0)
    lst_inp_np[0] = np.transpose(lst_inp_np[0])
    lst_inp_np[1] = np.append(lst_inp_np[1], [lst_inp_np[1][-1]], axis=0)

    lst_targ_np[0] = np.transpose(lst_targ_np[0])
    lst_targ_np[0] = np.append(lst_targ_np[0], [lst_targ_np[0][-1]], axis=0)
    lst_targ_np[0] = np.transpose(lst_targ_np[0])
    lst_targ_np[1] = np.append(lst_targ_np[1], [lst_targ_np[1][-1]], axis=0)

    for i in range(4):
        saver.save_np_f(lst_inp_np[i], "../data_2/test/" + names[i] + specific_name)
        saver.save_np_f(lst_targ_np[i], "../data_2/test/" + names[i] + specific_name + "_target")

    del lst_inp
    del lst_targ_np

# def save_generated_data_for_train(plot=False, specific_name = ''):
#     Path("../data/train").mkdir(parents=True, exist_ok=True)
#     lst = generate_data(plot, specific_name)
#     names = ["vel", "smoke", 'pressure']

#     for i in range(3):
#         saver.save_tensor_f(lst[i].values, '../data/train' + names[i] + specific_name)

#     del lst
    
# def save_generated_data_for_test(plot=False, specifin_name=''):
#     Path("../data/test/").mkdir(parents=True, exist_ok=True)
#     velocity, smoke, pressure = generate_data(plot, specific_name=specifin_name)

#     lst = grid_to_np(velocity, smoke, pressure)
#     names = ["vel_x", "vel_y", "smoke", 'pressure']
    
#     lst[0] = np.transpose(lst[0])
#     lst[0] = np.append(lst[0], [lst[0][-1]], axis=0)
#     lst[0] = np.transpose(lst[0])
#     lst[1] = np.append(lst[1], [lst[1][-1]], axis=0)

#     for i in range(4):
#         saver.save_np_f(lst[i], "../data/test/" + names[i] + specifin_name)

#     for i in range(2):
#         velocity, smoke, pressure = step(velocity, smoke, pressure)

#     lst = grid_to_np(velocity, smoke, pressure)
#     names = ["vel_x", "vel_y", "smoke", 'pressure']

#     lst[0] = np.transpose(lst[0])
#     lst[0] = np.append(lst[0], [lst[0][-1]], axis=0)
#     lst[0] = np.transpose(lst[0])
#     lst[1] = np.append(lst[1], [lst[1][-1]], axis=0)

#     for i in range(4):
#         saver.save_np_f(lst[i], "../data/test/" + names[i] + "_target" + specifin_name)
#         saver.save_np_scaled_img(lst[i], "../pics/" + names[i] + "_target" + specifin_name)

#     pylab.close('all')

def generate_noised_data(plot = False):
    velocity, smoke, pressure = generate_data(plot)
    lst = grid_to_np(velocity, smoke, pressure)
    lst[0] = np.transpose(lst[0])
    lst[0] = np.append(lst[0], [lst[0][-1]], axis=0)
    lst[0] = np.transpose(lst[0])
    lst[1] = np.append(lst[1], [lst[1][-1]], axis=0)

    names = ["vel_x", "vel_y", "smoke", 'pressure']

    for i in range(4):
        saver.save_np_f(lst[i], names[i])
        if plot:
            saver.save_np_scaled_img(np.where(np.isnan(lst[i]), 0, lst[i]), names[i])
    
    for i in range(2):
        velocity, smoke, pressure = step(velocity, smoke, pressure)

    lst2 = grid_to_np(velocity, smoke, pressure)
    lst2[0] = np.transpose(lst[0])
    lst2[0] = np.append(lst[0], [lst[0][-1]], axis=0)
    lst2[0] = np.transpose(lst[0])
    lst2[1] = np.append(lst[1], [lst[1][-1]], axis=0)
    
    for i in range(4):
        saver.save_np_f(lst2[i], names[i]+"_target")
        if plot:
            saver.save_np_scaled_img(np.where(np.isnan(lst2[i]), 0, lst2[i]), names[i]+"_target")

    for el in lst:
        el = scatter_noise(el)
    
    if plot:
        for i in range(4):
            saver.save_np_scaled_img(np.where(np.isnan(lst[i]), 0, lst[i]), names[i] + "_noise")

    return lst

def save_noised_data(plot=False):
    lst = generate_noised_data(plot)
    names = ["vel_x_noise", "vel_y_noise", "smoke_noise", 'pressure_noise']

    for i in range(4):
        saver.save_np_f(lst[i], names[i])


# def refractor_data():
#     for i in range(2,3):
#         for j in range(100):
#             smoke = saver.read_tensor_f("../data/train/smoke" + str(i) + "_" + str(j))
#             pres = saver.read_tensor_f("../data/train/pressure" + str(i) + "_" + str(j))
#             vel = saver.read_tensor_f("../data/train/vel" + str(i) + "_" + str(j))

#             pressure = CenteredGrid(pres, extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100])
#             smoke = CenteredGrid(smoke, extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100])
#             velocity = StaggeredGrid(vel, extrapolation.ZERO, x=100, y=100, bounds=Box[0:100, 0:100])

#             vel_x = velocity.values.vector[0].numpy('y,x')
#             vel_y = velocity.values.vector[1].numpy('y,x')

#             vel_x = np.transpose(vel_x)
#             vel_x = np.append(vel_x, [vel_x[-1]], axis=0)
#             vel_x = np.transpose(vel_x)
#             vel_y = np.append(vel_y, [vel_y[-1]], axis=0)
            
#             saver.save_np_f(pressure.values.numpy('y,x'), "../data/train/" + "pressure" + str(i) + "_" + str(j))
#             saver.save_np_f(smoke.values.numpy('y,x'), "../data/train/" + "smoke" + str(i) + "_" + str(j))
#             saver.save_np_f(vel_x, "../data/train/" + "vel_x" + str(i) + "_" + str(j))
#             saver.save_np_f(vel_y, "../data/train/" + "vel_y" + str(i) + "_" + str(j))

#             saver.save_np_scaled_img(pressure.values.numpy('y,x'), "../pics/" + "pressure" + str(i) + "_" + str(j))
#             saver.save_np_scaled_img(smoke.values.numpy('y,x'), "../pics/" + "smoke" + str(i) + "_" + str(j))
#             saver.save_np_scaled_img(vel_x, "../pics/" + "vel_x" + str(i) + "_" + str(j))
#             saver.save_np_scaled_img(vel_y, "../pics/" + "vel_y" + str(i) + "_" + str(j))

#             for k in range(2):
#                 velocity, smoke, pressure = step(velocity, smoke, pressure)

#             vel_x = velocity.values.vector[0].numpy('y,x')
#             vel_y = velocity.values.vector[1].numpy('y,x')

#             vel_x = np.transpose(vel_x)
#             vel_x = np.append(vel_x, [vel_x[-1]], axis=0)
#             vel_x = np.transpose(vel_x)
#             vel_y = np.append(vel_y, [vel_y[-1]], axis=0)
            
#             saver.save_np_f(pressure.values.numpy('y,x'), "../data/train/" + "pressure_target" + str(i) + "_" + str(j))
#             saver.save_np_f(smoke.values.numpy('y,x'), "../data/train/" + "smoke_target" + str(i) + "_" + str(j))
#             saver.save_np_f(vel_x, "../data/train/" + "vel_x_target" + str(i) + "_" + str(j))
#             saver.save_np_f(vel_y, "../data/train/" + "vel_y_target" + str(i) + "_" + str(j))

#             saver.save_np_scaled_img(pressure.values.numpy('y,x'), "../pics/" + "pressure_target" + str(i) + "_" + str(j))
#             saver.save_np_scaled_img(smoke.values.numpy('y,x'), "../pics/" + "smoke_target" + str(i) + "_" + str(j))
#             saver.save_np_scaled_img(vel_x, "../pics/" + "vel_x_target" + str(i) + "_" + str(j))
#             saver.save_np_scaled_img(vel_y, "../pics/" + "vel_y_target" + str(i) + "_" + str(j))

#             pylab.close('all')


save_noised_data(True)
