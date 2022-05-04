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

def multi_step(t, INFLOW = None):
    pressure = None
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100])
    velocity = StaggeredGrid(0, extrapolation.ZERO, x=100, y=100, bounds=Box[0:100, 0:100])

    for i in range(t):
        velocity, smoke, pressure = step(velocity, smoke, pressure, INFLOW=INFLOW)

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
    return velocity.values.vector[0].numpy('y,x'), velocity.values.vector[1].numpy('y,x'), smoke.values.numpy('y,x'), pressure.values.numpy('y,x')


def generate_data(plot = False):
    t = random.randint(50, 100)
    INFLOW_ = CenteredGrid(Sphere(center=(random.randint(10, 90), random.randint(10, 40)), radius=random.randint(5, 10)), extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100]) * 0.6

    velocity, smoke, pressure = multi_step(t, INFLOW_)

    if plot:
        vals = grid_to_val(velocity, smoke, pressure)
        names = ["vel_x", "vel_y", "smoke", 'pressure']
        for i in range(4):
            saver.plot_grid_val(vals[i], names[i])

    return velocity, smoke, pressure

def save_generated_data_for_train(plot=False):
    lst = generate_data(plot)
    names = ["vel", "smoke", 'pressure']

    for i in range(3):
        saver.save_tensor_f(lst[i].values, names[i])
    
def save_generated_data_for_test(plot=False):
    velocity, smoke, pressure = generate_data(plot)
    lst = grid_to_np(velocity, smoke, pressure)
    names = ["vel_x", "vel_y", "smoke", 'pressure']

    for i in range(4):
        saver.save_np_f(lst[i], names[i])

def generate_noised_data(plot = False):
    velocity, smoke, pressure = generate_data(plot)
    lst = grid_to_np(velocity, smoke, pressure)
    
    for el in lst:
        el = scatter_noise(el)
    
    if plot:
        names = ["vel_x", "vel_y", "smoke", 'pressure']
        for i in range(4):
            saver.save_np_scaled_img(np.where(np.isnan(lst[i]), 0, lst[i]), names[i] + "_noise")

    return lst

def save_noised_data(plot=False):
    lst = generate_noised_data(plot)
    names = ["vel_x_noise", "vel_y_noise", "smoke_noise", 'pressure_noise']

    for i in range(4):
        saver.save_np_f(lst[i], names[i])


def refractor_data():
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./pics").mkdir(parents=True, exist_ok=True)

    for i in range(1,5):
        for j in range(100):
            smoke = saver.read_tensor_f("smoke" + str(i) + "_" + str(j))
            pres = saver.read_tensor_f("pressure" + str(i) + "_" + str(j))
            vel = saver.read_tensor_f("vel" + str(i) + "_" + str(j))

            pressure = CenteredGrid(pres, extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100])
            smoke = CenteredGrid(smoke, extrapolation.BOUNDARY, x=100, y=100, bounds=Box[0:100, 0:100])
            velocity = StaggeredGrid(vel, extrapolation.ZERO, x=100, y=100, bounds=Box[0:100, 0:100])

            vel_x = velocity.values.vector[0].numpy('y,x')
            vel_y = velocity.values.vector[1].numpy('y,x')

            vel_x = np.transpose(vel_x)
            vel_x = np.append(vel_x, [vel_x[-1]], axis=0)
            vel_x = np.transpose(vel_x)
            vel_y = np.append(vel_y, [vel_y[-1]], axis=0)
            
            saver.save_np_f(pressure.values.numpy('y,x'), "./data/" + "pressure" + str(i) + "_" + str(j))
            saver.save_np_f(smoke.values.numpy('y,x'), "./data/" + "smoke" + str(i) + "_" + str(j))
            saver.save_np_f(vel_x, "./data/" + "vel_x" + str(i) + "_" + str(j))
            saver.save_np_f(vel_y, "./data/" + "vel_y" + str(i) + "_" + str(j))

            saver.save_np_scaled_img(pressure.values.numpy('y,x'), "./pics/" + "pressure" + str(i) + "_" + str(j))
            saver.save_np_scaled_img(smoke.values.numpy('y,x'), "./pics/" + "smoke" + str(i) + "_" + str(j))
            saver.save_np_scaled_img(vel_x, "./pics/" + "vel_x" + str(i) + "_" + str(j))
            saver.save_np_scaled_img(vel_y, "./pics/" + "vel_y" + str(i) + "_" + str(j))

            for k in range(5):
                velocity, smoke, pressure = step(velocity, smoke, pressure)

            vel_x = velocity.values.vector[0].numpy('y,x')
            vel_y = velocity.values.vector[1].numpy('y,x')

            vel_x = np.transpose(vel_x)
            vel_x = np.append(vel_x, [vel_x[-1]], axis=0)
            vel_x = np.transpose(vel_x)
            vel_y = np.append(vel_y, [vel_y[-1]], axis=0)
            
            saver.save_np_f(pressure.values.numpy('y,x'), "./data/" + "pressure_target" + str(i) + "_" + str(j))
            saver.save_np_f(smoke.values.numpy('y,x'), "./data/" + "smoke_target" + str(i) + "_" + str(j))
            saver.save_np_f(vel_x, "./data/" + "vel_x_target" + str(i) + "_" + str(j))
            saver.save_np_f(vel_y, "./data/" + "vel_y_target" + str(i) + "_" + str(j))

            saver.save_np_scaled_img(pressure.values.numpy('y,x'), "./pics/" + "pressure_target" + str(i) + "_" + str(j))
            saver.save_np_scaled_img(smoke.values.numpy('y,x'), "./pics/" + "smoke_target" + str(i) + "_" + str(j))
            saver.save_np_scaled_img(vel_x, "./pics/" + "vel_x_target" + str(i) + "_" + str(j))
            saver.save_np_scaled_img(vel_y, "./pics/" + "vel_y_target" + str(i) + "_" + str(j))

            pylab.close('all')


refractor_data()