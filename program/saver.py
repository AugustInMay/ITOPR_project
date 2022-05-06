import pylab
from phi.torch.flow import *
import pickle

def save_np_img(data, name:str):
    pylab.imsave(name + ".png", data, origin='lower', cmap='magma')
    pylab.close('all')

def scale_data(data):
    new_data = np.zeros(np.array(data.shape)*10)

    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j*10: (j+1)*10, k*10: (k+1)*10] = data[j, k]

    return new_data

def save_np_scaled_img(data, name:str):
    save_np_img(scale_data(data), name)

def plot_grid_val(grid, name:str):
    fig = phi.vis.plot(grid, cmap='magma')
    fig.set_size_inches(12, 10)
    fig.savefig(name + ".png", dpi=100)
    pylab.close(fig)
    

def save_tensor_f(data, name:str):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(data, handle)

def read_tensor_f(name:str):
    with open(name + '.pickle', 'rb') as handle:
        return pickle.load(handle)

def save_np_f(data, name:str):
    np.save(name, data)

def read_np_f(name:str):
    return np.load(name + '.npy')