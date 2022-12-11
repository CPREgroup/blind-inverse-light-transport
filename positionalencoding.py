import numpy as np
from scipy import special
from scipy import signal
import math
import torch

def positionencoding1D(W, L):

    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1

    x_el = []

    x_el_hf = []

    pe_1d = np.zeros((W, 2*L+1))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        x = np.sin(val * np.pi * x_linspace)
        x_el.append(x)

        x = np.cos(val * np.pi * x_linspace)
        x_el_hf.append(x)


    for x_i in range(0, W):

        p_enc = []

        for li in range(0, L):
            p_enc.append(x_el[li][x_i])
            p_enc.append(x_el_hf[li][x_i])

        p_enc.append(x_linspace[x_i])

        pe_1d[x_i] = np.array(p_enc)

    return pe_1d.astype('float32')



def positionencoding2D(W, H, L, basis_function):
    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1
    y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1

    x_el = []
    y_el = []

    x_el_hf = []
    y_el_hf = []

    pe_2d = np.zeros((W, H, 4*L+2))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        if basis_function == 'rbf':

            # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
            # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

            # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

            M_1 = np.random.rand(2, 2)

            phase_shift = np.random.rand(1) * np.pi

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
            x_el.append(x_1_y_1[0, :])
            y_el.append(x_1_y_1[1, :])

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
            x_el_hf.append(x_1_y_1[0, :])
            y_el_hf.append(x_1_y_1[1, :])

        elif basis_function == 'diric':

            x = special.diric(np.pi * x_linspace, val)
            x_el.append(x)

            x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
            x_el_hf.append(x)

            y = special.diric(np.pi * y_linspace, val)
            y_el.append(y)

            y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
            y_el_hf.append(y)

        elif basis_function == 'sawtooth':
            x = signal.sawtooth(val * np.pi * x_linspace)
            x_el.append(x)

            x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
            x_el_hf.append(x)

            y = signal.sawtooth(val * np.pi * y_linspace)
            y_el.append(y)

            y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
            y_el_hf.append(y)

        elif basis_function == 'sin_cos':

            x = np.sin(val * np.pi * x_linspace)
            x_el.append(x)

            x = np.cos(val * np.pi * x_linspace)
            x_el_hf.append(x)

            y = np.sin(val * np.pi * y_linspace)
            y_el.append(y)

            y = np.cos(val * np.pi * y_linspace)
            y_el_hf.append(y)

    for y_i in range(0, H):
        for x_i in range(0, W):

            p_enc = []

            for li in range(0, L):
                p_enc.append(x_el[li][x_i])
                p_enc.append(x_el_hf[li][x_i])

                p_enc.append(y_el[li][y_i])
                p_enc.append(y_el_hf[li][y_i])

            p_enc.append(x_linspace[x_i])
            p_enc.append(y_linspace[y_i])

            pe_2d[x_i, y_i] = np.array(p_enc)

    return pe_2d.astype('float32')


def positionencoding3D(W, H, Z ,L, basis_function):
    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1
    y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1
    z_linspace = (np.linspace(0, Z - 1, Z) / Z) * 2 - 1
    x_el = []
    y_el = []
    z_el = []
    x_el_hf = []
    y_el_hf = []
    z_el_hf = []
    pe_3d = np.zeros((W,H,Z,6 * L + 3))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        if basis_function == 'rbf':

            # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
            # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

            # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

            M_1 = np.random.rand(3, 3)

            phase_shift = np.random.rand(1) * np.pi

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace,z_linspace))))
            x_el.append(x_1_y_1[0, :, :])
            y_el.append(x_1_y_1[1, :, :])
            z_el.append(x_1_y_1[2, :, :])

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
            x_el_hf.append(x_1_y_1[0, :, :])
            y_el_hf.append(x_1_y_1[1, :, :])
            z_el_hf.append(x_1_y_1[2, :, :])

        elif basis_function == 'diric':

            x = special.diric(np.pi * x_linspace, val)
            x_el.append(x)

            x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
            x_el_hf.append(x)

            y = special.diric(np.pi * y_linspace, val)
            y_el.append(y)

            y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
            y_el_hf.append(y)

            z = special.diric(np.pi * z_linspace, val)
            z_el.append(z)

            z = special.diric(np.pi * z_linspace + np.pi / 2.0, val)
            z_el_hf.append(z)

        elif basis_function == 'sawtooth':
            x = signal.sawtooth(val * np.pi * x_linspace)
            x_el.append(x)

            x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
            x_el_hf.append(x)

            y = signal.sawtooth(val * np.pi * y_linspace)
            y_el.append(y)

            y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
            y_el_hf.append(y)

            z = signal.sawtooth(val * np.pi * z_linspace)
            z_el.append(z)

            z = signal.sawtooth(val * np.pi * z_linspace + np.pi / 2.0)
            z_el_hf.append(z)

        elif basis_function == 'sin_cos':

            x = np.sin(val * np.pi * x_linspace)
            x_el.append(x)

            x = np.cos(val * np.pi * x_linspace)
            x_el_hf.append(x)

            y = np.sin(val * np.pi * y_linspace)
            y_el.append(y)

            y = np.cos(val * np.pi * y_linspace)
            y_el_hf.append(y)

            z = np.sin(val * np.pi * z_linspace)
            z_el.append(z)

            z = np.cos(val * np.pi * z_linspace)
            z_el_hf.append(z)
    for z_i in range(0,Z):
        for y_i in range(0, H):
            for x_i in range(0, W):
                p_enc = []
                for li in range(0, L):
                    p_enc.append(x_el[li][x_i])
                    p_enc.append(x_el_hf[li][x_i])

                    p_enc.append(y_el[li][y_i])
                    p_enc.append(y_el_hf[li][y_i])

                    p_enc.append(z_el[li][z_i])
                    p_enc.append(z_el_hf[li][z_i])

                p_enc.append(x_linspace[x_i])
                p_enc.append(y_linspace[y_i])
                p_enc.append(z_linspace[z_i])

                pe_3d[x_i, y_i,z_i] = np.array(p_enc)

    return pe_3d.astype('float32')
def positionalencoding4d(d_model, batch, height, width, channel):
    """
    :param d_model: dimension of the model
    :param batch: batch of the positions
    :param height: height of the positions
    :param width: width of the positions
    :param channel: channel of the positions
    :return: d_model*height*width*channel position matrix
    """
    if d_model % 8 != 0:
        raise ValueError("d_model must can be divided by 8")
    pe = torch.zeros(d_model, batch, height, width, channel)
    # Each dimension use half of d_model
    d_model = int(d_model / 4)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pos_b = torch.arange(0., batch).unsqueeze(1)
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pos_c = torch.arange(0., channel).unsqueeze(1)
    pe[0:d_model:2, :, :, :, :] = torch.sin(pos_b * div_term).T.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width, channel)
    pe[1:d_model:2, :, :, :, :] = torch.cos(pos_b * div_term).T.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width, channel)
    pe[d_model:2 * d_model:2, :, :, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, batch, 1, width, channel)
    pe[d_model + 1:2 * d_model:2, :, :, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, batch, 1, width, channel)
    pe[2 * d_model:3 * d_model:2, :, :, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).unsqueeze(1).unsqueeze(-1).repeat(1, batch, height, 1, channel)
    pe[2 * d_model + 1:3 * d_model:2, :, :, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).unsqueeze(1).unsqueeze(-1).repeat(1, batch, height, 1, channel)
    pe[3 * d_model::2, :, :, :, :] = torch.sin(pos_c * div_term).T.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, batch, height, width, 1)
    pe[3 * d_model + 1::2, :, :, :, :] = torch.cos(pos_c * div_term).T.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, batch, height, width, 1)

    return pe
def positionalencoding3d(d_model, height, width, channel):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width*channel position matrix
    """
    if d_model % 6 != 0:
        raise ValueError("d_model must can be divided by 6")
    pe = torch.zeros(d_model, height, width, channel)
    # Each dimension use half of d_model
    d_model = int(d_model / 3)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pos_c = torch.arange(0., channel).unsqueeze(1)
    pe[0:d_model:2, :, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).unsqueeze(-1).repeat(1, 1, width, channel)
    pe[1:d_model:2, :, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).unsqueeze(-1).repeat(1, 1, width, channel)
    pe[d_model:2*d_model:2, :, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).unsqueeze(-1).repeat(1, height, 1, channel)
    pe[d_model + 1:2*d_model:2, :, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).unsqueeze(-1).repeat(1, height, 1, channel)
    pe[2*d_model::2, :, :, :] = torch.sin(pos_c * div_term).T.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)
    pe[2*d_model + 1::2, :, :, :] = torch.cos(pos_c * div_term).T.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)

    return pe
def positionalencoding2d(d_model,height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[d_model::2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[d_model + 1::2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)

    return pe
