import torch
import math
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
trans = transforms.Normalize((0.5,), (0.5,))
def phaseFromZernikes(zCoeffs, size, norm="noll", rot=0):
    """
    Creates an array of the sum of zernike polynomials with specified coefficeints

    Parameters:
        zCoeffs (Tensor): zernike Coefficients
        size (int): Diameter of returned array
        norm (string, optional): The normalisation of Zernike modes. Can be ``"noll"``, ``"p2v"`` (peak to valley), or ``"rms"``. default is ``"noll"``.
        rot (float) : Rotates the Zernike mode by rot radians around its centre. Defaults to zero.
    Returns:
        ndarray: a `size` x `size` array of summed Zernike polynomials
    """
    Zs = zernikeArray(zCoeffs.shape[1], size, norm=norm, rot=0).unsqueeze(0).to(device)
    zCoeffs = zCoeffs.unsqueeze(-1).unsqueeze(-1).to(device)
    phase = (Zs*zCoeffs).sum(dim=1)
    min_vals = phase.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = phase.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    phase_norm = (phase - min_vals) / (max_vals - min_vals)
    phase_norm = trans(phase_norm)
    wf = phase_norm * circle(size / 2., size)

    return wf


def zernIndex(j):
    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters:
        j (int): The Noll index for Zernike polynomials

    Returns:
        list: n, m values
    """
    count = 0
    n = torch.tensor([])
    for i in range(1, 15):
        mm = count * torch.ones((1, count + 1))
        n = torch.cat((n, mm), dim=1)
        count = count + 1
    n = int(n[:, j].item())

    count = 0
    m = torch.tensor([])
    for i in range(1, 16):
        mmm = torch.arange(-count, count + 1, 2)
        m = torch.cat((m, mmm))
        count = count + 1
    m = int(m[j].item())

    return [n, m]

def zernike_noll(j, N, rot=0):
    """
     Creates the Zernike polynomial with mode index j,
     where j = 0 corresponds to piston.

     Args:
        j (int): The noll j number of the zernike mode
        N (int): The diameter of the zernike more in pixels
        rot (float): Rotates the Zernike mode by rot radians around its centre. Defaults to zero.
     Returns:
        ndarray: The Zernike mode
     """
    n, m = zernIndex(j)
    return zernike_nm(n, m, N, rot=0)


import torch

def zernike_nm(n, m, N, rot=0):
    """
     Creates the Zernike polynomial with radial index, n, and azimuthal index, m.

     Args:
        n (int): The radial order of the zernike mode
        m (int): The azimuthal order of the zernike mode
        N (int): The diameter of the zernike more in pixels
        rot (float): Rotates the zernike by rot radians around its centre. Defaults to 0.
     Returns:
        Tensor: The Zernike mode
     """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords = torch.linspace(-1, 1, N)
    X, Y = torch.meshgrid(coords, coords, indexing='ij')
    R = torch.sqrt(X**2 + Y**2).to(device)
    theta = torch.atan2(X, Y).to(device)

    if m == 0:
        Z = torch.sqrt(torch.tensor(n + 1)).to(device) * zer_rad(n, 0, R) # 将 tensor 移动到 GPU 上
    else:
        if m > 0:  # j is even
            Z = torch.sqrt(torch.tensor(2 * (n + 1))).to(device) * zer_rad(n, m, R) * torch.cos((m * theta) + rot)
        else:   # i is odd
            m = abs(m)
            Z = torch.sqrt(torch.tensor(2 * (n + 1))).to(device) * zer_rad(n, m, R) * torch.sin((m * theta) + rot)

    # clip
    Z = Z * (R <= 1.0).to(torch.float32).to(device)

    return Z * circle(N / 2., N)


def zernikeArray(J, N, norm="noll", rot=0):

    maxJ = J
    Zs = torch.empty((maxJ, N, N))
    for j in range(maxJ):
        Zs[j] = zernike_noll(j, N, rot)

    return Zs


def factorial(n):
    """
    Compute the factorial of n using torch.Tensor operations.

    Parameters:
        n (int or torch.Tensor): The input value(s) for factorial computation.

    Returns:
        torch.Tensor: The factorial of n.
    """
    return torch.exp(torch.lgamma(torch.tensor(n + 1)))


def zer_rad(n, m, rho):
    R = torch.zeros_like(rho)
    ddif = round((n-m)/2)
    dsum = round((n+m)/2)
    for s in range(ddif+1):
        numer = ((-1)**s) * math.factorial(n-s)
        denom = math.factorial(s) * math.factorial(dsum-s) * math.factorial(ddif-s)
        R = R + numer/denom * (rho**(n-2*s))
    return R

def zernikeRadialFunc(n, m, r):
    """
    Function to calculate the Zernike radial function

    Parameters:
        n (int): Zernike radial order
        m (int): Zernike azimuthal order
        r (Tensor): 2-d tensor of radii from the centre of the array

    Returns:
        Tensor: The Zernike radial function
    """

    R = torch.zeros_like(r)
    for i in range(0, int((n - m) / 2) + 1):
        R += r ** (n - 2 * i) * (((-1) ** i) * factorial(n - i).clone()) / (
                     factorial(i).clone() *
                     factorial(int(0.5 * (n + m) - i)).clone() *
                     factorial(int(0.5 * (n - m) - i)).clone()
                        )
    return R


def circle(radius, size, circle_centre=(0, 0), origin="middle"):
    """
    Create a 2-D tensor: elements equal 1 within a circle and 0 outside.

    The default centre of the coordinate system is in the middle of the tensor:
    circle_centre=(0,0), origin="middle"
    This means:
    if size is odd  : the centre is in the middle of the central pixel
    if size is even : centre is in the corner where the central 4 pixels meet

    origin = "corner" is used e.g. by psfAnalysis:radialAvg()

    Examples: ...

    Parameters:
        radius (float)       : radius of the circle
        size (int)           : size of the 2-D tensor in which the circle lies
        circle_centre (tuple): coords of the centre of the circle
        origin (str)  : where is the origin of the coordinate system
                               in which circle_centre is given;
                               allowed values: {"middle", "corner"}

    Returns:
        tensor (float64) : the circle tensor
    """
    # (2) Generate the output tensor:
    C = torch.zeros((size, size), dtype=torch.float64)

    # (3.a) Generate the 1-D coordinates of the pixel's centres:
    coords = torch.arange(0.5, size, 1.0)

    # (3.c) Generate the 2-D coordinates of the pixel's centres:
    x, y = torch.meshgrid(coords, coords, indexing='ij')

    # (3.d) Move the circle origin to the middle of the grid, if required:
    if origin == "middle":
        x = x - size / 2.
        y = y - size / 2.

    # (3.e) Move the circle centre to the alternative position, if provided:
    x -= circle_centre[0]
    y -= circle_centre[1]

    # (4) Calculate the output:
    # if distance(pixel's centre, circle_centre) <= radius:
    #     output = 1
    # else:
    #     output = 0
    mask = x * x + y * y <= radius * radius
    C[mask] = 1

    # (5) Return:
    return C.to(device)

if __name__ == '__main__':
    # for i in range(20):
    #     n, m = zernIndex(i)
    #     print(f"{n},{m}")
    fig, axs = plt.subplots(6, 10, figsize=(12, 8))  # 创建一个6行10列的子图网格
    coef = torch.zeros(60, 60)
    for i in range(60):
        coef[i, :] = torch.eye(60)[i]
    phase = phaseFromZernikes(coef, 256)
    for i in range(60):
        phase_i = phase[i, :, :].unsqueeze(0)
        phase_i = np.transpose(phase_i.cpu(), (1, 2, 0))
        plt.imshow(phase_i)
        plt.colorbar()
        plt.show()
    for i in range(60):
        coef_i = coef[:, i]
        coef_i[i] = 1
        phase = phaseFromZernikes(coef_i, 256)
        phase = np.transpose(phase.cpu(), (1, 2, 0))
        plt.imshow(phase)
        plt.colorbar()
        plt.show()

    #     row = i // 10  # 计算当前图像的行索引
    #     col = i % 10  # 计算当前图像的列索引
    #
    #     ax = axs[row, col]  # 获取对应的子图对象
    #     ax.imshow(phase)
    #     ax.set_title(f'zernike {i + 1}')
    #     ax.axis('off')  # 关闭坐标轴
    #
    # plt.tight_layout()  # 调整子图之间的间距
    # plt.show()


