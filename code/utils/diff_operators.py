"""
    This file is borrow and modified from https://github.com/dsilvavinicius/differential_geometry_in_neural_implicits
"""
import torch
from torch.autograd import grad
import itertools

import numpy as np

def gaussian_curvature(grad, hess):
    '''
        curvature of a implicit surface (https://en.wikipedia.org/wiki/Gaussian_curvature#Alternative_formulas).
    '''
    # Append gradients to the last columns of the hessians.
    grad5d = torch.unsqueeze(grad, 2)
    grad5d = torch.unsqueeze(grad5d, -1)
    F = torch.cat((hess, grad5d), -1)

    # Append gradients (with and additional 0 at the last coord) to the last lines of the hessians.
    hess_size = hess.size()
    zeros_size = list(itertools.chain.from_iterable((hess_size[:3], [1, 1])))
    zeros = torch.zeros(zeros_size).to(grad.device)
    grad5d = torch.unsqueeze(grad, 2)
    grad5d = torch.unsqueeze(grad5d, -2)
    grad5d = torch.cat((grad5d, zeros), -1)

    F = torch.cat((F, grad5d), -2)
    grad_norm = torch.norm(grad, dim=-1)

    Kg = -torch.det(F)[-1].squeeze(-1) / (grad_norm[0] ** 4)
    return Kg


def mean_curvature(grad, hess):
    grad_norm = torch.norm(grad, dim=-1)
    Km = 0
    for i in range(3):
        Km += hess[..., i, i] * (1. / grad_norm - torch.pow(grad[..., i], 2) / torch.pow(grad_norm, 3))[...,None]

    Km *= 0.5

    return Km


def principal_curvature(grad, hess):
    Kg = gaussian_curvature(grad, hess).unsqueeze(-1)
    Km = mean_curvature(grad, hess).squeeze(0)
    A = torch.sqrt(torch.abs(torch.pow(Km, 2) - Kg) + 0.00001)
    Kmax = Km + A
    Kmin = Km - A

    return Kmin, Kmax


# Che, Wujun, Jean-Claude Paul, and Xiaopeng Zhang.
# "Lines of curvature and umbilical points for implicit surfaces.
# " Computer Aided Geometric Design 24.7 (2007): 395-409.
def principal_directions_base(grad, hess):

    def get_coefficient(grad, hess):
        A = grad[..., [1]] * hess[..., 0, 2] - grad[..., [2]] * hess[..., 0, 1]
        B = 0.5 * (grad[..., [2]] * hess[..., 0, 0] - grad[..., [0]] * hess[..., 0, 2] + grad[..., [1]] * hess[..., 1, 2] -
                   grad[..., [2]] * hess[..., 1, 1])
        C = 0.5 * (grad[..., [1]] * hess[..., 2, 2] - grad[..., [2]] * hess[..., 1, 2] + grad[..., [0]] * hess[..., 0, 1] -
                   grad[..., [1]] * hess[..., 0, 0])
        D = grad[..., [2]] * hess[..., 0, 1] - grad[..., [0]] * hess[..., 1, 2]
        E = 0.5 * (grad[..., [0]] * hess[..., 1, 1] - grad[..., [1]] * hess[..., 0, 1] + grad[..., [2]] * hess[..., 0, 2] -
                   grad[..., [0]] * hess[..., 2, 2])
        F = grad[..., [0]] * hess[..., 1, 2] - grad[..., [1]] * hess[..., 0, 2]

        U = A * grad[..., [2]] ** 2 - 2. * C * grad[..., [0]] * grad[..., [2]] + F * grad[..., [0]] ** 2
        V = 2 * (B * grad[..., [2]] ** 2 - C * grad[..., [1]] * grad[..., [2]] - E * grad[..., [0]] * grad[..., [2]] + F *
                 grad[..., [0]] * grad[..., [1]])
        W = D * grad[..., [2]] ** 2 - 2. * E * grad[..., [1]] * grad[..., [2]] + F * grad[..., [1]] ** 2

        return U, V, W

    U, V, W = get_coefficient(grad, hess)

    # Hz signal
    s = torch.sign(grad[..., [2]])

    # U == 0 and W == 0
    # print(U.abs().max(), W.abs().max())
    UW_mask = (torch.abs(U) < 1e-6) * (torch.abs(W) < 1e-6)
    UW_mask_shape = list(UW_mask.shape)
    UW_mask_shape[-1] *= 3
    UW_mask_3 = UW_mask.expand(UW_mask_shape)

    # U != 0 or W != 0
    mask = ~UW_mask
    mask_3 = ~UW_mask_3

    # if U != 0 or W != 0, but U == 0
    # we should switch the axis from (x,y,z) to (y,z,x)
    # then finally switch back from (y,z,x) to (x,y,z)
    U0_mask = (torch.abs(U) < 1e-6) * mask
    U0_mask_3 = U0_mask.expand(UW_mask_shape)
    if U0_mask.any():
        grad_U0 = torch.zeros_like(grad)
        grad_U0[...,0:2] = grad[...,1:3]
        grad_U0[...,2] = grad[...,0]

        hess_U0 = torch.zeros_like(hess)
        hess_U0[...,0,0] = hess[...,1,1]
        hess_U0[...,0,1] = hess[...,1,2]
        hess_U0[...,0,2] = hess[...,1,0]
        hess_U0[...,1,0] = hess[...,2,1]
        hess_U0[...,1,1] = hess[...,2,2]
        hess_U0[...,1,2] = hess[...,2,0]
        hess_U0[...,2,0] = hess[...,0,1]
        hess_U0[...,2,1] = hess[...,0,2]
        hess_U0[...,2,2] = hess[...,0,0]

        U_U0, V_U0, W_U0 = get_coefficient(grad_U0, hess_U0)

        s_U0 = torch.sign(grad_U0[..., [2]])

    def get_dir(s, U, V, W, grad):
        # first direction (U!=0 or W!=0)
        T1x = (-V + s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [2]]
        T1y = 2 * U * grad[..., [2]]
        T1z = (V - s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [0]] - 2 * U * grad[..., [1]]
        dir_min = torch.cat((T1x, T1y), -1)
        dir_min = torch.cat((dir_min, T1z), -1)

        # second direction (U!=0 or W!=0)
        T2x = (-V - s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [2]]
        T2y = 2 * U * grad[..., [2]]
        T2z = (V + s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [0]] - 2 * U * grad[..., [1]]
        dir_max = torch.cat((T2x, T2y), -1)
        dir_max = torch.cat((dir_max, T2z), -1)

        return dir_min, dir_max

    dir_min, dir_max = get_dir(s, U, V, W, grad)

    if U0_mask.any():
        dir_min_U0, dir_max_U0 = get_dir(s_U0, U_U0, V_U0, W_U0, grad_U0)

        dir_min_U0 = torch.cat([dir_min_U0[...,2:], dir_min_U0[...,:2]], dim=-1)
        dir_max_U0 = torch.cat([dir_max_U0[...,2:], dir_max_U0[...,:2]], dim=-1)

        dir_min = torch.where(U0_mask_3, dir_min_U0, dir_min)
        dir_max = torch.where(U0_mask_3, dir_max_U0, dir_max)

    if UW_mask.any():
        # first direction (U==0 and W==0)
        T1x_UW = torch.zeros_like(grad[..., [0]])
        T1y_UW = grad[..., [2]]
        T1z_UW = - grad[..., [1]]
        dir_min_UW = torch.cat((T1x_UW, T1y_UW), -1)
        dir_min_UW = torch.cat((dir_min_UW, T1z_UW), -1)

        # second direction (U==0 and W==0)
        T2x_UW = grad[..., [2]]
        T2y_UW = torch.zeros_like(grad[..., [0]])
        T2z_UW = - grad[..., [0]]
        dir_max_UW = torch.cat((T2x_UW, T2y_UW), -1)
        dir_max_UW = torch.cat((dir_max_UW, T2z_UW), -1)

        dir_min = torch.where(mask_3, dir_min, dir_min_UW)
        dir_max = torch.where(mask_3, dir_max, dir_max_UW)

    # computing the umbilical points
    # umbilical = torch.where(torch.abs(U)+torch.abs(V)+torch.abs(W)<1e-6, -1, 0)

    # normalizing the principal directions
    len_min = dir_min.norm(dim=-1).unsqueeze(-1)
    len_max = dir_max.norm(dim=-1).unsqueeze(-1)
    def check_dir_nan(dirs, len_dir, temp, grad):
        mask = (len_dir < 1e-12)[..., 0]
        dirs[mask] = torch.cross(temp[mask], grad[mask], dim=-1)
        len_dir[mask] = dirs[mask].norm(dim=-1)[..., None]
        return dirs, len_dir

    dir_min, len_min = check_dir_nan(dir_min, len_min, dir_max, grad)
    dir_max, len_max = check_dir_nan(dir_max, len_max, dir_min, grad)

    return dir_min / len_min, dir_max / len_max

def principal_directions(grad, hess):

    # df/dz != 0
    dir_min, dir_max = principal_directions_base(grad, hess)

    # df/dz == 0
    Z0_mask = (torch.abs(grad[..., -1]) < 2e-2).unsqueeze(dim=-1)
    if Z0_mask.any():
        Z0_mask_shape = list(Z0_mask.shape)
        Z0_mask_shape[-1] *= 3
        Z0_mask_3 = Z0_mask.expand(Z0_mask_shape)

        grad_Z0 = torch.zeros_like(grad)
        grad_Z0[...,0] = grad[...,2]
        grad_Z0[...,1:3] = grad[...,0:2]

        hess_Z0 = torch.zeros_like(hess)
        hess_Z0[...,0,0] = hess[...,2,2]
        hess_Z0[...,0,1] = hess[...,2,0]
        hess_Z0[...,0,2] = hess[...,2,1]
        hess_Z0[...,1,0] = hess[...,0,2]
        hess_Z0[...,1,1] = hess[...,0,0]
        hess_Z0[...,1,2] = hess[...,0,1]
        hess_Z0[...,2,0] = hess[...,1,2]
        hess_Z0[...,2,1] = hess[...,1,0]
        hess_Z0[...,2,2] = hess[...,1,1]

        dir_min_Z0, dir_max_Z0 = principal_directions_base(grad_Z0, hess_Z0)
        dir_min_Z0 = torch.cat([dir_min_Z0[...,1:3], dir_min_Z0[...,:1]], dim=-1)
        dir_max_Z0 = torch.cat([dir_max_Z0[...,1:3], dir_max_Z0[...,:1]], dim=-1)

        # df/dz == 0 && df/dy == 0
        Z0_Y0_mask = (torch.abs(grad[..., -2]) < 2e-2).unsqueeze(dim=-1) * Z0_mask
        if Z0_Y0_mask.any():
            Z0_Y0_mask_3 = Z0_Y0_mask.expand(Z0_mask_shape)

            grad_Z0_Y0 = torch.zeros_like(grad)
            grad_Z0_Y0[...,0:2] = grad[...,1:3]
            grad_Z0_Y0[...,2] = grad[...,0]

            hess_Z0_Y0 = torch.zeros_like(hess)
            hess_Z0_Y0[...,0,0] = hess[...,1,1]
            hess_Z0_Y0[...,0,1] = hess[...,1,2]
            hess_Z0_Y0[...,0,2] = hess[...,1,0]
            hess_Z0_Y0[...,1,0] = hess[...,2,1]
            hess_Z0_Y0[...,1,1] = hess[...,2,2]
            hess_Z0_Y0[...,1,2] = hess[...,2,0]
            hess_Z0_Y0[...,2,0] = hess[...,0,1]
            hess_Z0_Y0[...,2,1] = hess[...,0,2]
            hess_Z0_Y0[...,2,2] = hess[...,0,0]

            dir_min_Z0_Y0, dir_max_Z0_Y0 = principal_directions_base(grad_Z0_Y0, hess_Z0_Y0)
            dir_min_Z0_Y0 = torch.cat([dir_min_Z0_Y0[...,2:], dir_min_Z0_Y0[...,:2]], dim=-1)
            dir_max_Z0_Y0 = torch.cat([dir_max_Z0_Y0[...,2:], dir_max_Z0_Y0[...,:2]], dim=-1)

            dir_min_Z0 = torch.where(Z0_Y0_mask_3, dir_min_Z0_Y0, dir_min_Z0)
            dir_max_Z0 = torch.where(Z0_Y0_mask_3, dir_max_Z0_Y0, dir_max_Z0)


        dir_min = torch.where(Z0_mask, dir_min_Z0, dir_min)
        dir_max = torch.where(Z0_mask, dir_max_Z0, dir_max)

    return dir_min, dir_max


def check_dir_nan(dirs, len_dir, temp, grad):
    mask = (len_dir < 1e-12)[..., 0]
    dirs[mask] = torch.cross(temp[mask], grad[mask])
    len_dir[mask] = dirs[mask].norm(dim=-1)[..., None]

    return dirs, len_dir

def principal_curvature_parallel_surface(Kmin, Kmax, t):
    Kg = Kmin * Kmax
    Km = 0.5 * (Kmin + Kmax)

    # curvatures of the parallel surface [manfredo, pg253]
    aux = np.ones_like(Kg) - 2. * t * Km + t * t * Kg
    aux[np.absolute(aux) < 0.0000001] = 0.0000001

    aux_zero = aux[np.absolute(aux) < 0.0000001]
    if (aux_zero.size > 0):
        raise Exception('aux has zero members: ' + str(aux_zero))

    newKg = Kg / aux
    newKm = (Km - t * Kg) / aux

    A = np.sqrt(np.absolute(newKm ** 2 - newKg) + 0.00001)
    newKmax = newKm + A
    newKmin = newKm - A

    return newKmin, newKmax


def principal_curvature_region_detection(y, x):
    grad = gradient(y, x)
    hess = hessian(y, x)

    # principal curvatures
    min_curvature, max_curvature = principal_curvature(y, x, grad, hess)

    # Harris detector formula
    return min_curvature * max_curvature - 0.05 * (min_curvature + max_curvature) ** 2
    # return min_curvature*max_curvature - 0.5*(min_curvature+max_curvature)**2


def umbilical_indicator(y, x):
    grad = gradient(y, x)
    hess = hessian(y, x)

    # principal curvatures
    min_curvature, max_curvature = principal_curvature(y, x, grad, hess)

    # Harris detector formula
    # return min_curvature*max_curvature - 0.05*(min_curvature+max_curvature)**2
    return 1 - torch.abs(torch.tanh(min_curvature) - torch.tanh(max_curvature))


def tensor_curvature(y, x):
    grad = gradient(y, x)
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad.squeeze(-1) / grad_norm.unsqueeze(-1)

    T = -jacobian(unit_grad, x)[0]

    print(T)
    e, v = torch.eig(T, eigenvectors=True)

    print(e)
    print(v)

    return T


def gauss_bonnet_integral(grad, hess):
    Kg = gaussian_curvature(grad, hess).unsqueeze(-1)

    # remenber to restrict to the surface
    # Kg = torch.where(gt_sdf != -1, Kg, torch.zeros_like(Kg))

    aux = gradient.squeeze(-1) / torch.abs(gradient[:, :, 0].unsqueeze(-1))

    Kg = Kg * (aux.norm(dim=-1).unsqueeze(-1))
    return torch.sum(Kg) / (Kg.shape[1] * 0.5)


def hessian(y, x):
    """Hessian of y wrt x
    Parameters
    ----------
    y: torch.Tensor
        Shape [B, N, D_out], where B is the batch size (usually 1), N is the
        number of points, and D_out is the number of output channels of the
        network.
    x: torch.Tensor
        Shape [B, N, D_in],  where B is the batch size (usually 1), N is the
        number of points, and D_in is the number of input channels of the
        network.
    Returns
    -------
    h: torch.Tensor
        Shape: [B, N, D_out, D_in, D_in]
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            tmp = grad(dydx[..., j], x, grad_y, create_graph=True)
            h[..., i, j, :] = tmp[0].unsqueeze(1)[..., :]

    return h


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(
        y.device)  # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def get_curvature(dirs_flat, gradients, hessian):
    with torch.no_grad():
        gradients_clone = gradients.clone()
        # zero_mask = torch.abs(gradients_clone[...,-1]) < 1e-10
        # gradients_clone[...,-1][zero_mask] = 1e-8 * (torch.rand(1) - 1)
        dirs_rev = - dirs_flat
        tangent_dir = dirs_rev - (dirs_rev * gradients_clone).sum(dim=-1, keepdim=True) * gradients_clone
        tangent_dir = torch.nn.functional.normalize(tangent_dir, p=2, dim=-1).detach()

        # for Hessian matrix of a single point
        hessian = hessian.reshape([-1,3,3])
        hessian = hessian.reshape([-1, 3, 3]).unsqueeze(1).unsqueeze(0)

        K_min, K_max = principal_curvature(gradients_clone.unsqueeze(0), hessian)

        dir_min, dir_max = principal_directions(gradients_clone.unsqueeze(0), hessian)

        dir_min = dir_min[0]
        dir_max = dir_max[0]

        cos_theta = (tangent_dir * dir_max).sum(dim=-1)
        cos_theta_square = torch.pow(cos_theta, 2)[..., None]
        sin_theta_square = 1 - cos_theta_square

        Kappa_n = (K_min * cos_theta_square + K_max * sin_theta_square).detach()

    # [N, 1]
    return Kappa_n

# Section (3.5.3) Curvature Radius Estimation
def estimate_curvature(z_vals, dirs, normals):
    radius = torch.ones_like(z_vals) * 1e10
    normals_next = normals[:, 1:]
    normals_pre = normals[:, :-1]
    cross_vectors = torch.nn.functional.normalize(torch.cross(normals_pre, dirs[:, :-1]), dim=-1)
    normals_next = normals_next - (normals_next * cross_vectors).sum(dim=-1, keepdim=True) * cross_vectors
    normals_next = torch.nn.functional.normalize(normals_next, dim=-1)

    sinA_half = (normals_next - normals_pre).norm(dim=-1) / 2
    sinA_half = torch.sin(torch.arcsin(sinA_half.clamp(0, 1)) * 2)

    cos_next = (normals_next * dirs[:, :-1]).sum(dim=-1)
    cos_pre = (normals_pre * dirs[:, :-1]).sum(dim=-1)
    minus_mask = cos_pre > cos_next

    sin_nexthalf = (normals_next - dirs[:, :-1]).norm(dim=-1) / 2
    sin_next = torch.sin(torch.arcsin(sin_nexthalf.clamp(0, 1)) * 2)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    radius[:, :-1] = dists * sin_next / sinA_half.clamp(1e-9)
    radius[:, :-1][minus_mask] *= -1
    if torch.isnan(radius).any() or torch.isinf(radius).any():
        import ipdb
        ipdb.set_trace()
    return radius.detach()
