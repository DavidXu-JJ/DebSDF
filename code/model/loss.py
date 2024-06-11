import torch
from torch import nn
import utils.general as utils
import math

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    
    
class DebSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 depth_uncertainty_weight = 0.001,
                 normal_uncertainty_weight = 0.001,
                 dont_learn_big_uncer_iter=40000,
                 depth_big_uncer = 0.2,
                 normal_big_uncer = 0.3,
                 blend_big_uncer=0.25,
                 mono_init_iter = 10000,
                 enable_adap_smooth = True,
                 normal_uncertainty_blend_pow = 0.5,
                 end_step = -1,
                 nepochs = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.depth_uncertainty_weight = depth_uncertainty_weight
        self.normal_uncertainty_weight = normal_uncertainty_weight
        self.dont_learn_big_uncer_iter = dont_learn_big_uncer_iter
        self.depth_big_uncer = depth_big_uncer
        self.normal_big_uncer = normal_big_uncer
        self.mono_init_iter = mono_init_iter
        self.blend_big_uncer = blend_big_uncer
        self.normal_uncertainty_blend_pow = normal_uncertainty_blend_pow
        self.enable_adap_smooth = enable_adap_smooth

        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

        self.nepochs = nepochs

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, model_outputs, iter):
        grad_theta = model_outputs['grad_theta']
        ray_size = model_outputs['xyz_normal_uncertainty'].shape[0]
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self, model_outputs, iter):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']

        ray_size = model_outputs['blend_uncertainty'].shape[0]

        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)

        # Section (3.4) Uncertainty-Guided Smooth Regularization
        if iter < self.dont_learn_big_uncer_iter or (self.enable_adap_smooth == False):
            smooth_loss = torch.norm(normals_1 - normals_2, dim=-1).mean()
        else:
            uncer_mask = self.get_blend_uncertainty_mask(model_outputs, iter)
            temp_loss = torch.norm(normals_1 - normals_2, dim=-1)

            temp_mask = torch.ones(temp_loss.shape[0]).to(temp_loss.device)
            temp_mask[ray_size:] = uncer_mask
            temp_loss = temp_mask * temp_loss
            smooth_loss = torch.mean(temp_loss)
        return smooth_loss

    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        return self.depth_loss(depth_pred.reshape(1, 32, 32), (depth_gt * 50 + 0.5).reshape(1, 32, 32),
                               mask.reshape(1, 32, 32))

    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
        return l1, cos

    # xyz_uncertainty is already positive by employing abs() in network.py
    # Equation (11) Masked Normal uncertainty loss
    def get_normal_uncertainty_loss(self, xyz_uncertainty, normal_pred, normal_gt, mask, iter):
        # case 1
        # detach uncertainty
        xyz_uncertainty_detach = xyz_uncertainty.detach()
        uncertainty_square_detach = torch.pow(xyz_uncertainty_detach, 2) + 1e-5
        log_uncertainty_square_detach = torch.log(uncertainty_square_detach)
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)

        # dont detach normal_pred
        l1 = torch.abs(normal_pred - normal_gt)[0]
        l2 = torch.pow(l1, 2)

        normal_uncertainty_loss_batch_uncertainty_detach = torch.sum(log_uncertainty_square_detach, dim=-1) + torch.sum(
            l2 / uncertainty_square_detach, dim=-1)

        # case 2
        # dont detach uncertainty
        xyz_uncertainty_square = torch.pow(xyz_uncertainty, 2) + 1e-5
        log_xyz_uncertainty_square = torch.log(xyz_uncertainty_square)

        # detach normal_pred
        normal_pred_detach = normal_pred.detach()

        normal_pred_detach = torch.nn.functional.normalize(normal_pred_detach, p=2, dim=-1)

        l1_detach = torch.abs(normal_pred_detach - normal_gt)[0]
        l2_detach = torch.pow(l1_detach, 2)

        normal_uncertainty_loss_batch_pred_detach = torch.sum(log_xyz_uncertainty_square, dim=-1) + torch.sum(l2_detach / xyz_uncertainty_square, dim=-1)


        # Section (4.1) Implementation Details
        """
            We do not apply the estimated uncertainty regions 
            to guide ray sampling and smoothing for the first 40,000 iterations
        """
        if iter < self.dont_learn_big_uncer_iter:
            normal_uncertainty_loss = torch.mean(
                (normal_uncertainty_loss_batch_uncertainty_detach + normal_uncertainty_loss_batch_pred_detach) * mask
            )
        else:
            uncer_mask = xyz_uncertainty.mean(dim=-1) < self.normal_big_uncer
            normal_uncertainty_loss = torch.mean(
                (uncer_mask * normal_uncertainty_loss_batch_uncertainty_detach + normal_uncertainty_loss_batch_pred_detach) * mask
            )
        return normal_uncertainty_loss

    # depth_uncertainty is already positive by employing abs() in network.py
    # Equation (10) Masked Depth uncertainty loss
    def get_depth_uncertainty_loss(self, depth_uncertainty, depth_pred, depth_gt, mask, iter):
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
        depth_pred = scale.view(-1, 1, 1) * depth_pred + shift.view(-1, 1, 1)

        # case 1
        # detach uncertainty
        uncertainty_detach = depth_uncertainty.detach() + 1e-5
        log_uncertainty_detach = torch.log(uncertainty_detach)

        # dont detach depth_pred
        numerator = torch.abs(depth_pred - depth_gt)[0]

        depth_uncertainty_loss_batch_uncertainty_detach = log_uncertainty_detach + numerator / uncertainty_detach

        # case 2
        # dont detach uncertainty
        uncertainty = depth_uncertainty + 1e-5
        log_uncertainty = torch.log(uncertainty)

        # detach depth_pred
        depth_pred_detach = depth_pred.detach()
        numerator_detach = torch.abs(depth_pred_detach - depth_gt)[0]

        depth_uncertainty_loss_batch_pred_detach = log_uncertainty + numerator_detach / uncertainty

        # Section (4.1) Implementation Details
        """
            We do not apply the estimated uncertainty regions 
            to guide ray sampling and smoothing for the first 40,000 iterations
        """
        if iter < self.dont_learn_big_uncer_iter:
            depth_uncertainty_loss = torch.mean(
                (depth_uncertainty_loss_batch_uncertainty_detach + depth_uncertainty_loss_batch_pred_detach) * mask
            )
        else:
            uncer_mask = depth_uncertainty < self.depth_big_uncer
            depth_uncertainty_loss = torch.mean(
                (uncer_mask * depth_uncertainty_loss_batch_uncertainty_detach + depth_uncertainty_loss_batch_pred_detach) * mask
            )
        return depth_uncertainty_loss
        
    def forward(self, model_outputs, ground_truth, iter):
        rgb_gt = ground_truth['rgb'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]

        depth_uncertainty = model_outputs['depth_uncertainty']
        xyz_normal_uncertainty = model_outputs['xyz_normal_uncertainty']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs, iter)
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        if 'grad_theta' in model_outputs:
            smooth_loss = self.get_smooth_loss(model_outputs, iter)
        else:
            smooth_loss = torch.tensor(0.0).cuda().float()

        # Section (4.1) Implementation Details
        """
             The uncertainty localization is not stable at the initial stage.
             10,000 iterations initialization leads to a more stable uncertainty field.
        """
        if iter < self.mono_init_iter:
            if self.depth_weight != 0:
                depth_loss = self.get_depth_loss(depth_pred, depth_gt, mask)
                if isinstance(depth_loss, float):
                    depth_loss = torch.tensor(0.0).cuda().float()
            else:
                depth_loss = torch.tensor(0.0).cuda().float()

            if self.normal_cos_weight != 0 and self.normal_l1_weight != 0:
                normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
            else:
                normal_l1, normal_cos = torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()
        else:
            depth_uncertainty_loss = self.get_depth_uncertainty_loss(depth_uncertainty, depth_pred, depth_gt * 50 + 0.5, mask, iter)
            normal_uncertainty_loss = self.get_normal_uncertainty_loss(xyz_normal_uncertainty, normal_pred * mask, normal_gt, mask, iter)

        # compute decay weights
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0

        self.step += 1

        # Section (4.1) Implementation Details
        """
            We do not apply the estimated uncertainty regions 
            to guide ray sampling and smoothing for the first 40,000 iterations
            since the uncertainty localization is not stable at the initial stage.
        """
        if iter < self.mono_init_iter:
            loss = rgb_loss + \
                   self.eikonal_weight * eikonal_loss + \
                   self.smooth_weight * smooth_loss + \
                   decay * self.depth_weight * depth_loss + \
                   decay * self.normal_l1_weight * normal_l1 + \
                   decay * self.normal_cos_weight * normal_cos

            # set to zero because of initialization with naive monosdf
            depth_uncertainty_loss = torch.zeros(1)
            normal_uncertainty_loss = torch.zeros(1)
        else:
            loss = rgb_loss + \
                   self.eikonal_weight * eikonal_loss +\
                   self.smooth_weight * smooth_loss +\
                   decay * self.depth_uncertainty_weight * depth_uncertainty_loss + \
                   2 * decay * self.normal_uncertainty_weight * normal_uncertainty_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_uncertainty_loss': depth_uncertainty_loss,
            'normal_uncertainty_loss': normal_uncertainty_loss,
        }

        return output

    def get_depth_uncertainty_mask(self, model_outputs, iter):
        uncer_mask = None
        if iter >= self.dont_learn_big_uncer_iter:
            depth_uncertainty = model_outputs['depth_uncertainty']
            uncer_mask = depth_uncertainty < self.depth_big_uncer

        return uncer_mask

    def get_normal_uncertainty_mask(self, model_outputs, iter):
        uncer_mask = None
        if iter >= self.dont_learn_big_uncer_iter:
            xyz_uncertainty = model_outputs['xyz_normal_uncertainty']
            uncer_mask = xyz_uncertainty.mean(dim=-1) < self.normal_big_uncer

        return uncer_mask

    def get_blend_uncertainty_mask(self, model_outputs, iter):
        uncer_mask = None
        if iter >= self.dont_learn_big_uncer_iter:
            blend_uncertainty = model_outputs['blend_uncertainty']
            uncer_mask = blend_uncertainty < self.blend_big_uncer

        return uncer_mask

