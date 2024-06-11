import abc
from tkinter.messagebox import NO
import torch
from utils import rend_util, diff_operators

# Section (3.5.2) SDF to Density Mapping for Bias Reduction
def sdf_mapping(sdf, radius_of_curvature, gradients, ray_dirs, pts_blend_uncertainty, n):
    inf_mask = torch.isinf(radius_of_curvature)
    radius_of_curvature[inf_mask] = torch.sign(radius_of_curvature[inf_mask]) * 1e9

    # Proof in Supplementary Section (1)
    # The case of big radius will converge to s(t)/|cos θ(t)|
    big_abs_radius_mask = torch.abs(radius_of_curvature) > 1e3

    gradients = gradients.detach()
    norms = gradients.norm(dim=-1, p=2, keepdim=True)
    normals = gradients / norms
    abs_cos_theta = (normals * ray_dirs[:, None]).sum(dim=-1).abs()

    # Equation (24) Progressive Warm-up
    abs_cos_theta = torch.pow(abs_cos_theta, (2 * pts_blend_uncertainty.detach() * n).clamp(0.0, 1.0)).clamp(0, 1)

    theta = torch.acos(abs_cos_theta)
    abs_tan_theta = torch.tan(theta).abs()
    abs_tan_theta[torch.isinf(abs_tan_theta)] = 1e9

    sdf = sdf / norms[..., 0]
    # Equation (21)
    a = radius_of_curvature - sdf.detach()
    # Equation (19)
    L_pow2 = torch.pow(a, 2) - torch.pow((a + sdf), 2) * (1 - abs_cos_theta ** 2)
    intersect_mask = L_pow2 > 1e-9
    x = torch.zeros_like(sdf)
    # Equation (18)
    x[intersect_mask] = (a[intersect_mask] + sdf[intersect_mask]) * abs_cos_theta[intersect_mask] - \
                        torch.sign(a[intersect_mask]) * torch.pow(L_pow2[intersect_mask], 0.5)

    # Equation (20)
    sdf_cos_weight = sdf[~intersect_mask] / (abs_cos_theta[~intersect_mask].clamp(1e-9))
    sdf_tan_weight =  sdf[~intersect_mask] * abs_tan_theta[~intersect_mask]
    x[~intersect_mask] = sdf_cos_weight + sdf_tan_weight

    without_meaning_mask = ((sdf * a) <= 0) * (sdf.abs() >= a.abs())
    without_meaning_mask = without_meaning_mask + (sdf.abs() > x.abs()) + big_abs_radius_mask
    x[without_meaning_mask] = sdf[without_meaning_mask] / abs_cos_theta[without_meaning_mask].clamp(1e-8)

    return x

# RaySampler is same as in MonoSDF
class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self,near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass

# UniformSampler is same as in MonoSDF
class UniformSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, take_sphere_intersection=False, far=-1):
        #super().__init__(near, 2.0 * scene_bounding_sphere if far == -1 else far)  # default far is 2*R
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75 if far == -1 else far)  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    # dtu and bmvs
    def get_z_vals_dtu_bmvs(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            sphere_intersections = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
            far = sphere_intersections[:,1:]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals, near, far
    
    def near_far_from_cube(self, rays_o, rays_d, bound):
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far

    # currently this is used for replica scannet and T&T
    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            _, far = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
        
        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals, near, far    


# ErrorBoundSampler is same as in MonoSDF
# This is used before Section (3.5) Bias-aware SDF to Density Transformation is applied
class ErrorBoundSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters,
                 inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=1.0e-6):
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75)
        
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval, take_sphere_intersection=True) # replica scannet and T&T courtroom

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, model):
        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        z_vals, near, far = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                samples_sdf = model.implicit_network.get_sdf_vals(points_flat)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                ''' Sample more points proportional to the current error bound'''

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:,:-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral),max=1.e6) - 1.0) * transmittance[:,:-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                ''' Sample the final sample set to be used in the volume rendering integral '''

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))


            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)


        z_samples = samples
        #TODO Use near and far from intersection
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],1).cuda()
        if self.inverse_sphere_bg: # if inverse sphere then need to add the far sphere intersection
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:,1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1]-1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:,sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere, _, _ = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]


"""
    Note: ErrorBoundSamplerV2 is slightly different from ErrorBoundSampler
    The process of ErrorBoundSampler includes the inference of density.
    When Section (3.5) Bias-aware SDF to Density Transformation is applied,
    The inference of density need to be maintained the same as Section (3.5).
"""
class ErrorBoundSamplerV2(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters,
                 inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=1.0e-6):
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75)

        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval,
                                              take_sphere_intersection=True)  # replica scannet and T&T courtroom

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, model, n, indices):
        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        z_vals, near, far = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            N_samples = points.shape[1]
            # Calculating the SDF only for the new sampled points
            samples_sdf = []
            samples_gradients = []
            samples_blend_uncertainty = []
            ray_dirs_per_points_flat = ray_dirs[:, None].repeat(1, points.shape[1], 1).reshape(-1, 3)
            split = 1024*128
            for ps, rd in zip(torch.split(points_flat, split, dim=0), torch.split(ray_dirs_per_points_flat, split, dim=0)):
                sample_sdf, _, samples_gradient = model.implicit_network.get_outputs(ps)
                with torch.no_grad():
                    sample_rgb_flat, sample_depth_uncertainty, sample_xyz_normal_uncertainty = model.rendering_network(ps, samples_gradient, rd, _, indices)
                    sample_blend_uncertainty = torch.pow(sample_depth_uncertainty, 1 - model.normal_uncertainty_blend_pow) * torch.pow(sample_xyz_normal_uncertainty.mean(dim=-1), model.normal_uncertainty_blend_pow)

                samples_blend_uncertainty.append(sample_blend_uncertainty)
                samples_sdf.append(sample_sdf)
                samples_gradients.append(samples_gradient)
            samples_sdf = torch.cat(samples_sdf, dim=0)
            samples_gradients = torch.cat(samples_gradients, dim=0)
            samples_blend_uncertainty = torch.cat(samples_blend_uncertainty, dim=0)

            samples_gradients = samples_gradients.detach()
            samples_sdf = samples_sdf.detach()
            samples_blend_uncertainty = samples_blend_uncertainty.detach()
            samples_normals = torch.nn.functional.normalize(samples_gradients, p=2, dim=-1)
            samples_raduis = diff_operators.estimate_curvature(
                samples, ray_dirs[:, None].repeat(1, points.shape[1], 1), samples_normals.reshape(-1, N_samples, 3)
            )

            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                gradients_merge = torch.cat([gradients.reshape(-1, z_vals.shape[1] - samples.shape[1], 3),
                                           samples_gradients.reshape(-1, samples.shape[1], 3)], -2)
                radius_merge = torch.cat([radius.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                           samples_raduis.reshape(-1, samples.shape[1])], -1)
                blend_uncertainty_merge = torch.cat([blend_uncertainty.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                          samples_blend_uncertainty.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
                radius = torch.gather(radius_merge, 1, samples_idx).reshape(-1, 1)
                gradients = torch.cat([torch.gather(gradients_merge[..., 0], 1, samples_idx).reshape(-1, 1),
                                     torch.gather(gradients_merge[..., 1], 1, samples_idx).reshape(-1, 1),
                                     torch.gather(gradients_merge[..., 2], 1, samples_idx).reshape(-1, 1)], dim=-1)
                blend_uncertainty = torch.gather(blend_uncertainty_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf
                gradients = samples_gradients
                radius = samples_raduis
                blend_uncertainty = samples_blend_uncertainty

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star, ray_dirs, gradients, radius, blend_uncertainty, n)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star, ray_dirs, gradients, radius, blend_uncertainty, n)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            x = sdf_mapping(sdf.reshape(z_vals.shape), radius.reshape(z_vals.shape), gradients.reshape(*z_vals.shape, 3), ray_dirs, blend_uncertainty.reshape(z_vals.shape),n)
            density = model.density(x, beta=beta.unsqueeze(-1))

            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                ''' Sample more points proportional to the current error bound'''

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:, :-1] ** 2.) / (
                            4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                ''' Sample the final sample set to be used in the volume rendering integral '''

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))
            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples
        # TODO Use near and far from intersection
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],
                                                                                               1).cuda()
        if self.inverse_sphere_bg:  # if inverse sphere then need to add the far sphere intersection
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere, _, _ = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1. / self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals.detach(), z_samples_eik.detach()

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star, ray_dirs, gradients, radius, blend_uncertainty, n):
        x = sdf_mapping(sdf.reshape(z_vals.shape), radius.reshape(z_vals.shape), gradients.reshape(*z_vals.shape, 3), ray_dirs, blend_uncertainty.reshape(z_vals.shape), n)
        density = model.density(x, beta=beta)

        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]