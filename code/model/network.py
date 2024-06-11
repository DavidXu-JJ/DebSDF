import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from utils import diff_operators
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler, ErrorBoundSamplerV2, sdf_mapping
import matplotlib.pyplot as plt
import numpy as np
from hashencoder.hashgrid import _hash_encode, HashEncoder

class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = False
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        if self.use_grid_feature == True:
            # add dim channel when grid feature is activated
            dims[0] += self.grid_feature_dim
            self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                        per_level_scale=2, base_resolution=base_size,
                        log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn

            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.cache_sdf = None

    def forward(self, input, rt_embed=False):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input / self.divide_factor)
        else:
            # Note: no grid feature dim
            feature = torch.zeros(input.shape[0], 0).cuda()

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        if rt_embed:
            return x, embed
        else:
            return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]

        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = nn.Softplus(beta=100)

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        depth_uncertainty = x[:,3]
        xyz_normal_uncertainty = x[:,4:7]
        depth_uncertainty = torch.abs(depth_uncertainty) * 4
        xyz_normal_uncertainty = torch.abs(xyz_normal_uncertainty) * 4

        x = self.sigmoid(x[:,:3])

        return x, depth_uncertainty, xyz_normal_uncertainty


class DebSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.softrange_adaptive_density = conf.get_list("soft_range", default=[200000, 200001])
        self.apply_adaptive_density = conf.get_bool("apply_adaptive_density", default=False)

        self.implicit_network = ImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))

        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        sampling_method = conf.get_string('sampling_method', default="errorbounded")
        self.ray_sampler2 = ErrorBoundSamplerV2(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

        self.min_adap_density_cos = conf.get_float('min_adap_density_cos')
        self.normal_uncertainty_blend_pow = conf.get_float('normal_uncertainty_blend_pow')

        self.multires_pos = conf.get_config('implicit_network')['multires']

    def forward(self, input, indices, iter):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        # Section (3.5.4) Progressive Warm-up
        assert len(self.softrange_adaptive_density) == 2
        x1, x2 = self.softrange_adaptive_density
        assert x2 > x1
        A = math.log(2) / (x2 - x1)
        # n in code corresponds to p' in paper of Section (3.5.4)
        n = min(max(0, math.exp(A * (iter - x1)) - 1), 1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        """
            Note: 
                  (using ErrorBoundSamplerV2)
                  When (3.5) Bias-aware SDF to Density Transformation is applied,
                  the density value predicted in ray_sampler will change with approximated curvature.
                  The computation/approximation of curvature is slow in ray_sampler.
        """
        if n > 0.0 and self.apply_adaptive_density:
            z_vals, z_samples_eik = self.ray_sampler2.get_z_vals(ray_dirs, cam_loc, self, n, indices)
        else:
            z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        # Equation (8)
        # depth_uncertainty in code corresponds to u_d in the paper
        # xyz_normal_uncertainty corresponds to u_n  in the paper
        rgb_flat, depth_uncertainty, xyz_normal_uncertainty = self.rendering_network(
            points_flat, gradients, dirs_flat, feature_vectors, indices
        )
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        gradients = gradients.reshape(-1, N_samples, 3)
        normals = gradients / gradients.norm(2, -1, keepdim=True).clamp(1e-8)

        # Section (3.5) Bias-aware SDF to Density Transformation
        if n > 0.0 and self.apply_adaptive_density:
            pts_depth_uncertainty = depth_uncertainty.reshape(-1, N_samples)
            pts_normal_uncertainty = xyz_normal_uncertainty.mean(dim=-1).reshape(-1, N_samples)
            pts_blend_uncertainty = torch.pow(pts_depth_uncertainty, 1 - self.normal_uncertainty_blend_pow)\
                              * torch.pow(pts_normal_uncertainty, self.normal_uncertainty_blend_pow)

            # Section (3.5.3) Curvature Radius Estimation
            radius_of_curvature = diff_operators.estimate_curvature(z_vals, dirs, normals.detach())

            # Apply Section (3.5.2) SDF to Density Mapping for Bias Reduction
            weights = self.volume_rendering_adaptive(
                z_vals, sdf.reshape(-1, N_samples), gradients, ray_dirs, radius_of_curvature, pts_blend_uncertainty, n
            )
        else:
            weights = self.volume_rendering(z_vals, sdf.reshape(-1, N_samples))

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        # Equation (5) volume render depth
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # Equation (9) volume render depth uncertainty
        depth_uncertainty = torch.sum(weights * depth_uncertainty.reshape(z_vals.shape), 1, keepdims=True) \
                      / (weights.sum(dim=1, keepdims=True) +1e-8)

        # Equation (9) volume render normal uncertainty
        N_rays = depth_uncertainty.shape[0]
        xyz_normal_uncertainty_array = torch.zeros(N_rays, 3).cuda()
        dims = torch.arange(0, 3, 1)
        xyz_normal_uncertainty_array[:, dims] = (
                torch.sum(weights[..., None] * xyz_normal_uncertainty[:, dims].reshape(*z_vals.shape, 3), 1, keepdims=True)
                / (weights.sum(dim=1, keepdims=True) + 1e-8)
        )[:, 0]

        xyz_normal_uncertainty_map = xyz_normal_uncertainty_array.mean(dim=-1)

        # Equation (14) blend uncertainty
        blend_uncertainty = torch.pow((2 ** 0.5) * depth_uncertainty[:, 0], 1 - self.normal_uncertainty_blend_pow) \
                      * torch.pow(xyz_normal_uncertainty_map, self.normal_uncertainty_blend_pow)

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb': rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'depth_uncertainty': depth_uncertainty,
            'xyz_normal_uncertainty': xyz_normal_uncertainty_array,
            'xyz_normal_uncertainty_map': xyz_normal_uncertainty_map,
            'blend_uncertainty': blend_uncertainty,
            'n': n
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
                   
            grad_theta = self.implicit_network.gradient(eikonal_points)

            # split gradient to eikonal points and neighbour points
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        # Equation (5) volume render depth
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        return output

    def render_importance(self, input, indices, iter):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        assert len(self.softrange_adaptive_density) == 2
        x1, x2 = self.softrange_adaptive_density
        assert x2 > x1
        A = math.log(2) / (x2 - x1)
        n = min(max(0, math.exp(A * (iter - x1)) - 1), 1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        """
            Note: 
                  (using ErrorBoundSamplerV2)
                  When (3.5) Bias-aware SDF to Density Transformation is applied,
                  the density value predicted in ray_sampler will change with approximated curvature.
                  The computation/approximation of curvature is slow in ray_sampler.
        """
        if n > 0.0 and self.apply_adaptive_density:
            z_vals, z_samples_eik = self.ray_sampler2.get_z_vals(ray_dirs, cam_loc, self, n)
        else:
            z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        # Equation (8)
        # depth_uncertainty in code corresponds to u_d in the paper
        # xyz_normal_uncertainty corresponds to u_n  in the paper
        rgb_flat, depth_uncertainty, xyz_normal_uncertainty = self.rendering_network(
            points_flat, gradients, dirs_flat, feature_vectors, indices
        )

        # Equation (5) volume render depth
        gradients = gradients.reshape(-1, N_samples, 3)
        normals = gradients / gradients.norm(2, -1, keepdim=True).clamp(1e-8)

        # Section (3.5) Bias-aware SDF to Density Transformation
        if n > 0.0 and self.apply_adaptive_density:
            # Section (3.5.3) Curvature Radius Estimation
            radius_of_curvature = diff_operators.estimate_curvature(z_vals, dirs, normals.detach())

            # Apply Section (3.5.2) SDF to Density Mapping for Bias Reduction
            weights = self.volume_rendering_adaptive(
                z_vals, sdf.reshape(-1, N_samples), gradients, ray_dirs, radius_of_curvature, n
            )
        else:
            weights = self.volume_rendering(z_vals, sdf.reshape(-1, N_samples))

        # Equation (9) volume render depth uncertainty
        depth_uncertainty = torch.sum(weights * depth_uncertainty.reshape(z_vals.shape), 1, keepdims=True) \
                      / (weights.sum(dim=1, keepdims=True) + 1e-8)

        # Equation (9) volume render normal uncertainty
        N_rays = depth_uncertainty.shape[0]
        xyz_normal_uncertainty_array = torch.zeros(N_rays, 3).cuda()
        for dim in range(0, 3, 1):
            xyz_normal_uncertainty_array[:, dim] = (
                    torch.sum(weights * xyz_normal_uncertainty[:, dim].reshape(z_vals.shape), 1, keepdims=True)
                    / (weights.sum(dim=1, keepdims=True) + 1e-8)
            )[:, 0]

        xyz_normal_uncertainty_map = xyz_normal_uncertainty_array.mean(dim=-1)

        # Equation (14) blend uncertainty
        blend_uncertainty = torch.pow((2 ** 0.5) * depth_uncertainty[:, 0], 1 - self.normal_uncertainty_blend_pow) \
                      * torch.pow(xyz_normal_uncertainty_map, self.normal_uncertainty_blend_pow)

        output = {
            'blend_uncertainty': blend_uncertainty,
        }

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights

    def volume_rendering_adaptive(self, z_vals, sdf, gradients, ray_dirs, radius_of_curvature, pts_blend_uncertainty, n):
        # Apply Section (3.5.2) SDF to Density Mapping for Bias Reduction
        x = sdf_mapping(sdf, radius_of_curvature, gradients, ray_dirs, pts_blend_uncertainty, n)
        density_flat = self.density(x)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here
        return weights


