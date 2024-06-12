import glob
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import imageio
import matplotlib.pyplot as mplt
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth, find_max_blend_uncertainty_path

import torch.distributed as dist

class DebSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.uncertainty_map_refresh = self.conf.get_list('train.uncertainty_map_refresh', default=[40000, 100000])

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        if self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)
            self.testplot_dir = os.path.join(self.expdir, self.timestamp, 'test_plots')

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
            self.nepochs = int(self.max_total_iters / self.ds_len)
            print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        self.use_grid_feature = self.model.implicit_network.use_grid_feature
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'), nepochs = self.nepochs)

        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)

        if self.use_grid_feature == True:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()),
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam([
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)

        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        self.last_loading_path = None
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            # copy saved uncertainty map from old checkpoint to current checkpoint
            saved_maps = find_max_blend_uncertainty_path(old_checkpnts_dir)
            if saved_maps is not None and os.path.exists(saved_maps):
                self.train_dataset.load_uncertainty_map(saved_maps)
                self.refresh_dataloader()
                blend_uncertainty_checkpoint_epoch = saved_maps.split("BlendUncertainty")[-1]
                self.last_loading_path = self.checkpoints_path + '/BlendUncertainty{}'.format(blend_uncertainty_checkpoint_epoch)
                print('Copying importance map')
                os.system('cp -r {} {}'.format(saved_maps, self.checkpoints_path + '/BlendUncertainty{}'.format(blend_uncertainty_checkpoint_epoch)))

        self.is_continue = is_continue
        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()
        self.final_mesh_res = self.conf.get_int('train.final_mesh_res', default=1024)
        self.uncertainty_map_refresh = [i // self.ds_len for i in self.uncertainty_map_refresh]

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def final_evaluate_model(self, epoch):
        self.model.eval()
        surface_traces = plt.get_surface_sliding(path=self.plots_dir,
                                                 epoch=epoch + 1,
                                                 sdf=lambda x: self.model.module.implicit_network(x)[:, 0],
                                                 resolution=self.final_mesh_res,
                                                 grid_boundary=self.plot_conf['grid_boundary'],
                                                 level=0
                                                 )
        if not self.do_vis:
            return
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "depth"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "depth_files"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "rendering"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "normal"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "normal_files"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "depth_uncertainty"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "normal_uncertainty"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "normal_uncertainty_data"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "depth_uncertainty_data"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "depth_mask"))
        utils.mkdir_ifnotexists(os.path.join(self.plots_dir, "normal_mask"))
        for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            for k in ['rgb', 'depth', 'normal', 'mask']:
                ground_truth[k] = ground_truth[k].cuda()

            split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
            split_gt = utils.split_gt(ground_truth, self.total_pixels, n_pixels=self.split_n_pixels)
            res = []
            for s, s_gt in tqdm(zip(split, split_gt)):
                out = self.model(s, indices, self.iter_step)
                depth_uncertainty_mask = self.loss.get_depth_uncertainty_mask(out, self.iter_step)
                normal_uncertainty_mask = self.loss.get_normal_uncertainty_mask(out, self.iter_step)
                d = {'rgb_values': out['rgb_values'].detach(),
                     'normal_map': out['normal_map'].detach(),
                     'depth_values': out['depth_values'].detach(),
                     'depth_uncertainty': out['depth_uncertainty'].detach(),
                     'xyz_normal_uncertainty_map': out['xyz_normal_uncertainty_map'].detach(),
                     'depth_uncer_mask': None if depth_uncertainty_mask is None else depth_uncertainty_mask.detach(),
                     'normal_uncer_mask': None if normal_uncertainty_mask is None else normal_uncertainty_mask.detach(),
                     }
                if 'rgb_un_values' in out:
                    d['rgb_un_values'] = out['rgb_un_values'].detach()
                res.append(d)
                del out

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'],
                                           ground_truth['normal'], ground_truth['depth'])

            plt.last_plot(indices,
                          plot_data,
                          self.plots_dir,
                          self.nepochs,
                          self.img_res,
                          self.plot_conf['plot_nimgs']
                          )
        self.model.train()

    def run(self):
        epoch = self.start_epoch
        self.iter_step = self.start_epoch * self.ds_len

        if (self.is_continue == True and self.start_epoch == self.nepochs):
            print("Training has already been finished. Start Inferring...")
            self.do_vis = False
            self.final_evaluate_model(epoch)
            return

        print("training...")
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        for epoch in tqdm(range(self.start_epoch, self.nepochs + 1)):
            print("training:",epoch,"/",self.nepochs)

            # save checkpoint
            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            # refreshing the stored uncertainty map
            if self.GPU_INDEX == 0 and (epoch in self.uncertainty_map_refresh):
                print("loading uncertainty filter...")
                self.save_checkpoints(epoch)
                self.refresh_and_load_uncertainty_map(epoch)

            # plotting during training
            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0 and epoch != 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                for k in ['rgb', 'depth', 'normal', 'mask']:
                    ground_truth[k] = ground_truth[k].cuda()

                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                split_gt = utils.split_gt(ground_truth, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s,s_gt in tqdm(zip(split, split_gt)):
                    out = self.model(s, indices, self.iter_step)
                    depth_uncertainty_mask = self.loss.get_depth_uncertainty_mask(out, self.iter_step)
                    normal_uncertainty_mask = self.loss.get_normal_uncertainty_mask(out, self.iter_step)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach(),
                         'depth_values': out['depth_values'].detach(),
                         'depth_uncertainty': out['depth_uncertainty'].detach(),
                         'xyz_normal_uncertainty_map': out['xyz_normal_uncertainty_map'].detach(),
                         'depth_uncer_mask': None if depth_uncertainty_mask is None else depth_uncertainty_mask.detach(),
                         'normal_uncer_mask': None if normal_uncertainty_mask is None else normal_uncertainty_mask.detach(),
                         }
                    if 'rgb_un_values' in out:
                        d['rgb_un_values'] = out['rgb_un_values'].detach()
                    res.append(d)
                    del out, depth_uncertainty_mask, normal_uncertainty_mask
                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                del res
                plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])
                plt.plot(self.model.module.implicit_network,
                        indices,
                        plot_data,
                        self.plots_dir,
                        epoch,
                        self.img_res,
                        **self.plot_conf
                        )
                del plot_data, model_input, model_outputs
                self.model.train()
            # end of plotting during training

            self.train_dataset.change_sampling_idx(self.num_pixels)

            # training epoch
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                self.optimizer.zero_grad()
                model_outputs = self.model(model_input, indices, self.iter_step)

                loss_output = self.loss(model_outputs, ground_truth, self.iter_step)
                loss = loss_output['loss']
                loss.backward()
                self.optimizer.step()
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))

                self.iter_step += 1

                # tensorboard recording
                if self.GPU_INDEX == 0 and data_index % 25 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9},'
                        ' alpha={10}, n={11:.4f}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item(), model_outputs['n']))

                    self.writer.add_scalar('Loss/loss', loss.item(), self.iter_step)
                    self.writer.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/depth_uncertainty_loss', loss_output['depth_uncertainty_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/normal_uncertainty_loss', loss_output['normal_uncertainty_loss'].item(), self.iter_step)
                    self.writer.add_scalars('Model_output/depth_uncertainty_range', {
                        'Max':torch.max(model_outputs['depth_uncertainty']).item(),
                        'Mean':torch.mean(model_outputs['depth_uncertainty']).item(),
                        'Min':torch.min(model_outputs['depth_uncertainty']).item(),
                    }, self.iter_step)
                    self.writer.add_scalars('Model_output/x_normal_uncertainty', {
                        'Max':torch.max(model_outputs['xyz_normal_uncertainty'][:,0]).item(),
                        'Mean':torch.mean(model_outputs['xyz_normal_uncertainty'][:,0]).item(),
                        'Min':torch.min(model_outputs['xyz_normal_uncertainty'][:,0]).item(),
                    }, self.iter_step)
                    self.writer.add_scalars('Model_output/y_normal_uncertainty', {
                        'Max':torch.max(model_outputs['xyz_normal_uncertainty'][:,1]).item(),
                        'Mean':torch.mean(model_outputs['xyz_normal_uncertainty'][:,1]).item(),
                        'Min':torch.min(model_outputs['xyz_normal_uncertainty'][:,1]).item(),
                    }, self.iter_step)
                    self.writer.add_scalars('Model_output/z_normal_uncertainty', {
                        'Max':torch.max(model_outputs['xyz_normal_uncertainty'][:,2]).item(),
                        'Mean':torch.mean(model_outputs['xyz_normal_uncertainty'][:,2]).item(),
                        'Min':torch.min(model_outputs['xyz_normal_uncertainty'][:,2]).item(),
                    }, self.iter_step)

                    self.writer.add_scalar('Statistics/beta', self.model.module.density.get_beta().item(), self.iter_step)
                    self.writer.add_scalar('Statistics/alpha', 1. / self.model.module.density.get_beta().item(), self.iter_step)
                    self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)

                    if self.use_grid_feature == True:
                        self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                        self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                        self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                    else:
                        self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                        self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)

                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()
            self.refresh_dataloader()

        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch)

        if self.GPU_INDEX == 0:
            # Extracting final mesh for evaluation
            self.model.eval()
            surface_traces = plt.get_surface_sliding(path=self.plots_dir,
                                                     epoch=epoch + 1,
                                                     sdf=lambda x: self.model.module.implicit_network(x)[:, 0],
                                                     resolution=self.final_mesh_res,
                                                     grid_boundary=self.plot_conf['grid_boundary'],
                                                     level=0
                                                     )


    def refresh_dataloader(self):
        del self.train_dataloader
        del self.plot_dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift

        depth_uncertainty_map = model_outputs['depth_uncertainty'].reshape(batch_size, num_samples)

        xyz_normal_uncertainty_map = model_outputs['xyz_normal_uncertainty_map'].reshape(batch_size, num_samples)

        depth_uncer_mask = None
        if 'depth_uncer_mask' in model_outputs:
            depth_uncer_mask = model_outputs['depth_uncer_mask'].reshape(batch_size, num_samples)

        normal_uncer_mask = None
        if 'normal_uncer_mask' in model_outputs:
            normal_uncer_mask = model_outputs['normal_uncer_mask'].reshape(batch_size, num_samples)


        depth_uncertainty_map = depth_uncertainty_map.clamp(0. ,1.) * 255

        xyz_normal_uncertainty_map = xyz_normal_uncertainty_map.clamp(0. ,1.) * 255

        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'depth_uncertainty_map': depth_uncertainty_map,
            "xyz_normal_uncertainty_map": xyz_normal_uncertainty_map,
            'depth_uncer_mask': depth_uncer_mask,
            'normal_uncer_mask': normal_uncer_mask,
        }

        return plot_data

    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)

        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()

    def refresh_and_load_uncertainty_map(self, epoch):
        uncertainty_path = os.path.join(self.checkpoints_path, "BlendUncertainty{}".format(epoch))
        if(
                ( self.last_loading_path is not None)
                and (uncertainty_path == self.last_loading_path)
                and (self.train_dataset.uncer_sampler == True)
        ):
            print("The uncertainty map storage path exists.")
            return
        self.model.eval()
        self.train_dataset.change_sampling_idx(-1)
        utils.mkdir_ifnotexists(uncertainty_path)
        reduce_res = [192, 192]
        totol_pixels_sample = reduce_res[0] * reduce_res[1]
        self.train_dataset.change_res(reduce_res)
        self.train_dataset.uncer_sampler = True
        for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.plot_dataloader)):
            print(data_index)
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            for k in ['rgb', 'depth', 'normal', 'mask']:
                ground_truth[k] = ground_truth[k].cuda()
            split = utils.split_input(model_input, totol_pixels_sample, n_pixels=self.split_n_pixels)
            split_gt = utils.split_gt(ground_truth, totol_pixels_sample, n_pixels=self.split_n_pixels)
            res = []
            for s, s_gt in zip(split, split_gt):
                out = self.model.module.render_importance(s, indices, self.iter_step)
                d = {'blend_uncertainty': out['blend_uncertainty'].detach()}
                res.append(d)
                del out

            batch_size, num_samples, _ = ground_truth['rgb'].shape
            model_outputs = utils.merge_output(res, totol_pixels_sample, batch_size)
            blend_uncertainty_map = model_outputs['blend_uncertainty'].clone().detach().float().reshape(reduce_res)[None, None]
            blend_uncertainty_map = F.interpolate(blend_uncertainty_map, size=self.img_res, mode='bilinear', align_corners=True)
            self.train_dataset.load_uncertainty_by_idx(indices.item(), blend_uncertainty_map.reshape(-1))
            del model_outputs, res

        self.train_dataset.change_res(self.img_res)
        self.train_dataset.save_prob_maps(uncertainty_path)
        self.refresh_dataloader()

        self.model.train()