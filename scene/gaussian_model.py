#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        """
        Set up the activation functions and inverse activation functions for the GaussianModel.

        This function sets up the activation functions and inverse activation functions for the GaussianModel. It defines the `build_covariance_from_scaling_rotation` function, which takes in scaling, scaling_modifier, and rotation as parameters and returns the symmetric covariance matrix.

        This function does not take any parameters and does not return any values.
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation): # i  see scaling_modifer only in get_covariance function.
            L = build_scaling_rotation(scaling_modifier * scaling, rotation) # the shape is: Nx3x3 rotation @ Nx3x3 scaling matrix
            actual_covariance = L @ L.transpose(1, 2) # cf. formula (6) in vanilla 3dgs paper, the formula for covariance matrix \Sigma of a 3D gaussian. 
            # A more detailed illustration of the formula can be found in README.md.
            symm = strip_symmetric(actual_covariance) # makes it (N,1) tensor.
            return symm     # (N,6) tensor shape
        
        # the following is activation functions as well some inverse activation.
        self.scaling_activation = torch.exp  # vanilla 3dgs paper says this is for a smooth gradient
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation     # (N,6) tensor shape

        self.opacity_activation = torch.sigmoid #  1 / (1 + e^(-x)), squishes real numbers between 0 and 1.
        self.inverse_opacity_activation = inverse_sigmoid  
        # the logit function, takes a value between 0 and 1 (often interpreted as a probability) 
        # and transforms it to a real number on the entire number line.

        self.rotation_activation = torch.nn.functional.normalize  # By default, it normalizes along dimension 1 (across rows for a matrix)

        ## xyz is not `activated`, nor is features.

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)  # absent in vanilla 3dgs 
        # I asked about this with jumpat, the great author of repo <seganygaussians>, who told me that 
        # this _mask is designed for the SAGA GUI to support interactive segmentation, specifically, the Undo operation. 
        # _mask is a counter to trace the current segmentation state and save the previous states.
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0) # radii is something occuring in rendering process, determininng the range of (concerned) point cloud.
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None # initialzie the optimizer
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # the following is absent in vanilla 3dgs
        self.old_xyz = []
        self.old_mask = []

        self.old_features_dc = []
        self.old_features_rest = []
        self.old_opacity = []
        self.old_scaling = []
        self.old_rotation = []

    def capture(self): # this is for saving checkpoint by torch.save in train_scene.py
        return (
            self.active_sh_degree,
            self._xyz,
            self._mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),  # state_dict() creates a dictionary containing the current state of the optimizer. This state includes all the internal variables used by the optimizer to update the model's parameters during training. 
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args): # this is for loading non-zeroth checkpoint for initialization in train_scene.py, like an inverse operation against `capture()`.
        (self.active_sh_degree, 
        self._xyz,
        self._mask,   # absent in vanilla 3dgs
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        ## why did't vanilla 3dgs author merge the two lines into the set bracket like others?

        self.optimizer.load_state_dict(opt_dict)  #  load_state_dict() takes a dictionary (the state_dict) as input and loads it into the existing optimizer object. It essentially restores the optimizer to the state it was in when the state_dict was created.

    # The following 6 get_ properties are convenient to leave activation when obtaining. 6 among the 7 tensors are optimzed in the project via pytorch.
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property  
    def get_mask(self):
        return self._mask
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float): #pcd: a class of three np.arrays.
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() 
        # pcd.points: 1 np.array, make it float and sent to cuda.

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) 
        # pcd.points: 1 np.array, make it float and sent to cuda, then from rgb to spherical harmonics. Each rgb channel is now waiting for several SH polynomicals to descriibe via coefficients.

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features initialization: e.g. (N, 3, 9), N being number of points. 
        # In spherical harmonics,  m and l are integers that define the specific harmonic function and its properties. Here's a breakdown of their roles:

            # 1. Orbital Angular Momentum Quantum Number (l):
            # Represents the total angular momentum of the wavefunction associated with the spherical harmonic.
            # Higher values of l correspond to more complex shapes of the wavefunction, with more nodes and lobes.
            # l can take non-negative integer values, starting from 0: l = 0, 1, 2, 3, ...
    
            # 2. Magnetic Angular Momentum Quantum Number (m):
            # Represents the z-component of the angular momentum.
            # Defines the orientation of the spherical harmonic around the z-axis.
            # m can range from -l to +l, including 0: m = -l, -l+1, ..., 0, ..., l-1, l
            # For a given l, there are always 2l + 1 possible values of m.
            # Relationship between m and l:
    
            # The number of unique spherical harmonics for a given l is determined by 2l+1.
            # For example:
            # When l=0 (the simplest case), there's only one spherical harmonic (m = 0). This represents a sphere.
            # When l=1, there are three spherical harmonics (m = -1, 0, 1). These describe more complex shapes like p-orbitals in atoms.
            # Visualization:

        # Spherical harmonics can be visualized as functions over the unit sphere, representing the probability density of finding an electron in an atom. Different values of l and m correspond to different shapes of these probability distributions.
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics

        # And send it to cuda after being made sure to be float.

        features[:, :3, 0 ] = fused_color # corresponding to feature_dc in the below.
        features[:, 3:, 1:] = 0.0 
        # corresponding to feature_rest in the below. 
        # But this line of code seems not to be functioning in reality beacuse the 1st dimension is actually null. And the tensor is initialized with zeros already.

        print("Number of points at initialisation : ", fused_point_cloud.shape[0]) 

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) # dist2 may be of shape of (N,) where N is the number of points.
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # scales is of shape (N,3), i.e. for every point there is a three-dimensional scaling coefficient.
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") # rots is of shape (N,4), i.e. for every point there is a four-dimensional quaternion.
        rots[:, 0] = 1 # the first of an initialized quaternion is usually 1.
        mask = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") # the `mask` here is not used. absent in vanilla 3dgs

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # (N,1), every point has its opacity that is restored from 0-1 range to real number range.

        # The following 6 things are the parameters to be learnt through gradient descent training.
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # the transpose here as well the one in next line maybe represents athta  for dimension 1, it is the 'scale' of sh; for dimension 2, it si the 'content' of sh?
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self._mask = nn.Parameter(mask.requires_grad_(True))
        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # the following l registers a dict of waht parameters to be trained, with   `params`, `lr`, and `name` three keys. Altogether 6 things to learn.
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},

            # {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) # this is torch optimizer, 
                                                                # it has state_dict(), load_state_dict(), zero_grad(), step(), param_groups, etc.
                                                                # You can see them in this file.
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)  # this return a function caller:  lr init, lr final, lr dealy mult, and max steps, the corresponding lr would be output such that the step-lr is log-linearly.

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups: # in optimizer of pytorch, param_groups is the director of optimization.
            if param_group["name"] == "xyz": # which means other parameters than xyz positions are being trained with fixed lr.
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):   
        mkdir_p(os.path.dirname(path))

        # release all the learnt parameters to cpu 
        xyz = self._xyz.detach().cpu().numpy() # when it comes to save, it has to be on cpu
        # mask = self._mask.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # every point has their normals?? seems like they are all 0.
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # f_dc has shape (N, d)
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # f_rest has shape (N, r)
        opacities = self._opacity.detach().cpu().numpy() 
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # `.detach()` detaches the tensor from the computational graph.  
        # Does not change the device: The tensor remains on the original device (CPU or GPU) it was on before calling .detach().
        # `cpu()` moves the tensor to the CPU memory.
        # if the tensor was already on the CPU, cpu() has no effect.

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()] 
        #  'f4' represents a data type in NumPy for a 32-bit floating point number.
        #  which means that dtype_full is a python list of one sort.

        elements = np.empty(xyz.shape[0], dtype=dtype_full) # elements is an array of N vertices, with shape (N,)
        # attributes = np.concatenate((xyz, mask, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) if has_mask else np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) # the shape of attributes is (N, d + r + 3 + 1 + 1 + 3), where N is the number of points in the point cloud and d and r are the dimensions of f_dc and f_rest, respectively.
        elements[:] = list(map(tuple, attributes))
        # The map(tuple, attributes) function converts each row of attributes into a tuple, 
        # and then the list() function converts the resulting iterator into a list of tuples. 
        # The elements[:] syntax is used to assign the values to all the elements of the elements array.
        # the attributes is align with dtype_full now, as you can check.

        el = PlyElement.describe(elements, 'vertex') # syntax in PlyElement.
        PlyData([el]).write(path)

    def save_mask(self, path): # it's effectively save the _mask
        mkdir_p(os.path.dirname(path))
        mask = self._mask.detach().cpu().numpy()    
        np.save(path, mask)

    def reset_opacity(self): # used in densification phase in train_scene.py, claimed to be an effective way to moderate the increae of the number of gaussians.
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))  # cannot exceed 0.01
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # mask = np.asarray(plydata.elements[0]["mask"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        # self._mask = nn.Parameter(torch.tensor(mask, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # Two conditions on optimizer.state's keys:
                # Integers: By default, PyTorch uses sequential integers (0, 1, 2, ...) to identify parameter groups.
                # Custom Names: If you explicitly define names for parameter groups during optimizer creation, those names will be used as keys in the state dictionary.
                    # e.g.:
                    # import torch.optim as optim
                    # # Create a model
                    # model = torch.nn.Linear(10, 5)

                    # # Option 1: Default integer keys
                    # optimizer1 = optim.SGD(model.parameters(), lr=0.01)
                    # state_dict1 = optimizer1.state

                    # # Option 2: Custom group names
                    # param_groups = [{'params': model.fc1.parameters(), 'lr': 0.001},
                    #                 {'params': model.fc2.parameters(), 'lr': 0.01}]
                    # optimizer2 = optim.SGD(param_groups)
                    # state_dict2 = optimizer2.state

                    # # Print the state dictionaries (keys will differ)
                    # print(state_dict1)  # Might output: {'0': {...}, '1': {...}} (assuming two FC layers)
                    # print(state_dict2)  # Might output: {'fc1': {...}, 'fc2': {...}} (using custom names)
                # Also, not that:
                # In PyTorch, optimizer.param_groups and optimizer.state are distinct dictionaries 
                # that serve different purposes related to optimizer configuration and state management:
                    # param_groups = [{'params': model.fc1.parameters(), 'lr': 0.01},  # Lower learning rate for fc1
                    # {'params': model.fc2.parameters(), 'lr': 0.005}   # Higher learning rate for fc2
                    # ]
                    # optimizer.state = {
                    #     # Key for the parameter group (might be an integer or name)
                    #     '0': {
                    #         # Exponential moving average of the squared gradients
                    #         'exp_avg_sq': torch.tensor([0.1, 0.2, 0.3, ...]),  # Example values for each parameter
                    #         # Exponential moving average of the gradients (estimated moment)
                    #         'exp_avg': torch.tensor([0.01, 0.02, 0.03, ...]),  # Example values for each parameter
                    #         # Number of steps taken since the beginning (for bias correction)
                    #         'step': 100
                    #     }
                    # }
                    # ！！！But, If you define your parameter groups with custom names during optimizer creation, 
                    # those names will be used as keys in the state dictionary.！！！
                    # This provides clearer identification of the parameter groups within the state.

                stored_state["exp_avg"] = torch.zeros_like(tensor)  
                # The key 'exp_avg' in the PyTorch optimizer state dictionary refers to the exponential moving average of gradients. 
                # This value is used by optimizers like Adam (Adaptive Moment Estimation) and RMSprop (Root Mean Square Prop) 
                # to address issues with noisy gradients during training.
                # Optimizers like Adam and RMSprop leverage the 'exp_avg' value along with another key, 
                # 'exp_avg_sq' (exponential moving average of squared gradients), 
                # to adapt the learning rate for each parameter based on the recent gradient history.
                    # It's important to note that not all optimizers in PyTorch use the 'exp_avg' key. 
                    # For instance, SGD (Stochastic Gradient Descent) relies solely on the current gradient for updates and doesn't require exponential averaging.
                # The shape of exp_avg in the optimizer's state dictionary from torch.state_dict()  matches the shape of the corresponding tensor it tracks the average for.
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # in PyTorch optimizers, the state dictionary (state) of a tensor can hold various entries besides exp_avg and exp_avg_sq depending on the specific optimizer used. Here are some common entries you might encounter:
                # 1. Momentum based optimizers (e.g., SGD with momentum, Adam):

                # momentum: This stores the momentum term used by the optimizer. It helps the optimizer to converge faster by accumulating the gradients in previous steps.
                # 2. Adam family optimizers (Adam, AdamW, Adadelta, etc.):

                # beta1 and beta2: These are hyperparameters that control the exponential decay rates of the first and second moment estimates, respectively, used in Adam variants.
                # eps: A small value for numerical stability, similar to the eps used in normalization.
                # 3. RMSprop and variants (RMSprop, Adagrad):

                # square_avg: This stores the squared average of the gradients used for normalization in RMSprop variants.
                # 4. Other optimizers:

                # Some optimizers might have additional state entries specific to their algorithms. It's always best to consult the documentation for the specific optimizer you're using to understand its state dictionary entries.
                # Here are some resources for further exploration:

                # PyTorch documentation on optimizers: https://pytorch.org/docs/stable/optim.html
                # Specific optimizer documentation (e.g., Adam): https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                #  In summary, exp_avg and exp_avg_sq are commonly used for momentum and Adam-based optimizers, but the state dictionary can hold various other entries depending on the specific optimizer and its underlying algorithm.


                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask): # here the mask is the filter that picks out the gaussians  in view frustum.
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None) # group['params'] is a list, so index it with [0]; group['params'][0] shoudl be a class variable e.g. self._xxxx.
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask] 
                # if the state has the propoerties like `exp_avg` and `exp_avg_sq`, 

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # i optimization only masked tensors needs be optimized, so it be reestablished; otherwise, the tensor still gets optimized completely.
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        # self._mask = optimizable_tensors["mask"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    @torch.no_grad()
    def segment(self, mask=None): 
        # I asked the author of this code 'jumpat' to explain how this function works. 
        # This function emoved the background Gaussians and its related attributes from the model. 
        # The remained Gaussians describe the target object you segmented (stored as precomputed_mask.pt, which is a binary mask)
        assert mask is not None
            # mask = (self._mask > 0)
        mask = mask.squeeze() # squeeze() function removes all dimensions of a tensor that have a size of 1
        # assert mask.shape[0] == self._xyz.shape[0]
        if torch.count_nonzero(mask) == 0:
            mask = ~mask
            print("Seems like the mask is empty, segmenting the whole point cloud. Please run seg.py first.")

        self.old_xyz.append(self._xyz)
        self.old_mask.append(self._mask)

        self.old_features_dc.append(self._features_dc)
        self.old_features_rest.append(self._features_rest)
        self.old_opacity.append(self._opacity)
        self.old_scaling.append(self._scaling)
        self.old_rotation.append(self._rotation)
        
        if self.optimizer is None:
            self._xyz = self._xyz[mask]
            # self._mask = self._mask[mask]

            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._opacity = self._opacity[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]

        else:
            optimizable_tensors = self._prune_optimizer(mask)

            self._xyz = optimizable_tensors["xyz"]

            # self._mask = optimizable_tensors["mask"]

            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]

            self.denom = self.denom[mask]

        # print(self.segment_times, torch.unique(self._mask))
        self.segment_times += 1
        tmp = self._mask[self._mask == self.segment_times]
        tmp[mask] += 1
        self._mask[self._mask == self.segment_times] = tmp

        

        # print(self._mask[self._mask == self.segment_times][mask].shape)
        # print(self.segment_times, torch.unique(self._mask), torch.unique(mask))
        
    def roll_back(self):
        try:
            self._xyz = self.old_xyz.pop()
            # self._mask = self.old_mask.pop()

            self._features_dc = self.old_features_dc.pop()
            self._features_rest = self.old_features_rest.pop()
            self._opacity = self.old_opacity.pop()
            self._scaling = self.old_scaling.pop()
            self._rotation = self.old_rotation.pop()

            
            self._mask[self._mask == self.segment_times+1] -= 1
            self.segment_times -= 1
        except:
            pass
    
    @torch.no_grad()
    def clear_segment(self):
        try:
            self._xyz = self.old_xyz[0]
            # self._mask = self.old_mask[0]

            self._features_dc = self.old_features_dc[0]
            self._features_rest = self.old_features_rest[0]
            self._opacity = self.old_opacity[0]
            self._scaling = self.old_scaling[0]
            self._rotation = self.old_rotation[0]

            self.old_xyz = []
            self.old_mask = []

            self.old_features_dc = []
            self.old_features_rest = []
            self.old_opacity = []
            self.old_scaling = []
            self.old_rotation = []

            self.segment_times = 0
            self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")
        except:
            # print("Roll back failed. Please run gaussians.segment() first.")
            pass

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                # i think the different values here is why the original and cloned tensors evovles differently in the course of subsequent optimziation.

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                # why the store_state operation on  `exp_avg` and `exp_avg_sq` is not needed here?

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # register new changes to class instance variables
        d = {"xyz": new_xyz,
        # "mask": new_mask,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        # self._mask = optimizable_tensors["mask"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # regions with high variance as described in the paper.
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # repeat N times along dimension 1.
        means =torch.zeros((stds.size(0), 3),device="cuda") # means are zerolike matrix.
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # in the area surrounding old xyz, sample from a normal distribution.
        # When you apply unsqueeze(-1) to the output tensor of torch.normal, 
        # you're essentially adding a new dimension of size 1 to the last dimension of the generated random tensor.
        # the new_xyz is of shape (N, 3, 1)


        # new_mask = self._mask[selected_pts_mask].repeat(N,1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # regions `under-constructed` as described in the paper.
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) # it is `norm` not `normalize`.
        # creates a mask (selected_pts_mask) based on a condition 
        # where the Euclidean norm of grads along the last dimension is compared against grad_threshold. 
        # If the norm is greater than or equal to grad_threshold, the corresponding entry in the mask is set to True; 
        # otherwise, it is set to False.
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        # new_mask = self._mask[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Au contriare de my guess, the clone and split is not done by deplicating a new GaussianModel class instance;
        # they just catenate the tensors to the original ones.

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1