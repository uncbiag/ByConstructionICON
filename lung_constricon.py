
import random
import os
import torch
import numpy as np
import torch.nn.functional as F

import footsteps

import icon_registration as icon
import icon_registration.networks as networks
import footsteps
footsteps.initialize()
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils


import icon_registration as icon

class ICONSquaringVelocityField(icon.RegistrationModule):
   def __init__(self, net):
       super().__init__()
       self.net = net
       self.n_steps = 7

   def forward(self, image_A, image_B):
       velocityfield_delta_a = (
           self.net(image_A, image_B) 
           -self.net(image_B, image_A))/ 2**self.n_steps 
       velocityfield_delta = velocityfield_delta_a

       for _ in range(self.n_steps):
         velocityfield_delta = velocityfield_delta + self.as_function(
             velocityfield_delta)(velocityfield_delta + self.identity_map)
       def transform_AB(coordinate_tensor):
           coordinate_tensor = coordinate_tensor + self.as_function(velocityfield_delta)(coordinate_tensor)
           return coordinate_tensor

       velocityfield_delta2 = -velocityfield_delta_a

       for _ in range(self.n_steps):
         velocityfield_delta2 = velocityfield_delta2 + self.as_function(
             velocityfield_delta2)(velocityfield_delta2 + self.identity_map)
       def transform_BA(coordinate_tensor):
           coordinate_tensor = coordinate_tensor + self.as_function(velocityfield_delta2)(coordinate_tensor)
           return coordinate_tensor
       return transform_AB, transform_BA
  
class TwoStepInverseConsistent(icon.RegistrationModule):
  def __init__(self, phi, psi):
    super().__init__()
    self.netPhi = phi
    self.netPsi = psi
  def forward(self, image_A, image_B):
    root_phi_AB, root_phi_BA = self.netPhi(image_A, image_B)
    # image_A \circ root_phi_AB \circ root_phi_AB ~ image_B
    # phi_AB = root_phi_AB \circ root_phi_AB

    # image_A \circ root_phi_AB ~ image_B \circ root_phi_BA

    A_tilde = self.as_function(image_A)(root_phi_AB(self.identity_map))
    B_tilde = self.as_function(image_B)(root_phi_BA(self.identity_map))

    psi_AB, psi_BA = self.netPsi(A_tilde, B_tilde)

    # A_tilde \circ root_psi_AB ~ B_tilde \circ root_psi_BA

    # expand A_tilde

    # image_A \circ root_phi_AB \circ root_psi_AB ~ image_B \circ root_phi_BA \circ root_psi_BA

    # right compose with root_psi_AB

    # image_A \circ root_phi_AB \circ root_psi_AB \circ root_psi_AB ~ image_B \circ root_phi_BA \circ root_psi_BA \circ root_psi_AB

    # \circ root_psi_AB cancels \circ root_psi_BA on the right side

    # image_A \circ root_phi_AB \circ root_psi_AB \circ root_psi_AB ~ image_B \circ root_phi_BA

    # right compose with root_psi_AB, cancels on the right side again

    # image_A \circ root_phi_AB \circ root_psi_AB \circ root_psi_AB \circ root_phi_AB ~ image_B 

    #               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #               here is our final AB transform


    return (
        lambda coord: root_phi_AB(psi_AB(root_phi_AB(coord))),
        lambda coord: root_phi_BA(psi_BA(root_phi_BA(coord)))
    )

class UnwrapHalfwaynet(icon.RegistrationModule):
  def __init__(self, net):
    super().__init__()
    self.net = net
  def forward(self, image_A, image_B):
    AB, BA = self.net(image_A, image_B)
    return AB


input_shape = [1, 1, 175, 175, 175]

lmbda = 0.001

inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))
inner_net3 = ICONSquaringVelocityField(networks.tallUNet2(dimension=3))

threestep_consistent_net = icon.losses.BendingEnergyNet(
    UnwrapHalfwaynet(
        TwoStepInverseConsistent(inner_net, 
             TwoStepInverseConsistent(inner_net2, inner_net3))
    )
    , icon.LNCC(sigma=5), lmbda=lmbda)
threestep_consistent_net.assign_identity_map(input_shape)


net_par = torch.nn.DataParallel(threestep_consistent_net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

net_par.train()

BATCH_SIZE=3
GPUS =3
WITH_AUGMENT = True

class lung_dataloader():
    def __init__(self, data_path, scale, batch_size, with_augment=False) -> None:
        self.current_ite = 0
        self.batch_size = batch_size
        self.with_augment = with_augment

        img = torch.load(f"{data_path}/lungs_train_{scale}_scaled", map_location='cpu')
        mask = torch.load(f"{data_path}/lungs_seg_train_{scale}_scaled", map_location='cpu')

        self.data = torch.stack([(torch.cat(d, 0)+1)*torch.cat(m, 0) for d,m in zip(img, mask)], dim=0)
        self.current_idx_list = np.arange(self.data.shape[0])
        self._shuffle()
    
    def _shuffle(self) -> None:
        np.random.shuffle(self.current_idx_list)

    def make_batch(self):
        if self.current_ite + self.batch_size > len(self.data):
            self._shuffle()
            self.current_ite = 0
        
        batch = self.data[self.current_idx_list[self.current_ite:self.current_ite+self.batch_size]]
        for i in range(self.batch_size):
            if random.random() > .5:
                temp = batch[i, 0]
                batch[i, 0] = batch[i, 1]
                batch[i, 1] = temp
        self.current_ite += self.batch_size
        
        if self.with_augment:
            return augment(batch[:, 0].cuda(), batch[:,1].cuda())
        else:
            return batch[:, 0].cuda(), batch[:,1].cuda()


def augment(image_A, image_B):
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.Tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1 
        identity = identity * (torch.randint_like(identity, 0, 2) * 2  - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    
    noise = torch.randn((image_A.shape[0], 3, 4))

    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
   
    warped_A = F.grid_sample(image_A, forward_grid, padding_mode='border')

    noise = torch.randn((image_A.shape[0], 3, 4))
    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
    warped_B = F.grid_sample(image_B, forward_grid, padding_mode='border')

    return warped_A, warped_B


dataloader = lung_dataloader(
    "/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI",
    scale="2xdown",
    batch_size = GPUS * BATCH_SIZE,
    with_augment=WITH_AUGMENT
)

icon.train_batchfunction(net_par, optimizer, dataloader.make_batch, unwrapped_net=threestep_consistent_net, steps=50000)

