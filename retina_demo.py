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

def get_dataset_triangles_noise(
    split=None, data_size=128, hollow=False, samples=6000, batch_size=128
):
    x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
    x = np.reshape(x, (1, data_size, data_size))
    y = np.reshape(y, (1, data_size, data_size))
    cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
    theta = np.random.random((samples, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((samples, 1, 1)) > 0.5

    triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(np.pi / 3) / np.cos(
        (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
    )

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
    if hollow:
        triangles = 1 - triangles**2
        circles = 1 - circles**2

    images = isTriangle * triangles + (1 - isTriangle) * circles 

    images = images + np.random.randn(*images.shape) * .3

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(images, 1)))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2



#ds, _ = icon_registration.data.get_dataset_mnist(split="train", number=5)
#ds, _ = icon_registration.data.get_dataset_triangles(split="train", hollow=True, data_size=30)
ds1, ds2 = icon_registration.data.get_dataset_retina(fixed_vertical_offset=100)#100

#ds, _ = get_dataset_triangles_noise(split="train", hollow=True, data_size=30)

sample_batch = next(iter(ds1))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])

import icon_registration as icon

class ICONSquaringVelocityField(icon.RegistrationModule):
   def __init__(self, net):
       super().__init__()
       self.net = net
       self.n_steps = 10

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

def evaluate(net, prefix):
  net.similarity = icon.LNCC(sigma=5)
  net.train()
  net.to(device)

  optim = torch.optim.Adam(net.parameters(), lr=0.0005)
  curves = icon.train_datasets(net, optim, ds1, ds2, epochs=16)
  for i, name in enumerate(curves[0]._fields):

    plt.subplot(2, 3, i + 1)
    plt.plot([getattr(c, name) for c in curves])
    if name=="inverse_consistency_loss":
      name = "bending_energy_loss"
    plt.title(name)
  plt.tight_layout()
  plt.savefig(footsteps.output_dir + prefix + "MNIST_curves.pdf")
  plt.show()
  def show(tensor):
    plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach())
    plt.xticks([])
    plt.yticks([])
  image_A = next(iter(ds1))[0].to(device)[:6]
  image_B = next(iter(ds2))[0].to(device)[:6]
  with torch.no_grad():
    print(net(image_A, image_B))
    try:
      net.prepare_for_viz(image_A, image_B)
    except:
      pass
  plt.subplot(2, 2, 1)
  plt.title("Moving Images")
  show(image_A)
  plt.subplot(2, 2, 2)
  plt.title("Fixed Images")
  show(image_B)
  plt.subplot(2, 2, 3)
  plt.title("Warped Images")
  show(net.warped_image_A)
  plt.contour(torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach())
  plt.contour(torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach())
  plt.subplot(2, 2, 4)
  plt.title("Difference Images")
  show(net.warped_image_A - image_B)
  plt.tight_layout()
  plt.savefig(footsteps.output_dir_impl + prefix + "MNIST.pdf")
  plt.show()
  plt.title("Detail of deformations")

  show(net.warped_image_A * 0)
  plt.contour(torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach(), levels=np.linspace(0, 1, 35))
  plt.contour(torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach(), levels=np.linspace(0, 1, 35))
  plt.savefig(footsteps.output_dir + prefix + "MNIST_deformations.pdf")
  plt.show()
  plt.title("Composition of A->B and B->A transforms")
  plt.contour(net.phi_AB(net.phi_BA(net.identity_map))[0, 0].cpu().detach(), levels=35)
  plt.contour(net.phi_AB(net.phi_BA(net.identity_map))[0, 1].cpu().detach(), levels=35)
  plt.savefig(footsteps.output_dir + prefix + "MNIST composition.pdf")
  return curves

lmbda = 0.001

inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=2))
inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=2))
inner_net3 = ICONSquaringVelocityField(networks.tallUNet2(dimension=2))

threestep_consistent_net = icon.losses.BendingEnergyNet(
    UnwrapHalfwaynet(
        TwoStepInverseConsistent(inner_net, 
             TwoStepInverseConsistent(inner_net2, inner_net3))
    )
    , icon.ssd, lmbda=lmbda)
threestep_consistent_net.assign_identity_map(sample_batch.shape)
threestep_consistent_curves = evaluate(threestep_consistent_net, "")

inner_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=2))
inner_net2 = ICONSquaringVelocityField(networks.tallUNet2(dimension=2))

twostep_consistent_net = icon.losses.BendingEnergyNet(
    UnwrapHalfwaynet(
        TwoStepInverseConsistent(inner_net, inner_net2)
    )
    , icon.ssd, lmbda=lmbda)
twostep_consistent_net.assign_identity_map(sample_batch.shape)
twostep_consistent_curves = evaluate(twostep_consistent_net, "")

onestep_consistent_net = ICONSquaringVelocityField(networks.tallUNet2(dimension=2))

onestep_consistent_net = icon.losses.BendingEnergyNet(
    UnwrapHalfwaynet(
        onestep_consistent_net
    )
    , icon.ssd, lmbda=lmbda)
onestep_consistent_net.assign_identity_map(sample_batch.shape)
onestep_consistent_curves = evaluate(onestep_consistent_net, "")

inner_net = icon.network_wrappers.SquaringVelocityField(networks.tallUNet2(dimension=2))
inner_net2 = icon.network_wrappers.SquaringVelocityField(networks.tallUNet2(dimension=2))

regular_net = icon.losses.BendingEnergyNet(
        icon.TwoStepRegistration(inner_net, inner_net2)
    , icon.ssd, lmbda=lmbda)
regular_net.assign_identity_map(sample_batch.shape)
twostep_regular_curves = evaluate(regular_net, "")

inner_net = icon.network_wrappers.SquaringVelocityField(networks.tallUNet2(dimension=2))

regular_net = icon.losses.BendingEnergyNet(
        inner_net
    , icon.ssd, lmbda=lmbda)
regular_net.assign_identity_map(sample_batch.shape)
onestep_regular_curves = evaluate(regular_net, "")

inner_net = icon.network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=2))
inner_net2 = icon.network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=2))

net = icon.losses.GradientICON(
        icon.TwoStepRegistration(inner_net, inner_net2)
    , icon.ssd, lmbda=3.5)
net.assign_identity_map(sample_batch.shape)
twostep_ddf_gradicon = evaluate(net, "")

for i, name in enumerate(twostep_consistent_curves[0]._fields):

    plt.subplot(2, 3, i + 1)
    plt.plot([getattr(c, name) for c in threestep_consistent_curves])
    plt.plot([getattr(c, name) for c in twostep_consistent_curves])
    plt.plot([getattr(c, name) for c in onestep_consistent_curves])
    plt.plot([getattr(c, name) for c in twostep_regular_curves])
    plt.plot([getattr(c, name) for c in onestep_regular_curves])
    plt.plot([getattr(c, name) for c in twostep_ddf_gradicon])

    if name=="inverse_consistency_loss":
      name = "bending_energy_loss"
    plt.title(name)
plt.subplot(2, 3, 6)
plt.plot([[0, 0, 0, 0, 0, 0]])
plt.legend(["three cons", "two cons", "one cons", "two svf", "one svf", "two gradicon"])
plt.savefig(footsteps.output_dir + "oonf.pdf")


