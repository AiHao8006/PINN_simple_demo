'''
To solve Poisson's equation with PINN.

Poisson's equation: (d^2/dx^2 + d^2/dy^2) \phi = \rho(x, y)
\rho(x, y) = 1.5 for all x, y

Domain: x \in [-1, 1], y \in [-1, 1],
Boundary conditions: \phi(x, -1) = cos(pi * x / 2), \phi(x, 1) = \phi(\pm 1, y) = 0

In PINN, we define: f = (d^2/dx^2 + d^2/dy^2) \phi - \rho(x, y) -> 0
'''
import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from used_funcs import *

path = ""

# figure size
plt.figure(figsize=(12, 6))
# figure location in the screen
plt.get_current_fig_manager().window.wm_geometry('+10+10')

model = PINN(hidden_layers=100)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list = []
for epoch in range(1001):
    # Poisson's equation
    batch_r = batch_get_random_xy(12800, location=None, requires_grad=True)
    batch_phi = model(batch_r)
    # partial derivatives
    batch_phi_r = torch.autograd.grad(batch_phi, batch_r, create_graph=True, grad_outputs=torch.ones_like(batch_phi))[0]
    batch_phi_xx = torch.autograd.grad(batch_phi_r[:, 0], batch_r, create_graph=True, grad_outputs=torch.ones_like(batch_phi_r[:, 0]))[0][:, 0].reshape(-1, 1)
    batch_phi_yy = torch.autograd.grad(batch_phi_r[:, 1], batch_r, create_graph=True, grad_outputs=torch.ones_like(batch_phi_r[:, 1]))[0][:, 1].reshape(-1, 1)
    # get the loss term 1, named loss_f
    # batch_f = batch_phi_xx + batch_phi_yy - 1.5   # \rho(x, y) = 1.5
    batch_f = batch_phi_xx + batch_phi_yy + 1.5
    loss_f = batch_f.pow(2)

    # lower boundary
    batch_r_lower = batch_get_random_xy(640, location='lower boundary', requires_grad=False)
    batch_phi_lower = model(batch_r_lower)
    loss_lower = (batch_phi_lower - torch.cos(batch_r_lower[:, 0].reshape(-1, 1) * math.pi / 2)).pow(2)
    # upper boundary
    batch_r_upper = batch_get_random_xy(640, location='upper boundary', requires_grad=False)
    batch_phi_upper = model(batch_r_upper)
    loss_upper = batch_phi_upper.pow(2)
    # left boundary
    batch_r_left = batch_get_random_xy(640, location='left boundary', requires_grad=False)
    batch_phi_left = model(batch_r_left)
    loss_left = batch_phi_left.pow(2)
    # right boundary
    batch_r_right = batch_get_random_xy(640, location='right boundary', requires_grad=False)
    batch_phi_right = model(batch_r_right)
    loss_right = batch_phi_right.pow(2)

    # total loss
    loss = torch.cat((loss_f, loss_lower, loss_upper, loss_left, loss_right)).mean()
    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    if epoch % 10 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))
        plot_results(model, loss_list, plot=False, save=False, save_path=path + 'results.png')
        # torch.save(model.state_dict(), path + 'model.pth')
plt.show()

