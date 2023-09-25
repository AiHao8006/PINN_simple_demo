import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_results(model, loss_list, rho_list=None, plot=True, save=False, save_path='results.png'):
    '''
    plot 2 subfigures, the loss curve and the potential contour
    '''
    # plot the loss curve
    plt.clf()
    plt.subplot(1, 2, 1)
    if rho_list is None:
        plt.plot(loss_list)
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
    else:
        # plot loss in log scale (left), plot rho in linear scale (right), the two curves are in the same subfigure
        plt.plot(loss_list)
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss'], loc='upper left', bbox_to_anchor=(0.1, 0.6))
        plt.twinx()
        plt.plot(rho_list, color='red')
        plt.xlabel('epoch')
        plt.ylabel('rho')
        # add a vertical line to indicate the epoch when rho is close to the true value
        plt.axhline(y=1.5, color='red', linestyle='--')
        # ledgend
        plt.legend(['rho', 'true rho'], loc='upper right', bbox_to_anchor=(0.9, 0.6))

    # plot the potential contour
    plt.subplot(1, 2, 2)
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    phi = model(xy)
    phi = phi.reshape(100, 100).detach().numpy()
    plt.contourf(X, Y, phi, 100, cmap=plt.cm.hot)
    C = plt.contour(X, Y, phi, 10, colors='black')
    plt.clabel(C, inline=True, fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    if plot:
        plt.pause(0.1)
    if save:
        plt.savefig(save_path)

class PINN(nn.Module):
    def __init__(self, inputs=2, outputs=1, hidden_layers=100) -> None:
        super(PINN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(inputs, hidden_layers),
            nn.Tanh())
        self.block = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers),
            nn.Tanh(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.Tanh())
        self.output_layer = nn.Linear(hidden_layers, outputs)
    def forward(self, xy):
        layer = self.input_layer(xy)
        layer = self.block(layer) + layer
        layer = self.block(layer) + layer
        layer = self.block(layer) + layer
        phi = self.output_layer(layer)
        return phi
    
def batch_get_random_xy(batch_size, location=None, requires_grad=True):
    if location == 'lower boundary':
        xy = torch.rand((batch_size, 2), requires_grad=requires_grad) * 2 - 1
        xy[:, 1] = -1
    elif location == 'upper boundary':
        xy = torch.rand((batch_size, 2), requires_grad=requires_grad) * 2 - 1
        xy[:, 1] = 1
    elif location == 'left boundary':
        xy = torch.rand((batch_size, 2), requires_grad=requires_grad) * 2 - 1
        xy[:, 0] = -1
    elif location == 'right boundary':
        xy = torch.rand((batch_size, 2), requires_grad=requires_grad) * 2 - 1
        xy[:, 0] = 1
    else:
        xy = torch.rand((batch_size, 2), requires_grad=requires_grad) * 2 - 1
    return xy