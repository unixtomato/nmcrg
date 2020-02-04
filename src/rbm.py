import numpy as np

import torch

import os
import sys
import timeit
import shutil
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('... device on {}'.format(device), file=sys.stderr)

class RBM(torch.nn.Module):
    def __init__(self, input_width, filter_width):
        super().__init__()

        self.pad_width = filter_width - 2 if input_width % 2 == 0 else filter_width - 1

        W = np.asarray(
            np.random.uniform(low=-1e-5, high=1e-5, size=(1, 1, filter_width, filter_width)),
            dtype=np.float32
        )

        self.W = torch.nn.Parameter(torch.tensor(W, requires_grad=True))


    def periodic_padding(self, v):
        v = torch.cat([v, v[..., :, :self.pad_width]], axis=3)
        v = torch.cat([v, v[..., :self.pad_width, :]], axis=2)
        return v

    def free_energy(self, v_sample):
        v_sample = self.periodic_padding(v_sample)
        wx_b = torch.nn.functional.conv2d(v_sample, self.W, stride=2)
        hidden_term = torch.mean(torch.log(torch.exp(-wx_b) + torch.exp(wx_b)))
        return -hidden_term

    def propup(self, vis):
        vis = self.periodic_padding(vis)
        pre_sigmoid_activation = torch.nn.functional.conv2d(vis, self.W, stride=2) * 2
        return [pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = torch.bernoulli(h1_mean)
        h1_sample = 2 * h1_sample - 1
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        # transposed convolutoin
        pre_sigmoid_activation = torch.nn.functional.conv_transpose2d(hid, self.W, stride=2) * 2

        # periodic reduce summing
        pre_sigmoid_activation[..., :self.pad_width, :].add_(pre_sigmoid_activation[..., -self.pad_width:, :])
        pre_sigmoid_activation[..., :, :self.pad_width].add_(pre_sigmoid_activation[..., :, -self.pad_width:])
        pre_sigmoid_activation = pre_sigmoid_activation[..., :-self.pad_width, :-self.pad_width]
        
        return [pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = torch.bernoulli(v1_mean)
        v1_sample = 2 * v1_sample - 1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def contrastive(self, input, gibbs_steps):
        v1_sample = input
        for k in range(gibbs_steps):
            [pre_sigmoid_h1, h1_mean, h1_sample, 
             pre_sigmoid_v1, v1_mean, v1_sample] = self.gibbs_vhv(v1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def forward(self, input, gibbs_steps=3):

        with torch.no_grad():
            [
                pre_sigmoid_nh, 
                nh_mean, 
                nh_sample,
                pre_sigmoid_nv, 
                nv_mean, 
                nv_sample
            ] = self.contrastive(input, gibbs_steps)

        return nv_sample, pre_sigmoid_nv


def load_data(fname, input_width, root="datasets", dtype=np.int):

    print('... loading data', file=sys.stderr)
    
    with open('{}/{}'.format(root, fname), "r") as f:
        data = np.fromfile(f, dtype=dtype)

    data = data.astype(np.float32).reshape((-1, 1, input_width, input_width))
    data = torch.tensor(data, device=device)
    return data


def test_rbm(
    input_width,
    dataset,
    data_folder,
    plot_folder,
    filter_width=8,
    batch_size=50,
    gibbs_steps=3,
    training_epochs=100,
    starter_learning_rate=0.001
):

    # load data
    data = load_data(dataset, input_width, dtype=np.int32, root=data_folder)
    n_train_batches = data.shape[0] // batch_size

    # build rbm models
    rbm = RBM(input_width, filter_width).to(device)

    # build optimizer
    optimizer = torch.optim.Adam(rbm.parameters(), lr=starter_learning_rate)

    # monitor cost with cross entropy
    cross_entropy = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.ones((1, input_width, input_width), device=device),
        reduction='none'
    )


    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):

        # go through the mini batches
        mean_cost = []

        for index in range(n_train_batches):

            input = data[index * batch_size: (index + 1) * batch_size]
            chain_end, pre_sigmoid = rbm(input, gibbs_steps)
            
            cost = rbm.free_energy(input) - rbm.free_energy(chain_end)
            cost.backward()

            optimizer.step()
            optimizer.zero_grad()

            monitoring_cost = torch.mean(torch.sum(cross_entropy(pre_sigmoid, (input + 1) / 2), dim=(1,2,3)))
            mean_cost.append(monitoring_cost.detach().cpu().numpy())


        # plot filters after each training epoch
        plt.imshow(rbm.W.detach().cpu().numpy().reshape(filter_width, filter_width), 'gray')
        plt.axis('off')
        plt.colorbar()
        plt.savefig('{}/filters_at_epoch_{}.png'.format(plot_folder, epoch))
        plt.clf()

        # save parameters
        with open('{}/models_at_epoch_{}.pkl'.format(plot_folder, epoch), 'wb') as f:
            params = [param.detach().cpu().numpy() for key, param in rbm.state_dict().items()]
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('epoch {}, cost is {:.4f}'.format(epoch, np.mean(mean_cost)))

    print('time elapsed {}'.format(timeit.default_timer() - start_time))


