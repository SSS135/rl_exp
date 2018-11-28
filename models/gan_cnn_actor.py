import math
from itertools import chain

import torch
from ppo_pytorch.common.make_grid import make_grid
from ppo_pytorch.models.actors import Actor, ActorOutput
from ppo_pytorch.models.utils import image_to_float, make_conv_heatmap
from torch import nn as nn, autograd as autograd
from torch.autograd import Variable


class GanCNNActor(Actor):
    """
    Convolution network.
    """

    def __init__(self, obs_space, action_space, head_factory, cnn_kind='large',
                 activation=nn.ReLU, actor_includes_conv=True, **kwargs):
        """
        Args:
            obs_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            cnn_kind: Type of cnn.
                'small' - small CNN from arxiv DQN paper (Mnih et al. 2013)
                'large' - bigger CNN from Nature DQN paper (Mnih et al. 2015)
                'custom' - largest CNN of custom structure
            activation: Activation function
        """
        super().__init__(obs_space, action_space, **kwargs)
        self.activation = activation
        self.actor_includes_conv = actor_includes_conv
        assert cnn_kind in ('small', 'large', 'custom')

        ngf, ndf = 64, 64
        nz = 64
        nc = 4
        self.disc_start_net = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 0, bias=True),
            ),
        )
        self.disc_end_net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
        )
        self.gen_start_net = nn.Sequential(
            nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=True),
        )
        self.gen_end_net = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            ),
        )

        # create head
        self.actor_linear = nn.Sequential(
            nn.Linear(nz, nz*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz*2, nz*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = head_factory(nz*2, self.pd)

        self.actor_linear_head = nn.Sequential(
            self.actor_linear,
            self.head,
        )

        self.reset_weights()

    def params_disc(self):
        return chain(self.disc_start_net.parameters(), self.disc_end_net.parameters())

    def params_gen(self):
        return chain(self.gen_start_net.parameters(), self.gen_end_net.parameters())

    def params_head(self):
        return chain(self.head.parameters(), self.actor_linear.parameters())

    def params_actor(self):
        if self.actor_includes_conv:
            return chain(self.disc_start_net.parameters(), self.params_head())
        else:
            return self.params_head()

    def reset_weights(self):
        super().reset_weights()
        self.head.reset_weights()

    def forward_on_hidden_code(self, hidden):
        return self.actor_linear_head(hidden)

    def generate(self, hidden_code, state):
        x = hidden_code.contiguous().view(*hidden_code.shape, 1, 1)
        for i, conv in enumerate(self.gen_end_net):
            # run conv layer
            x = conv(x)

        # log
        if self.do_log:
            state = image_to_float(state.data)
            img = torch.cat([x.data.unsqueeze(2), state.unsqueeze(2)], 2)
            img = img[:min(img.shape[0], 2)]
            img = img.view(-1, 1, *img.shape[3:])
            img = img.clamp(-1, 1).div_(2).add_(0.5)
            img = make_grid(img, nrow=round(math.sqrt(img.shape[0])), normalize=False, fill_value=0.1)
            self.logger.add_image('reconstructed state', img, self._step)

        return x

    def discriminate(self, state, detach_hidden_from_conv=False):
        x = state
        for i, layer in enumerate(self.disc_start_net):
            # run conv layer
            x = layer(x)
            # log
            if self.do_log:
                self.log_conv_activations(i, layer[0], x)
                self.log_conv_filters(i, layer[0])

        # if self.do_log:
        #     self.logger.add_histogram('conv linear', x, self._step)

        conv_out = x
        disc = self.disc_end_net(x).view(-1)
        hidden = self.gen_start_net(x.detach() if detach_hidden_from_conv else x).view(x.shape[0], -1)
        return disc, hidden, conv_out

    def forward(self, input) -> ActorOutput:
        log_policy_attention = self.do_log and input.is_leaf
        input = image_to_float(input)
        if log_policy_attention:
            input = Variable(input.data, requires_grad=True)

        disc, hidden, conv_out = self.discriminate(input)

        # hidden = self.conv_actor_linear(conv_out)

        ac_out = self.actor_linear_head(hidden)
        ac_out.conv_out = conv_out
        ac_out.hidden_code = hidden
        ac_out.gan_d = disc

        if log_policy_attention:
            self.log_policy_attention(input, ac_out)

        return ac_out

    def log_conv_activations(self, index: int, conv: nn.Conv2d, x: Variable):
        img = x[0].data.unsqueeze(1).clone()
        img = make_conv_heatmap(img)
        img = make_grid(img, nrow=round(math.sqrt(conv.out_channels)), normalize=False, fill_value=0.1)
        self.logger.add_image('conv activations {} img'.format(index), img, self._step)
        self.logger.add_histogram('conv activations {} hist'.format(index), x[0], self._step)

    def log_conv_filters(self, index: int, conv: nn.Conv2d):
        channels = conv.in_channels * conv.out_channels
        shape = conv.weight.data.shape
        kernel_h, kernel_w = shape[2], shape[3]
        img = conv.weight.data.view(channels, 1, kernel_h, kernel_w).clone()
        max_img_size = 100 * 5
        img_size = channels * math.sqrt(kernel_h * kernel_w)
        if img_size > max_img_size:
            channels = channels * (max_img_size / img_size)
            channels = math.ceil(math.sqrt(channels)) ** 2
            img = img[:channels]
        img = make_conv_heatmap(img, scale=2 * img.std())
        img = make_grid(img, nrow=round(math.sqrt(channels)), normalize=False, fill_value=0.1)
        self.logger.add_image('conv featrues {} img'.format(index), img, self._step)
        self.logger.add_histogram('conv features {} hist'.format(index), conv.weight.data, self._step)

    def log_policy_attention(self, states, head_out):
        states_grad = autograd.grad(
            head_out.probs.abs().mean() + head_out.state_values.abs().mean(), states,
            only_inputs=True, retain_graph=True)[0]
        img = states_grad.data[:4]
        img.abs_()
        img /= img.view(4, -1).pow(2).mean(1).sqrt_().add_(1e-5).view(4, 1, 1, 1)
        img = img.view(-1, 1, *img.shape[2:]).abs()
        # img = make_conv_heatmap(img, scale=2*img.std())
        img = make_grid(img, 4, normalize=True, fill_value=0.1)
        self.logger.add_image('state attention', img, self._step)