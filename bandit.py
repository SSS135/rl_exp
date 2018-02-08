from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import torch
import torch.optim as optim
from scipy.ndimage.filters import gaussian_filter
from torch.autograd import Variable

FT = lambda x: torch.FloatTensor(np.asarray(x, dtype=np.float32))
FTC = lambda x: FT(x).cuda()
T = torch.from_numpy
V = Variable
VG = partial(Variable, requires_grad=True)
NP = lambda x: (x.data.cpu() if isinstance(x, Variable) else x.cpu()).numpy()


np.seterr('raise')


class Bandit:
    def __init__(self, size, batches, drift=0):
        self.size = size
        self.batches = batches
        self.drift = drift
        self.mean = rng.randn(batches, size)
        self.std = np.ones((batches, size)) # 0.5 + rng.rand(batches, size)
        self.br = np.arange(batches)

    def step(self, arm):
        assert np.shape(arm) == (self.batches,)
        assert np.all((0 <= arm) & (arm < self.size))

        r = rng.normal(self.mean[self.br, arm], self.std[self.br, arm])
        if self.drift != 0:
            self.mean += self.drift * rng.randn(self.batches, self.size)
        return r


def optimistic_init_solver(bandit, steps, eps=0.1, init_reward=0):
    avg_reward = init_reward * np.ones((bandit.batches, bandit.size))
    num_arm_pull = np.ones((bandit.batches, bandit.size))
    pulled_arms = np.zeros((bandit.batches, steps))
    br = np.arange(bandit.batches)
    for i in range(steps):
        arm = np.argmax(avg_reward, axis=1)

        rand_mask = rng.rand(bandit.batches) < eps
        rand_count = np.sum(rand_mask)
        if rand_count > 0:
            arm[rand_mask] = rng.randint(bandit.size, size=rand_count)

        r = bandit.step(arm)
        avg_reward[br, arm] = (avg_reward[br, arm] * num_arm_pull[br, arm] + r) / (num_arm_pull[br, arm] + 1)
        pulled_arms[:, i] = arm
        num_arm_pull[br, arm] += 1
    return pulled_arms


def simple_solver(bandit, steps, eps=0.1):
    avg_reward = np.zeros((bandit.batches, bandit.size))
    num_arm_pull = np.zeros((bandit.batches, bandit.size))
    pulled_arms = np.zeros((bandit.batches, steps))
    br = np.arange(bandit.batches)
    for i in range(steps):
        arm = np.argmax(avg_reward, axis=1)

        rand_mask = rng.rand(bandit.batches) < eps
        rand_count = np.sum(rand_mask)
        if rand_count > 0:
            arm[rand_mask] = rng.randint(bandit.size, size=rand_count)

        r = bandit.step(arm)
        avg_reward[br, arm] = (avg_reward[br, arm] * num_arm_pull[br, arm] + r) / (num_arm_pull[br, arm] + 1)
        pulled_arms[:, i] = arm
        num_arm_pull[br, arm] += 1
    return pulled_arms


def ucb_solver(bandit, steps, c=1):
    avg_reward = np.zeros((bandit.batches, bandit.size))
    num_arm_pull = np.zeros((bandit.batches, bandit.size))
    pulled_arms = np.zeros((bandit.batches, steps))
    br = np.arange(bandit.batches)
    for i in range(steps):
        ucb_q = avg_reward + c * np.sqrt(np.log(i + 1) / (num_arm_pull + 1e-7))
        arm = np.argmax(ucb_q, axis=1)
        r = bandit.step(arm)
        avg_reward[br, arm] = (avg_reward[br, arm] * num_arm_pull[br, arm] + r) / (num_arm_pull[br, arm] + 1)
        pulled_arms[:, i] = arm
        num_arm_pull[br, arm] += 1
    return pulled_arms


def grad_solver(bandit, steps, eps=0.1, lr=0.1, init_reward=0):
    avg_reward = init_reward * np.ones((bandit.batches, bandit.size))
    pulled_arms = np.zeros((bandit.batches, steps))
    br = np.arange(bandit.batches)
    for i in range(steps):
        arm = np.argmax(avg_reward, axis=1)

        rand_mask = rng.rand(bandit.batches) < eps
        rand_count = np.sum(rand_mask)
        if rand_count > 0:
            arm[rand_mask] = rng.randint(bandit.size, size=rand_count)

        r = bandit.step(arm)
        avg_reward[br, arm] += lr * (r - avg_reward[br, arm])
        pulled_arms[:, i] = arm
    return pulled_arms


# std_err = []
# std_std_err = []
# avg_err = []
# # poisson_data = None
#
#
# def reset_stats():
#     global poisson_data, std_err, std_std_err, avg_err
#     std_err = []
#     std_std_err = []
#     avg_err = []
    # poisson_data = FT(rng.poisson(1, 64*1024*1024))


# reset_stats()


# def get_poisson(size, batches):
#     i = rng.randint(poisson_data.size - size, size=batches)
#     return poisson_data[i: i+size]


def ucuc_grad_solver(bandit, steps, lam=0.1, init_reward=1.5, uc=0, ucuc=15, heads=100, bs2_heads=100, p=0.3,
                     uc_var=False, ucuc_var=False, bernouli=False):
    avg_reward = init_reward * torch.randn((bandit.batches, bandit.size, heads)).cuda()
    avg_reward = VG(avg_reward)
    pulled_arms = torch.zeros(bandit.batches, steps)
    br = torch.arange(0, bandit.batches).long().cuda()
    sr = torch.arange(0, bandit.size).long().cuda()
    sgd = optim.SGD([avg_reward], lr=lam)
    rand_idx_store = torch.LongTensor(64 * 1024 * 1024).random_(heads).cuda()

    if ucuc == 0:
        bs2_std = V(torch.zeros(1).cuda())

    def get_rand_idx(*sizes):
        count = np.prod(sizes)
        start = rng.randint(0, len(rand_idx_store) - count)
        return rand_idx_store[start:start + count].view(*sizes)

    for i in range(steps):
        r_std = (torch.var if uc_var else torch.std)(avg_reward, -1)
        r_mean = torch.mean(avg_reward, -1)
        if ucuc != 0:
            bs2_idx = get_rand_idx(bandit.batches, bandit.size, heads * bs2_heads)
            bs2 = avg_reward.data.gather(-1, bs2_idx)
            bs2 = bs2.view(bandit.batches, bandit.size, heads, bs2_heads)
            bs2_std = V((torch.var if ucuc_var else torch.std)(bs2.mean(-1), -1))
        arm = (r_mean + uc * r_std + ucuc * bs2_std).data.max(dim=1)[1]

        r = V(FTC(bandit.step(NP(arm))))
        std_poisson = rng.binomial(1, p, (bandit.batches, heads)) if bernouli else rng.poisson(p, (bandit.batches, heads))
        std_poisson = V(FTC(std_poisson))
        avg_reward_err = (avg_reward[br, arm] - r.view(-1, 1)) ** 2 * std_poisson

        err = avg_reward_err.sum()
        sgd.zero_grad()
        err.backward()
        sgd.step()

        pulled_arms[:, i] = arm.cpu()

        if i == 0 or (i + 1) % 100 == 0:
            print(i, NP(r_std.mean()), NP(bs2_std.mean()))

    r_mean = torch.mean(avg_reward, -1)
    avg_err = NP(r_mean) - bandit.mean
    print('mae', np.mean(np.abs(avg_err)), 'mse', np.mean(avg_err**2))
    return NP(pulled_arms)


def uc_var_solver(bandit, steps, avg_lam=0.1, var_lam=0.3, init_reward=1.5, uc=1):
    avg_reward = torch.zeros(bandit.batches, bandit.size).cuda()
    var_reward = init_reward * torch.ones(bandit.batches, bandit.size).cuda()
    avg_reward = VG(avg_reward)
    pulled_arms = torch.zeros(bandit.batches, steps)
    br = torch.arange(0, bandit.batches).long().cuda()
    sgd = optim.SGD([{'params': avg_reward, 'lr': avg_lam}], lr=0)

    for i in range(steps):
        # var_reward = log_var_reward.exp()
        arm = (avg_reward.data + uc * var_reward).max(dim=1)[1]

        r = V(FTC(bandit.step(NP(arm))))
        avg_reward_err = (avg_reward[br, arm] - r) ** 2
        var_reward[br, arm] = (1 - var_lam) * var_reward[br, arm] + var_lam * avg_reward_err.data
        # var_reward_err = (var_reward[br, arm] - avg_reward_err.detach()) ** 2

        err = avg_reward_err.sum() #+ var_reward_err.sum()
        sgd.zero_grad()
        err.backward()
        sgd.step()

        pulled_arms[:, i] = arm.cpu()

    avg_err = NP(avg_reward) - bandit.mean
    std_err = NP(var_reward) - bandit.std
    print('avg mae', np.mean(np.abs(avg_err)), 'avg mse', np.mean(avg_err**2))
    print('std mae', np.mean(np.abs(std_err)), 'std mse', np.mean(std_err**2))
    return NP(pulled_arms)


def solve_many(solver, size=10, restarts=1000, steps=250, drift=0):
    bandit = Bandit(size, restarts, drift)
    pulled = solver(bandit, steps=steps)
    best = np.broadcast_to(np.argmax(bandit.mean, axis=1).reshape(-1, 1), (restarts, steps))
    pulled_arms = pulled == best
    return pulled_arms.mean(0)


def evaluate_solvers(solvers, size=10, restarts=1000, steps=250, sigma=9, drift=0):
    for i, s in enumerate(solvers):
        y = solve_many(s, size, restarts, steps, drift)
        y = gaussian_filter(y, sigma=sigma)
        plt.plot(y, label=str(i))
    plt.legend()
    plt.show()
