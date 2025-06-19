import sys
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch related stuff
import torch 
import torch.distributions as D

# SBI related stuff
import sbi
from sbi.inference import SNLE
from sbi.inference import SNPE
from sbi import utils as sutils

# modules
from metrics import *
from classifier import c2st
from SupportPoints import *

# task specific things (because the fucking import doesn't work)
# different budgets tested
budgets = [200]

# hyper parameters and other vars
num_budgets = len(budgets)
num_simulations = 10
num_rounds = 2
# hyperparameters for TwoMoons
theta_dim = 2
x_dim = 2
n = 100
a_l = -np.pi/2
a_u = np.pi/2
r_mu = 0.1
r_sig = 0.01
x_obs = torch.zeros(1, x_dim)

task = "2Moon"
iter = 1
path = f'results/SNLE/{task}/{iter}'

def save_sample(tensor, name):
  Path(f"{path}/samples").mkdir(parents=True, exist_ok=True)
  torch.save(tensor, f"{path}/samples/{name}.pt") 
results_mmd = torch.ones(num_simulations, num_budgets, 5)*float('nan')
results_c2st = torch.ones(num_simulations, num_budgets, 5)*float('nan')
timings = torch.ones(num_simulations, num_budgets, 5)*float('nan')

# true_sample - check "SLCP - generate true posterior.ipynb"
true_sample_full = torch.load("./saved/SLCP_sample_8points_mcmc.pt") 


# prior from TwoMoons
prior = sutils.BoxUniform(low=-1*torch.ones(theta_dim), high=1*torch.ones(theta_dim))

# simulator from TwoMoons
def simulator(theta):
    n = theta.shape[0] 
    a = D.Uniform(torch.tensor(a_l), torch.tensor(a_u)).sample((n,))
    r = D.Normal(torch.tensor(r_mu), torch.tensor(r_sig)).sample((n,))
    px = r*torch.cos(a)+0.25
    py = r*torch.sin(a)
    x = px - torch.abs(torch.sum(theta, dim=1))/np.sqrt(2)
    x = x.unsqueeze(1)
    y = py + (theta[:,1]-theta[:,0])/np.sqrt(2)
    y = y.unsqueeze(1)
    return torch.cat((x,y), 1)

# surrogate prior from TwoMoons (took rough boundaries of x's generated from prior theta's) 
x_min = torch.tensor([-1.5,-2]) 
x_max = torch.tensor([0.6, 2])
sur_prior = sutils.BoxUniform(low=x_min, high=x_max)

# true_sample (i don't even know where I need this?)
def gen_posterior_samples(obs=torch.tensor([0.0, 0.0]), prior=None, n_samples=1):
    # use opposite rotation as above
    c = 1/np.sqrt(2)

    theta = torch.zeros((n_samples, 2))
    for i in range(n_samples):
        p = simulator(torch.zeros(1,2))
        q = torch.zeros(2)
        q[0] = p[0,0] - obs[0]
        q[1] = obs[1] - p[0,1]

        if np.random.rand() < 0.5:
            q[0] = -q[0]

        theta[i, 0] = c * ( q[0] - q[1] )
        theta[i, 1] = c * ( q[0] + q[1] )

    return theta
    
true_sample = gen_posterior_samples(n_samples=1000)

# surrogate method implementation 
for i in range(num_budgets):
    for j in range(num_simulations):
        n = budgets[i]//num_rounds

        # Surrogate (on all but first round)
        """
        t1 = time.time()
        proposal = prior
        inference = SNPE(prior, density_estimator='nsf')
        for k in range(num_rounds):
            # 1st iteration use real simulator, train surrogate
            if k == 0:
                theta = proposal.sample((n * num_rounds,))
                x_sim = simulator(theta)

                # train surrogate
                inference2 = SNPE(sur_prior, density_estimator='nsf')
                density_estimator = inference2.append_simulations(theta=x_sim, x=theta).train()
                surrogate = inference2.build_posterior(density_estimator)

            # other itarations, use surrogate
            else:
                theta = proposal.sample((10 * n * num_rounds,))
                x_sim = torch.zeros(10 * n * num_rounds, x_dim)
                for l in range(len(theta)):
                    x_sim[l] = surrogate.sample((1,), x=theta[l, :], show_progress_bars=False)


            # train posterior NDE
            density_estimator = inference.append_simulations(theta, x_sim, proposal).train()
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_obs)


        sample_post = posterior.sample((1000,), x=x_obs)
        t2 = time.time()
        save_sample(sample_post, f"sur_{n}_{j}")

        mmd = MMD2(true_sample[:1000,:], sample_post)
        results_mmd[j,i,1] = mmd

        c2st_score = c2st(true_sample[:1000,], sample_post)
        results_c2st[j,i,1] = c2st_score
        """

        # support points
        t3 = time.time()
        proposal = prior
        inference = SNPE(prior, density_estimator='nsf')

        for k in range(num_rounds):
            theta = proposal.sample((n * 2,))
            theta_ss, _ = do_ccp(theta, n)
            theta_ss_ = constrain_points(theta_ss, proposal).reshape(-1, theta_dim)
            x_sim_ss = simulator(theta_ss_)
            density_estimator = inference.append_simulations(theta_ss_, x_sim_ss, proposal).train()
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_obs)

        sample_post1 = posterior.sample((1000,), x=x_obs)
        #t4 = time.time()
        save_sample(sample_post1, f"sp_{n}_{j}")

        #mmd = MMD2(true_sample[:1000,:], sample_post1)
        #results_mmd[j,i,2] = mmd

        c2st_score = c2st(true_sample[:1000,], sample_post1)
        results_c2st[j,i,2] = c2st_score

        #timings[j,i,1] = t2 - t1
        #timings[j,i,2] = t4 - t3

        #torch.save(results_mmd, f'{path}/0res_mmd.pkl')
        torch.save(results_c2st, f'{path}/1res_c2st.pkl')
        #torch.save(timings, f'{path}/timings.pkl')

#mmd_means = results_mmd.nanmean(dim=0)
c2st_means = results_c2st.nanmean(dim=0)
#timings_means = timings.nanmean(dim=0)

#np.savetxt(f'{path}/0mmd_means.csv', mmd_means.numpy(), delimiter=",")
np.savetxt(f'{path}/1c2st_means.csv', c2st_means.numpy(), delimiter=",")
#np.savetxt(f'{path}/timings_means.csv', timings_means.numpy(), delimiter=",")

#mmd_vars = results_mmd.var(dim=0)
c2st_vars = results_c2st.var(dim=0)
#timings_vars = timings.var(dim=0)

#np.savetxt(f'{path}/0mmd_vars.csv', mmd_vars.numpy(), delimiter=",")
np.savetxt(f'{path}/1c2st_vars.csv',c2st_vars.numpy(), delimiter=",")
#np.savetxt(f'{path}/timings_vars.csv',timings_vars.numpy(), delimiter=",")
