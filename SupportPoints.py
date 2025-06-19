import torch

def ccp(x, Y):
    # initialize output tensor with same shape as x
    x_new = torch.zeros_like(x)

    # get number of samples in x and Y where N > n
    n = x.shape[0]
    N = Y.shape[0]

    for i in range(n):
        xi = x[i,:] # get i-th row of x

        # calculate euclidean distance of xi to original data Y
        ti = (xi-Y).pow(2)
        ti2 = torch.sum(ti, dim=1).sqrt()
        ti3 = ti2[ti2 != 0]
        # sum of inverse of nonzero distances
        qi = torch.sum(ti3.pow(-1))

        # only select rows where dist > 0
        a = Y[ti2!=0,:]
        b = ti2[ti2!=0].unsqueeze(1)
        outerDist = torch.sum(a/b, dim=0)

        # calculate euclidean distance of xi to all other support points x
        ui = xi-x
        ui2 = torch.sum((xi-x).pow(2), dim=1).sqrt()

        # only select rows where dist > 0
        a = ui[ui2!=0,:]
        b = ui2[ui2!=0].unsqueeze(1)
        innerDist = torch.sum(a/b, dim=0)

        # calculate weighted sum of inner dist and outer dist, scaled by qi
        x_new[i,:] = qi.pow(-1) * ((N/n) * innerDist + outerDist)
    return x_new

# itaratively apply ccp
def do_ccp(Y, n=100, n_rounds_max=250, tolerance=1e-3):
    # create uniform distribution p
    N = Y.shape[0]
    p = torch.tensor([1/N]*N)

    # sample n random indices from range of Y-rows with distribution p
    idx = p.multinomial(num_samples=n, replacement=False)

    # initialize support points by selecting idx rows of Y
    x = Y[idx, :]

    for j in range(n_rounds_max):
        # use ccp to get new support points
        x2 = ccp(x,Y)
        # calculate distance to given support points
        d = torch.sum((x-x2).pow(2))
        # if distace small enough -> support points have converged
        if (d < tolerance):
            return x2, j
        x = x2

    # return final support points and maximum number of iterations
    return x, j

# filters set of points based on proability under given distribution
def constrain_points(x, dist):
  keep = torch.isfinite(dist.log_prob(x))
  return x[keep]