import torch
from scem import stein
from scem.gen import CSGRBMPosterior


def main():
    from scem.ebm import GaussianRBM
    seed = 13
    torch.manual_seed(seed)
    n = 170
    dx = 10
    dz = 10
    # Instantiate a GRBM object
    X = torch.randn([n, dx])
    W = torch.randn([dx, dz]) / (dx * dz)**0.5
    c = torch.randn([dz, ])
    b = torch.randn([dx, ])
    grbm = GaussianRBM(W, b, c)

    # Approximate the score with exact posterior
    cs = CSGRBMPosterior(grbm)
    n_sample = 200
    approx_score = stein.ApproximateScore(
        grbm.score_joint_obs, cs)
    # Compare to the marginal score
    marginal_score_mse = (torch.mean(
        (approx_score(X, n_sample=n_sample)-grbm.score_marginal_obs(X))**2))
    print('Marginal score mse: {}'.format(marginal_score_mse))


if __name__ == '__main__':
    main()