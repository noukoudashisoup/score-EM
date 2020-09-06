import torch
from scem import ebm


def main():
    # score function of GRBM's p(z|x)
    def grbm_joint_score_latent(X, Z, W, c):
        n, dz = Z.shape
        lattice_ranges = 2 * torch.ones(dz)
        S = torch.empty([n, dz])
        for j in range(dz):
            Z_ = Z.clone()
            Z_[:, j] = (Z[:, j] + 1) % lattice_ranges[j]
            S[:, j] = torch.exp(-torch.sum((X@W+c)*(Z_-Z), dim=1)) - 1.
        return S

    seed = 13
    torch.manual_seed(seed)
    n = 100
    dx = 100
    dz = 10
    X = torch.randn([n, dx])
    Z = torch.randint(0, 2, [n, dz])
    Z_oh = torch.eye(2)[Z]
    W = torch.randn([dx, dz]) / (dx * dz)**0.5
    c = torch.randn([dz, ])
    b = torch.randn([dx, ])
    grbm = ebm.GaussianRBM(W, b, c)
    with torch.no_grad():
        s1 = grbm.score_joint_latent(X, Z_oh)
        s2 = grbm_joint_score_latent(X, 1.*Z, W, c)
    print("score error: {}".format(torch.mean((s1-s2)**2)))


if __name__ == '__main__':
    main()