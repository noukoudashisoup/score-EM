import torch
from scem import gen, stein, kernel, ebm
from scem import util


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.)


# learning GRBM posterior with kcsd
def main():
    seed = 13
    torch.manual_seed(seed)
    n = 300
    dx = 10 
    dz = 3

    # instantiate a GRBM object
    n_cat = 2
    X = torch.randn([n, dx])
    W = torch.randn([dx, dz]) / (dx * dz)**0.5
    c = torch.randn([dz, ])
    b = torch.randn([dx, ])
    grbm = ebm.GaussianRBM(W, b, c)

    # define kernels
    med2 = util.pt_meddistance(X)**2
    kx = kernel.PTKGauss(torch.tensor([med2]))
    kz = kernel.OHKGauss(n_cat, torch.tensor([dz*1.0]))

    # q(z|x)
    cs = gen.CSGRBMBernoulliFamily(dx, dz)
    cs.apply(init_weights)
    
    # optimizer settings
    learning_rate = 1e-3
    weight_decay = 0.
    optimizer = torch.optim.Adam(cs.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    # evaluate true grad log p(X) for comparison
    S = grbm.score_marginal_obs(X)

    # approximate score function
    approx_score = stein.ApproximateScore(
        grbm.score_joint_obs, cs)

    # kcsd training
    T = 2000
    n_sample = 200
    for t in range(T):
        Z = cs.sample(1, X, seed+t)
        Z = Z.squeeze(0)
        loss = stein.kcsd_ustat(
            X, Z, grbm.score_joint_latent, kx, kz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if t%100 == 0:
            cs = cs.eval()
            Z = cs.sample(1, X, seed+t)
            Z = Z.squeeze(0)
            loss = stein.kcsd_ustat(
                X, Z, grbm.score_joint_latent, kx, kz)
            marginal_score_mse = (torch.mean(
                (approx_score(X, n_sample=n_sample)-S)**2))
            print('(iter, loss, score mse): {}, {}. {}'.format(t, loss, marginal_score_mse))
            cs = cs.train()


if __name__ == '__main__':
    main()