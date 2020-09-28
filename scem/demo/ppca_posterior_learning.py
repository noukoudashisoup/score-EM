import torch
from scem import gen, stein, kernel, ebm
from scem import util


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.)


# learning PPCA posterior with kcsd
def main():
    seed = 13
    torch.manual_seed(seed)
    n = 200
    dx = 4
    dz = 2
    W = torch.randn([dx, dz])
    var = torch.tensor([2.0])
    ppca = ebm.PPCA(W, var)
    Z = torch.randn(n, dz)
    X = Z @ W.T + torch.randn(n, dx) * var**0.5

    med2 = util.pt_meddistance(X, seed=seed+1)**2
    kx = kernel.KGauss(torch.tensor([med2]))

    # q(z|x)
    cs = gen.PTCSGaussLinearMean(dx, dz)
    cs.apply(init_weights)
    
    # optimizer settings
    learning_rate = 1e-4
    weight_decay = 1e-3
    optimizer = torch.optim.Adam(cs.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    # evaluate true log p(X) for comparison
    S = ppca.score_marginal_obs(X)

    # approximate score function
    approx_score = stein.ApproximateScore(
        ppca.score_joint_obs, cs)


    # kcsd training
    T = 2000
    n_sample = 200
    kz = kernel.KGauss(torch.tensor([1.0]))
    for t in range(T):
        Z = cs.sample(1, X, seed+t)
        Z = Z.squeeze(0)
        med2_z = util.pt_meddistance(Z, seed+1)**2
        kz.sigma2 = torch.tensor([med2_z])
        loss = stein.kcsd_ustat(
            X, Z, ppca.score_joint_latent, kx, kz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (t+1)%200 == 0:
            marginal_score_mse = (torch.mean(
                (approx_score(X, n_sample=n_sample)-S)**2))
            print('(iter, loss, score mse): {}, {}. {}'.format(t, loss, marginal_score_mse))


if __name__ == '__main__':
    main()