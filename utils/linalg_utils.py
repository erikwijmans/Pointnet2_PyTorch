import torch
from enum import Enum

PDist2Order = Enum('PDist2Order', 'd_first d_second')


def pdist2(
        X: torch.Tensor,
        Z: torch.Tensor = None,
        order: PDist2Order = PDist2Order.d_second
) -> torch.Tensor:
    r""" Calculates the pairwise distance between X and Z

    D[b, i, j] = l2 distance X[b, i] and Z[b, j]

    Parameters
    ---------
    X : torch.Tensor
        X is a (B, N, d) tensor.  There are B batches, and N vectors of dimension d
    Z: torch.Tensor
        Z is a (B, M, d) tensor.  If Z is None, then Z = X

    Returns
    -------
    torch.Tensor
        Distance matrix is size (B, N, M)
    """

    if order == PDist2Order.d_second:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Z is None:
            Z = X
            G = X @ Z.transpose(-2, -1)
            S = (X * X).sum(-1, keepdim=True)
            R = S.transpose(-2, -1)
        else:
            if Z.dim() == 2:
                Z = Z.unsqueeze(0)
            G = X @ Z.transpose(-2, -1)
            S = (X * X).sum(-1, keepdim=True)
            R = (Z * Z).sum(-1, keepdim=True).transpose(-2, -1)
    else:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Z is None:
            Z = X
            G = X.transpose(-2, -1) @ Z
            R = (X * X).sum(-2, keepdim=True)
            S = R.transpose(-2, -1)
        else:
            if Z.dim() == 2:
                Z = Z.unsqueeze(0)
            G = X.transpose(-2, -1) @ Z
            S = (X * X).sum(-2, keepdim=True).transpose(-2, -1)
            R = (Z * Z).sum(-2, keepdim=True)

    return torch.abs(R + S - 2 * G).squeeze(0)


def pdist2_slow(X, Z=None):
    if Z is None: Z = X
    D = torch.zeros(X.size(0), X.size(2), Z.size(2))

    for b in range(D.size(0)):
        for i in range(D.size(1)):
            for j in range(D.size(2)):
                D[b, i, j] = torch.dist(X[b, :, i], Z[b, :, j])
    return D


if __name__ == "__main__":
    X = torch.randn(2, 3, 5)
    Z = torch.randn(2, 3, 3)

    print(pdist2(X, order=PDist2Order.d_first))
    print(pdist2_slow(X))
    print(torch.dist(pdist2(X, order=PDist2Order.d_first), pdist2_slow(X)))
