import torch


def householder(u, v):
    """
    Compute the Householder matrix H that reflects vector u onto vector v.
    """
    if not (torch.abs(u.norm() - 1) < 1e-6) or not (torch.abs(v.norm() - 1) < 1e-6):
        raise ValueError("Input vectors should be normalized")

    assert (
        u.shape == v.shape
    ), f"Input vectors should have the same shape, but got {u.shape} and {v.shape}"

    if torch.allclose(u, v):
        return torch.eye(u.shape[0], device=u.device)

    w = u - v
    w_norm = w.norm()
    if w_norm == 0:
        return torch.eye(u.shape[0], device=u.device)

    w_outer = torch.outer(w, w)
    H = torch.eye(u.shape[0], device=w_outer.device) - 2.0 * w_outer / w_norm**2

    return H
