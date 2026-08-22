import torch
import torch.nn.functional as F

# ==============================================================================
# HELPER FUNCTION
# ==============================================================================

def get_embedding(model, x):
    """
    Extracts the latent embedding from the TerraMind model.
    """
    return model({"S2L2A": x})[-1]

# ==============================================================================
# ATTACK ALGORITHMS
# ==============================================================================

def fgsm_cos(model, x, epsilon=0.01):
    x_adv = x.clone().detach().requires_grad_(True)
    emb = get_embedding(model, x_adv)
    emb_orig = get_embedding(model, x).detach()

    loss = -F.cosine_similarity(emb, emb_orig, dim=-1).mean()
    loss.backward()

    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return torch.clamp(x_adv, 0, 1).detach()

def fgsm_l2(model, x, epsilon=0.01):
    x_adv = x.clone().detach().requires_grad_(True)
    emb = get_embedding(model, x_adv)
    emb_orig = get_embedding(model, x).detach()

    loss = torch.norm(emb - emb_orig, p=2, dim=-1).mean()
    loss.backward()

    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return torch.clamp(x_adv, 0, 1).detach()

def pgd_cos(model, x, epsilon=0.01, alpha=0.002, steps=10):
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        emb = get_embedding(model, x_adv)
        emb_orig = get_embedding(model, x_orig).detach()

        loss = -F.cosine_similarity(emb, emb_orig, dim=-1).mean()
        loss.backward()

        x_adv = x_adv + alpha * x_adv.grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0, 1).detach()

    return x_adv

def pgd_l2(model, x, epsilon=0.01, alpha=0.002, steps=10):
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        emb = get_embedding(model, x_adv)
        emb_orig = get_embedding(model, x_orig).detach()

        loss = torch.norm(emb - emb_orig, p=2, dim=-1).mean()
        loss.backward()

        x_adv = x_adv + alpha * x_adv.grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0, 1).detach()

    return x_adv

def pgd_spectral_group(model, x, target_channels, epsilon=0.01, alpha=0.002, steps=10):
    x_orig = x.clone().detach()
    x_adv = x_orig.clone()

    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    for ch in target_channels:
        x_adv[:, ch, :, :] = x_adv[:, ch, :, :] + noise[:, ch, :, :]
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        emb = get_embedding(model, x_adv)
        emb_orig = get_embedding(model, x_orig).detach()

        loss = -F.cosine_similarity(emb, emb_orig, dim=-1).mean()
        loss.backward()

        grad = x_adv.grad
        update = torch.zeros_like(x_adv)

        for ch in target_channels:
            update[:, ch, :, :] = alpha * grad[:, ch, :, :].sign()

        x_adv = x_adv + update

        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0, 1).detach()

    return x_adv
