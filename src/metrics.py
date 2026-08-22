import torch
import torch.nn.functional as F

def compute_metrics(model, x_orig, x_adv):
    """
    Computes the Cosine Similarity and L2 Euclidean Distance between the latent 
    representations of the original and adversarial patches.
    """
    model.eval() # Ensure the model is in evaluation mode
    
    with torch.no_grad():
        # Extract embeddings (TerraMind specific format)
        f_orig = model({"S2L2A": x_orig})[-1]
        f_adv  = model({"S2L2A": x_adv})[-1]

    # Calculate Cosine Similarity
    cos = F.cosine_similarity(
        f_orig.flatten(1),
        f_adv.flatten(1)
    ).mean().item()

    # Calculate L2 Distance
    l2 = torch.norm(f_adv - f_orig, p=2).item()

    return cos, l2
