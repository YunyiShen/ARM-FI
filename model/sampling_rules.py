import torch 
from coresets.algorithms import construct_lr_coreset_with_kmeans
import numpy as np
import torch.nn.functional as F


def random_rule(candidates, npairs):
    """
    given candidates, return pairs of candidates that should be compared
    args:
        candidates: tensor of size [N, D] for N candidates with D features
        npairs: number of pairs to return
    returns:
        team1, team2: tensors of size [npairs, D] for the pairs of candidates
    """
    n_responses = len(candidates)
    assert npairs < n_responses * (n_responses - 1) / 2 # only subset pairs
    row_indices, col_indices = torch.triu_indices(n_responses, n_responses, offset=1)  # Excludes diagonal
    indices = torch.randperm(len(row_indices))[:npairs]
    rows = row_indices[indices]
    cols = col_indices[indices]
    team1 = candidates[rows]
    team2 = candidates[cols]
    return team1, team2, rows, cols

def levelset_rule(candidates, model, npairs, inverse=False):
    """
    given candidates, return pairs of candidates that should be compared
    args:
        candidates: tensor of size [N, D] for N candidates with D features
        model: a model that takes in candidates and outputs scores
        npairs: number of pairs to return
        inverse: if True, return the pairs with the smallest difference in scores, i.e., levelset rule
                 if False, return the pairs with the largest difference in scores, i.e., largest vs smallest rule
    returns:
        team1, team2: tensors of size [npairs, D] for the pairs of candidates

    """
    n_responses = len(candidates)
    assert npairs < n_responses * (n_responses - 1) / 2 # only subset pairs
    with torch.no_grad():  # Turn off gradients for prediction
        scores = model(candidates).squeeze()  # Predicted scores, shape: (N,)
    score_diffs = torch.abs(scores.unsqueeze(0) - scores.unsqueeze(1))  # Shape: (N, N)
    triu_mask = torch.triu(torch.ones_like(score_diffs, dtype=torch.bool), diagonal=1)
    score_diffs = score_diffs[triu_mask] 
    row_indices, col_indices = torch.triu_indices(n_responses, n_responses, offset=1)  # Excludes diagonal
    score_diffs_flat = score_diffs.flatten()  # Shape: (N*N,)
    diffs, indices = torch.topk(score_diffs_flat, 
                                       k=npairs, 
                                       largest=inverse)  # Smallest n_pairs
    rows = row_indices[indices] #indices // n_responses  # Row indices of the pairs
    cols = col_indices[indices] #indices % n_responses   # Column indices of the pairs
    team1 = candidates[rows]
    team2 = candidates[cols]
    #breakpoint()
    return team1, team2, rows, cols


def noisy_levelset_rule(candidates, model, npairs, portion_levelset,inverse=False):
    """
    given candidates, return pairs of candidates that should be compared
    args:
        candidates: tensor of size [N, D] for N candidates with D features
        model: a model that takes in candidates and outputs scores
        npairs: number of pairs to return
        inverse: if True, return the pairs with the smallest difference in scores, i.e., levelset rule
                 if False, return the pairs with the largest difference in scores, i.e., largest vs smallest rule
    returns:
        team1, team2: tensors of size [npairs, D] for the pairs of candidates

    """
    n_responses = len(candidates)
    assert npairs < n_responses * (n_responses - 1) / 2 # only subset pairs
    with torch.no_grad():  # Turn off gradients for prediction
        scores = model(candidates).squeeze()  # Predicted scores, shape: (N,)
    score_diffs = torch.abs(scores.unsqueeze(0) - scores.unsqueeze(1))  # Shape: (N, N)
    triu_mask = torch.triu(torch.ones_like(score_diffs, dtype=torch.bool), diagonal=1)
    score_diffs = score_diffs[triu_mask] 
    row_indices, col_indices = torch.triu_indices(n_responses, n_responses, offset=1)  # Excludes diagonal
    
    
    
    score_diffs_flat = score_diffs.flatten()  # Shape: (N*N,)
    take = int(npairs*portion_levelset)
    diffs, indices = torch.topk(score_diffs_flat, 
                                       k=take, 
                                       largest=inverse)  # Smallest n_pairs
    rows = row_indices[indices] #indices // n_responses  # Row indices of the pairs
    cols = col_indices[indices]
    
    team1 = candidates[rows]
    team2 = candidates[cols]
    random_rows = torch.randperm(n_responses)[:npairs-take]
    random_cols = torch.randperm(n_responses)[:npairs-take]
    random_team1 = candidates[random_rows]
    random_team2 = candidates[random_cols]
    team1 = torch.cat([team1, random_team1], dim=0)
    team2 = torch.cat([team2, random_team2], dim=0)
    #breakpoint()
    rows = torch.cat([rows, random_rows], dim=0)
    cols = torch.cat([cols, random_cols], dim=0)
    return team1, team2, rows, cols

def coreset_rule(candidates, model, npairs, K = 10):
    '''
    candidates: tensor of size [N, D] for N candidates with D features
    model: a model that takes in candidates and outputs embeddings
    npairs: number of pairs to return
    K: number of centers for kmeans

    returns:
        team1, team2: tensors of size [npairs, D] for the pairs of candidates
    '''
    n_responses = len(candidates)
    assert npairs < n_responses * (n_responses - 1) / 2 # only subset pairs
    with torch.no_grad():  # Turn off gradients for prediction
        embeddings = model.embd(candidates)  # Predicted embeddings, shape: (N, D)
    embd_diffs = embeddings[None, :, :] - embeddings[:, None, :]  # Shape: (N, N, D)
    upper_triangle_indices = torch.triu_indices(n_responses, n_responses, offset=1)  # Shape: [2, batch(batch-1)/2]
    i_indices, j_indices = upper_triangle_indices

    embd_diffs_flat = embd_diffs[i_indices, j_indices]  # Shape: [batch(batch-1)/2, D]
    embd_diffs_flat = embd_diffs_flat.detach().cpu().numpy()
    # Construct coreset
    coreset_indices = construct_lr_coreset_with_kmeans(embd_diffs_flat, K, 
                                                       output_size_param = npairs, index_only = True)
    rows = i_indices[torch.tensor(coreset_indices)]
    cols = j_indices[torch.tensor(coreset_indices)]

    return candidates[rows], candidates[cols], rows, cols

def dopt_rule(candidates, model, npairs, batch_size = None):
    '''
    Bayesian D-optimal design rule
    candidates: tensor of size [N, D] for N candidates with D features
    model: a model that takes in candidates and outputs embeddings by model.embd, as well as applying last layer by model.lastlayer
    npairs: number of pairs to return

    '''
    n_responses = len(candidates)
    assert npairs < n_responses * (n_responses - 1) / 2 # only subset pairs
    with torch.no_grad():  # Turn off gradients for prediction
        embeddings = model.embd(candidates)  # Predicted embeddings, shape: (N, D)
    
    embd_diffs = embeddings[None, :, :] - embeddings[:, None, :]  # Shape: (N, N, D)
    upper_triangle_indices = torch.triu_indices(n_responses, n_responses, offset=1)  # Shape: [2, batch(batch-1)/2]
    i_indices, j_indices = upper_triangle_indices
    embd_diffs_flat = embd_diffs[i_indices, j_indices]  # Shape: [batch(batch-1)/2, D]
    with torch.no_grad():
        logits = model.lastlayer(embd_diffs_flat)
        variances = torch.sigmoid(logits) # size [npairs,]
        variances = variances * (1.-variances) # convert to variance
    weights = torch.ones_like(variances) # weight we take gradient over
    # require grad
    weights.requires_grad = True
    XtX = torch.zeros(embd_diffs_flat.shape[1], 
                           embd_diffs_flat.shape[1]) # size [D, D]
    
    # accumulate XtX in batch
    #breakpoint()
    '''
    for i in range(0, embd_diffs_flat.shape[0], batch_size):
        end = min(i + batch_size, embd_diffs_flat.shape[0])
        breakpoint()
        XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * variances[i:end][:, None, None] * weights[i:end][:, None, None]).sum(dim = 0)
    '''

    if batch_size is None:
        XtX = embd_diffs_flat[:,None,:] * embd_diffs_flat[:,:,None] * variances[:, None, None] * weights[:, None, None] # size [npairs, D, D]
        XtX = XtX.sum(dim=0) # size [D, D]
    else:
        for i in range(0, embd_diffs_flat.shape[0], batch_size):
            end = min(i + batch_size, embd_diffs_flat.shape[0])
            XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * variances[i:end][:, None, None] * weights[i:end][:, None, None]).sum(dim = 0)
    detXtX = torch.logdet(XtX)
    # take gradient over weights
    gradients_to_weights = torch.autograd.grad(detXtX, weights, create_graph=True)[0]
    #breakpoint()
    # find the pair with the largest gradient
    max_grad_idx = torch.topk(gradients_to_weights, npairs).indices
    rows = i_indices[max_grad_idx]
    cols = j_indices[max_grad_idx]

    return candidates[rows], candidates[cols], rows, cols

def XtX_rule(candidates, model, npairs, batch_size = None):
    '''
    Bayesian D-optimal design rule assuming OLS fit rather than logistic
    candidates: tensor of size [N, D] for N candidates with D features
    model: a model that takes in candidates and outputs embeddings by method model.embd
    npairs: number of pairs to return

    '''
    n_responses = len(candidates)
    assert npairs < n_responses * (n_responses - 1) / 2 # only subset pairs
    with torch.no_grad():  # Turn off gradients for prediction
        embeddings = model.embd(candidates)  # Predicted embeddings, shape: (N, D)
    
    embd_diffs = embeddings[None, :, :] - embeddings[:, None, :]  # Shape: (N, N, D)
    upper_triangle_indices = torch.triu_indices(n_responses, n_responses, offset=1)  # Shape: [2, batch(batch-1)/2]
    i_indices, j_indices = upper_triangle_indices
    embd_diffs_flat = embd_diffs[i_indices, j_indices]  # Shape: [batch(batch-1)/2, D]
    weights = torch.ones(embd_diffs_flat.shape[0]) # weight we take gradient over
    # require grad
    weights.requires_grad = True
    XtX = torch.zeros(embd_diffs_flat.shape[1], 
                           embd_diffs_flat.shape[1]) # size [D, D]
    
    # accumulate XtX in batch
    #breakpoint()
    '''
    for i in range(0, embd_diffs_flat.shape[0], batch_size):
        end = min(i + batch_size, embd_diffs_flat.shape[0])
        breakpoint()
        XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * variances[i:end][:, None, None] * weights[i:end][:, None, None]).sum(dim = 0)
    '''

    if batch_size is None:
        XtX = embd_diffs_flat[:,None,:] * embd_diffs_flat[:,:,None] * weights[:, None, None] # size [npairs, D, D]
        XtX = XtX.sum(dim=0) # size [D, D]
    else:
        for i in range(0, embd_diffs_flat.shape[0], batch_size):
            end = min(i + batch_size, embd_diffs_flat.shape[0])
            XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * weights[i:end][:, None, None]).sum(dim = 0)
    detXtX = torch.logdet(XtX)
    # take gradient over weights
    gradients_to_weights = torch.autograd.grad(detXtX, weights, create_graph=True)[0]
    #breakpoint()
    # find the pair with the largest gradient
    max_grad_idx = torch.topk(gradients_to_weights, npairs).indices
    rows = i_indices[max_grad_idx]
    cols = j_indices[max_grad_idx]

    return candidates[rows], candidates[cols], rows, cols


