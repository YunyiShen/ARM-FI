import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from coresets.algorithms import construct_lr_coreset_with_kmeans
import numpy as np
import torch.nn.functional as F

### get candidates ###
def random_candidate_xprompt(embd, reward, size_of_prompts, size_of_candidates):
    idx = torch.randperm(embd.shape[1])[:size_of_prompts]
    tmp = embd[:,idx,:]
    reward = reward[:,idx]
    tmp = tmp.reshape(-1,tmp.shape[2])
    reward = reward.reshape(-1)
    all_pairs = torch.combinations(torch.arange(tmp.shape[0]), r=2)
    all_pairs = all_pairs[torch.randperm(all_pairs.shape[0])[:size_of_candidates],:]

    team1 = tmp[all_pairs[:,0],:]
    team2 = tmp[all_pairs[:,1],:]

    reward_team1 = reward[all_pairs[:,0]]
    reward_team2 = reward[all_pairs[:,1]]
    return team1, team2, reward_team1, reward_team2


    

def random_candidate_withinprompt(embd, reward, size_of_prompts, size_of_candidates):
    perprompt = embd.shape[0]
    assert size_of_candidates <= perprompt * (perprompt-1) / 2 * size_of_prompts
    idx = torch.randperm(embd.shape[1])[:size_of_prompts]
    tmp = embd[:,idx,:]
    reward = reward[:,idx]
    all_pairs = torch.combinations(torch.arange(embd.shape[0]), r=2)
    team1 = tmp[all_pairs[:,0],:,:]
    team2 = tmp[all_pairs[:,1],:,:]

    reward_team1 = reward[all_pairs[:,0],:]
    reward_team2 = reward[all_pairs[:,1],:]
    #breakpoint()
    team1 = team1.reshape(-1,team1.shape[2])
    team2 = team2.reshape(-1,team2.shape[2])
    reward_team1 = reward_team1.reshape(-1)
    reward_team2 = reward_team2.reshape(-1)

    sumsamplint_idx = torch.randperm(team1.shape[0])[:size_of_candidates]
    team1 = team1[sumsamplint_idx,:]
    team2 = team2[sumsamplint_idx,:]
    reward_team1 = reward_team1[sumsamplint_idx]
    reward_team2 = reward_team2[sumsamplint_idx]
    return team1, team2, reward_team1, reward_team2

def random_candidate(embd, reward, size_of_prompts, size_of_candidates, xprompt = False):
    if xprompt:
        return random_candidate_xprompt(embd, reward, size_of_prompts, size_of_candidates)
    else:
        return random_candidate_withinprompt(embd, reward, size_of_prompts, size_of_candidates)




### rules ###

def check_input(candidate1, candidate2, npairs):
    if len(candidate1) < npairs or len(candidate2) < npairs:
        raise ValueError("Not enough candidates to sample from")
    if len(candidate1) != len(candidate2):
        raise ValueError("Candidates must be of the same length")
    

def random_rule(candidates1, candidates2, npairs):
    """

    Args:
        candidates1: list of candidates
        candidate2: list of candidates
        npairs: number of pairs to sample
    return:
        candidates1: list of candidates
        candidate2: list of candidates
        random_idx: indices of the selected pairs
    """
    check_input(candidates1, candidates2, npairs)
    #breakpoint()
    random_idx = np.random.choice(candidates1.shape[0], npairs, replace=False)
    return candidates1[random_idx], candidates2[random_idx], random_idx


def levelset_rule(candidates1, candidates2, model, npairs, inverse=False):
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
    check_input(candidates1, candidates2, npairs)
    with torch.no_grad():  # Turn off gradients for prediction
        scores1 = model(candidates1).squeeze()  # Predicted scores, shape: (N,)
        scores2 = model(candidates2).squeeze()
    score_diffs = torch.abs(scores1-scores2).to(device)
    diffs, indices = torch.topk(score_diffs, 
                                k=npairs, 
                                largest=inverse)
    return candidates1[indices], candidates2[indices], indices

def coreset_rule(candidates1, candidates2, model, npairs, K = 10):
    """
    given candidates, return pairs of candidates that should be compared
    args:
        candidates: tensor of size [N, D] for N candidates with D features
        model: a model that takes in candidates and outputs embeddings via method embd
        npairs: number of pairs to return
        K: number of centers to use in kmeans for constructing the coreset
    returns:
        team1, team2: tensors of size [npairs, D] for the pairs of candidates

    """
    check_input(candidates1, candidates2, npairs)
    with torch.no_grad():  # Turn off gradients for prediction
        embd1 = model.embd(candidates1)  # Predicted embeddings, shape: (N, D)
        embd2 = model.embd(candidates2)
    embd_diffs = embd1 - embd2
    coreset_indices = construct_lr_coreset_with_kmeans(embd_diffs.cpu(), K, 
                                                       output_size_param = npairs, index_only = True)
    return candidates1[coreset_indices], candidates2[coreset_indices], coreset_indices

def dopt_rule(candidates1, candidates2, model, npairs, batch_size = None):
    '''
    (Bayesian) D-optimal design rule
    candidates: tensor of size [N, D] for N candidates with D features
    model: a model that takes in candidates and outputs embeddings by model.embd, as well as applying last layer by model.lastlayer
    npairs: number of pairs to return

    '''
    check_input(candidates1, candidates2, npairs)
    with torch.no_grad():  # Turn off gradients for prediction
        embd1 = model.embd(candidates1)  # Predicted embeddings, shape: (N, D)
        embd2 = model.embd(candidates2)
    embd_diffs_flat = embd1 - embd2
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

    if batch_size is None:
        XtX = embd_diffs_flat[:,None,:] * embd_diffs_flat[:,:,None] * variances[:, None, None] * weights[:, None, None] # size [npairs, D, D]
        XtX = XtX.sum(dim=0) # size [D, D]
    else:
        for i in range(0, embd_diffs_flat.shape[0], batch_size):
            end = min(i + batch_size, embd_diffs_flat.shape[0])
            XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * variances[i:end][:, None, None] * weights[i:end][:, None, None]).sum(dim = 0)
    # add small number to diagonal to make it invertible
    XtX += 1e-6 * torch.eye(XtX.shape[0]).to(device)
    detXtX = torch.logdet(XtX).to(device)
    # take gradient over weights
    gradients_to_weights = torch.autograd.grad(detXtX, weights, create_graph=True)[0]
    #breakpoint()
    # find the pair with the largest gradient
    max_grad_idx = torch.topk(gradients_to_weights, npairs).indices
    return candidates1[max_grad_idx], candidates2[max_grad_idx], max_grad_idx


def pastaware_dopt_rule(candidates1, candidates2, past1, past2, model, npairs, batch_size = None):
    '''
    (Bayesian) D-optimal design rule, with past awareness
    candidates: tensor of size [N, D] for N candidates with D features
    past1, past2: tensor of size [M, D] for M past candidates with D features
    model: a model that takes in candidates and outputs embeddings by model.embd, as well as applying last layer by model.lastlayer
    npairs: number of pairs to return

    '''
    check_input(candidates1, candidates2, npairs)
    with torch.no_grad():  # Turn off gradients for prediction
        embd1 = model.embd(candidates1)  # Predicted embeddings, shape: (N, D)
        embd2 = model.embd(candidates2)
        pastembd1 = model.embd(past1)
        pastembd2 = model.embd(past2)
    embd_diffs_flat = embd1 - embd2
    pastembd_diffs_flat = pastembd1 - pastembd2
    with torch.no_grad():
        logits = model.lastlayer(embd_diffs_flat)
        variances = torch.sigmoid(logits) # size [npairs,]
        variances = variances * (1.-variances) # convert to variance

        past_logits = model.lastlayer(pastembd_diffs_flat)
        past_variances = torch.sigmoid(past_logits) # size [npairs,]
        past_variances = past_variances * (1.-past_variances) # convert to variance

    weights = torch.ones_like(variances) # weight we take gradient over
    # require grad
    weights.requires_grad = True
    XtX = torch.zeros(embd_diffs_flat.shape[1], 
                           embd_diffs_flat.shape[1]) # size [D, D]
    
    # accumulate XtX in batch
    #breakpoint()
    if batch_size is None:
        pastXtX = pastembd_diffs_flat[:,None,:] * pastembd_diffs_flat[:,:,None] * past_variances[:, None, None] # size [npairs, D, D]
        pastXtX = pastXtX.sum(dim=0) # size [D, D]
    else:
        for i in range(0, pastembd_diffs_flat.shape[0], batch_size):
            end = min(i + batch_size, pastembd_diffs_flat.shape[0])
            pastXtX += (pastembd_diffs_flat[i:end][:,None,:] * pastembd_diffs_flat[i:end][:,:,None] * past_variances[i:end][:, None, None]).sum(dim = 0)

    if batch_size is None:
        XtX = embd_diffs_flat[:,None,:] * embd_diffs_flat[:,:,None] * variances[:, None, None] * weights[:, None, None] # size [npairs, D, D]
        XtX = XtX.sum(dim=0) # size [D, D]
    else:
        for i in range(0, embd_diffs_flat.shape[0], batch_size):
            end = min(i + batch_size, embd_diffs_flat.shape[0])
            XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * variances[i:end][:, None, None] * weights[i:end][:, None, None]).sum(dim = 0)
    # add small number to diagonal to make it invertible
    XtX += 1e-6 * torch.eye(XtX.shape[0]).to(device)
    XtX += pastXtX
    detXtX = torch.logdet(XtX).to(device)
    # take gradient over weights
    gradients_to_weights = torch.autograd.grad(detXtX, weights, create_graph=True)[0]
    #breakpoint()
    # find the pair with the largest gradient
    max_grad_idx = torch.topk(gradients_to_weights, npairs).indices
    return candidates1[max_grad_idx], candidates2[max_grad_idx], max_grad_idx



def XtX_rule(candidates1, candidates2, model, npairs, batch_size = None):
    '''
    Bayesian D-optimal design rule assuming OLS fit rather than logistic
    candidates: tensor of size [N, D] for N candidates with D features
    model: a model that takes in candidates and outputs embeddings by method model.embd
    npairs: number of pairs to return

    '''
    check_input(candidates1, candidates2, npairs)
    with torch.no_grad():  # Turn off gradients for prediction
        embd1 = model.embd(candidates1)  # Predicted embeddings, shape: (N, D)
        embd2 = model.embd(candidates2)
    embd_diffs_flat = embd1 - embd2
    weights = torch.ones(embd_diffs_flat.shape[0]).to(device) # weight we take gradient over
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
        XtX = embd_diffs_flat[:,None,:].to(device) * embd_diffs_flat[:,:,None].to(device) * weights[:, None, None].to(device) # size [npairs, D, D]
        XtX = XtX.sum(dim=0) # size [D, D]
    else:
        for i in range(0, embd_diffs_flat.shape[0], batch_size):
            end = min(i + batch_size, embd_diffs_flat.shape[0])
            XtX += (embd_diffs_flat[i:end][:,None,:] * embd_diffs_flat[i:end][:,:,None] * weights[i:end][:, None, None]).sum(dim = 0)
    XtX += 1e-6 * torch.eye(XtX.shape[0]).to(device)
    detXtX = torch.logdet(XtX).to(device)
    # take gradient over weights
    gradients_to_weights = torch.autograd.grad(detXtX, weights, create_graph=True)[0]
    #breakpoint()
    # find the pair with the largest gradient
    max_grad_idx = torch.topk(gradients_to_weights, npairs).indices
    return candidates1[max_grad_idx], candidates2[max_grad_idx], max_grad_idx
