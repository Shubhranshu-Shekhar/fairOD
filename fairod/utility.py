import torch
import pickle
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


def column_wise_norm(x):
    '''
    Returns column wise normalized array
    '''
    x_normed = (x - x.min(0, keepdim=True)[0]) / (x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0])
    return x_normed


def load_data(path):
    '''
    Data loading utility. Assumes the directory contain X.pkl, y.pkl, pv.pkl.
    '''
    with open(path + 'X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(path + 'pv.pkl', 'rb') as f:
        S = pickle.load(f)
    with open(path + 'y.pkl', 'rb') as f:
        y = pickle.load(f)

    return X, S, y


def group_train(model, dataloader, device, recon_criterion, sens_croterion, group_criterion,
                optimizer, alpha, beta, gamma, **kwargs):
    '''
    Trainer method for fair representation that uses a group based regulaizer. See Eq. 13, and L_gf for fairOD-C.
    '''
    model.train()
    ndcg_norm_maj = kwargs.get("ndcg_norm_maj")
    ndcg_norm_min = kwargs.get("ndcg_norm_min")

    training_loss = 0
    recon_loss = 0
    sens_loss = 0
    group_loss = 0
    for batch_idx, (X_i, S_i, A_i) in enumerate(dataloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        optimizer.zero_grad()
        recon_output = model(X_i)
        r_loss = recon_criterion(recon_output, X_i)
        s_loss = sens_croterion(recon_output, X_i, S_i)
        if ndcg_norm_maj:
            g_loss = group_criterion(recon_output, X_i, S_i, A_i, ndcg_norm_maj, ndcg_norm_min)
        else:
            g_loss = group_criterion(recon_output, X_i, S_i, A_i)
        loss = alpha * r_loss + beta * s_loss + gamma * g_loss

        loss.backward()
        optimizer.step()

        recon_loss += r_loss.item()
        sens_loss += s_loss.item()
        group_loss += g_loss.item()
        training_loss += loss.item()

    recon_loss /= len(dataloader)
    sens_loss /= len(dataloader)
    group_loss /= len(dataloader)
    training_loss /= len(dataloader)
    return recon_loss, sens_loss, training_loss, group_loss


def group_test(model, dataloader, device, recon_criterion, sens_criterion, group_criterion,
               alpha, beta, gamma, **kwargs):
    '''
    Evaluate method for fair representation that uses a group based regulaizer.
    '''
    model.eval()
    ndcg_norm_maj = kwargs.get("ndcg_norm_maj")
    ndcg_norm_min = kwargs.get("ndcg_norm_min")

    testing_loss = 0
    for batch_idx, (X_i, S_i, A_i) in enumerate(dataloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        recon_output = model(X_i)
        recon_loss = recon_criterion(recon_output, X_i)
        sens_loss = sens_criterion(recon_output, X_i, S_i)
        if ndcg_norm_maj:
            group_loss = group_criterion(recon_output, X_i, S_i, A_i, ndcg_norm_maj, ndcg_norm_min)
        else:
            group_loss = group_criterion(recon_output, X_i, S_i, A_i)
        loss = alpha * recon_loss + beta * sens_loss + gamma * group_loss

        testing_loss += loss.item()

    testing_loss /= len(dataloader)
    return testing_loss


def cor_train(model, dataloader, device, recon_criterion, sens_criterion, optimizer, alpha, beta):
    '''
    Trainer method for fair representation that does NOT use a group based regulaizer.
    Only regularizer used is a based on sensitive/protected attribute.
    '''
    model.train()
    training_loss = 0
    recon_loss = 0
    sens_loss = 0
    for batch_idx, (X_i, S_i) in enumerate(dataloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        optimizer.zero_grad()
        recon_output = model(X_i)
        recon_loss = recon_criterion(recon_output, X_i)
        sens_loss = sens_criterion(recon_output, X_i, S_i)
        loss = alpha * recon_loss + beta * sens_loss
        loss.backward()
        optimizer.step()
        recon_loss += recon_loss.item()
        sens_loss += sens_loss.item()
        training_loss += loss.item()

    recon_loss /= len(dataloader)
    sens_loss /= len(dataloader)
    training_loss /= len(dataloader)

    return recon_loss, sens_loss, training_loss


def cor_test(model, dataloader, device, recon_criterion, sens_criterion, alpha, beta):
    '''
    Evaluation method for fair representation that does NOT use a group based regulaizer.
    '''
    model.eval()
    testing_loss = 0

    for batch_idx, (X_i, S_i) in enumerate(dataloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        recon_output = model(X_i)
        recon_loss = recon_criterion(recon_output, X_i)
        sens_loss = sens_criterion(recon_output, X_i, S_i)
        loss = alpha * recon_loss + beta * sens_loss
        testing_loss += loss.item()

    testing_loss /= len(dataloader)
    return testing_loss


def cor_sens_criterion(X_hat, X, S):
    '''
    Correlation based regularizer for sensitive/protected attribute.
    '''
    ell = torch.pow(X_hat - X, 2).sum(dim=1)
    l_centered = ell - ell.mean()
    S_centered = S.type(torch.FloatTensor) - S.type(torch.FloatTensor).mean()
    return torch.abs(torch.dot(l_centered, S_centered) / torch.sqrt(
        torch.dot(l_centered, l_centered) * torch.dot(S_centered, S_centered)))


def group_corr_criterion(X_hat, X, S, AE_scores):
    '''
    This reguralizer promotes correlation of reconstruction error per-group with
    corrsponding unregularized autoencoder OD scores_local
    '''
    ind_1 = [i for i, s_i in enumerate(list(S)) if s_i == 1]
    ind_0 = [i for i, s_i in enumerate(list(S)) if s_i == 0]

    ell = torch.pow(X_hat - X, 2).sum(dim=1)
    ell_1 = ell[ind_1]
    ell_1_centered = ell_1 - ell_1.mean()

    scores_1 = AE_scores[ind_1]
    scores_1_centered = scores_1 - scores_1.mean()

    ell_0 = ell[ind_0]
    ell_0_centered = ell_0 - ell_0.mean()

    scores_0 = AE_scores[ind_0]
    scores_0_centered = scores_0 - scores_0.mean()

    corr_0 = torch.dot(ell_0_centered, scores_0_centered) / torch.sqrt(
        torch.dot(ell_0_centered, ell_0_centered) * torch.dot(scores_0_centered, scores_0_centered))
    corr_1 = torch.dot(ell_1_centered, scores_1_centered) / torch.sqrt(
        torch.dot(ell_1_centered, ell_1_centered) * torch.dot(scores_1_centered, scores_1_centered))
    return - 1.0 * (corr_0 + corr_1)


def approx_ndcg_loss(f, s, p=2, scale=1):
    '''
    Computes approx NDCG
    '''
    sigm = torch.nn.Sigmoid()

    numerator = torch.pow(p, s) - 1.0
    denominator = torch.log2(1.0 + (sigm(-scale * (f.unsqueeze(dim=1) * torch.ones([1, len(f)]) -
                                                   (f.unsqueeze(dim=1) * torch.ones([1, len(f)])).transpose(0,
                                                                                                            1)))).sum(
        dim=1))
    value = (numerator / denominator).sum()
    return value


def group_ndcg_criterion(X_hat, X, S, AE_scores, ndcg_majority_const, ndcg_minority_const):
    '''
     This reguralizer promotes rank preservation of reconstruction error per-group with corresponding
     unregularized autoencoder OD scores.
     Proposed GroupFidelity criterion.
    '''
    ind_1 = [i for i,s_i in enumerate(list(S)) if s_i == 1 ]
    ind_0 = [i for i,s_i in enumerate(list(S)) if s_i == 0 ]

    ell = torch.pow(X_hat - X, 2).sum( dim =1)

    ell_1 = ell[ind_1]
    ell_0 = ell[ind_0]

    a_1 = AE_scores[ind_1]
    a_0 = AE_scores[ind_0]

    minority_ndcg = (1.0 / ndcg_minority_const) * approx_ndcg_loss(ell_1, a_1, p=64, scale=100)
    majority_ndcg = (1.0/ ndcg_majority_const) * approx_ndcg_loss(ell_0, a_0, p=64, scale=100)
    return_val = (1-minority_ndcg) + (1-majority_ndcg)
    return return_val


def evaluate(y_true, S, scores):
    # store evaluated metrics as a dict
    results = {}

    # per group true outliers
    y_true_majority = y_true[S == 0]
    y_true_minority = y_true[S == 1]

    # count how many outliers
    n_flagged = sum(y_true == 1)

    # find top flagged indices
    top_k = np.argsort(-scores)[:n_flagged]

    maj_outliers = [r for r in top_k if S[r] == 0.]
    min_outliers = [r for r in top_k if S[r] == 1.]

    # evaluate ROC ==> tuple (majority_roc, minority_roc)
    majority_roc = roc_auc_score(y_true_majority, scores[S == 0])
    minority_roc = roc_auc_score(y_true_minority, scores[S == 1])
    results["roc"] = (majority_roc, minority_roc)

    # evaluate AP
    majority_ap = average_precision_score(y_true_majority, scores[S == 0])
    minority_ap = average_precision_score(y_true_minority, scores[S == 1])
    results["ap"] = (majority_ap, minority_ap)

    # evaluate flag-rates ==> tuple (majority_flag_rate, minority_flag_rate)
    n_majority = int(sum(S == 0))
    n_minority = int(sum(S == 1))

    majority_flag = (len(maj_outliers) / n_flagged) * n_flagged / n_majority
    minority_flag = (len(min_outliers) / n_flagged) * n_flagged / n_minority

    results["flag_rate"] = (majority_flag, minority_flag)

    return results
