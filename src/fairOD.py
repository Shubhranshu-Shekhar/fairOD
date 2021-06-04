import sys
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy.random as rnd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score as AP_score
from scipy.stats import rankdata

import pickle

import seaborn as sns
sns.set(style="white")

import os

# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# np.random.seed(0)

script_dir = os.path.dirname(__file__)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class UncorrelatedRepresentationLearner(nn.Module):
    '''
    Based on the idea from:  https://arxiv.org/pdf/1901.04562.pdf 
    Trying to uncorrelated the outcome (here reconstruction error) from sensitive attribute 
    '''
    def __init__(self, layer_dims):
        # torch.manual_seed(0)
        super(UncorrelatedRepresentationLearner, self).__init__()
        layer_structures = list(zip(layer_dims, layer_dims[1:]))
        n_layers = len(layer_structures)
        g = []
        
        for layer_id, (input_dim, output_dim) in enumerate(layer_structures):
            g.append(nn.Linear(input_dim, output_dim, bias=True))
            g.append(nn.ReLU())
        self.g = nn.Sequential(*g) # produces the latent vector
        
        self.f = nn.Sequential(nn.Linear(output_dim, output_dim, bias=True),
                               nn.ReLU(),
                               nn.Linear(output_dim, layer_dims[0], bias=True)
                              ) # reconstructor for input 
            
        
    def forward(self, x):
        gx = self.g(x)
        reconstruction = self.f(gx)
        return reconstruction



def cor_sens_criterion(X_hat, X, S):
    # torch.manual_seed(0)
    ell = torch.pow(X_hat - X, 2).sum( dim =1)

    l_centered = ell - ell.mean()

    S_centered = S.type(torch.FloatTensor) - S.type(torch.FloatTensor).mean()

    return torch.abs(torch.dot( l_centered, S_centered )/torch.sqrt(torch.dot(l_centered, l_centered) * torch.dot(S_centered, S_centered)))


class FairRepresentationLearner(nn.Module):
    '''
    Based on the idea from:  https://arxiv.org/pdf/1901.04562.pdf 
    Trying to uncorrelated the outcome (here reconstruction error) from sensitive attribute 
    '''
    def __init__(self, layer_dims):
        super(FairRepresentationLearner, self).__init__()
        layer_structures = list(zip(layer_dims, layer_dims[1:]))
        n_layers = len(layer_structures)
        g = []
        
        for layer_id, (input_dim, output_dim) in enumerate(layer_structures):
            g.append(nn.Linear(input_dim, output_dim, bias=True))
            g.append(nn.ReLU())
        self.g = nn.Sequential(*g) # produces the latent vector
        
        self.f = nn.Sequential(nn.Linear(output_dim, output_dim, bias=True),
                               nn.ReLU(),
                               nn.Linear(output_dim, layer_dims[0], bias=True)
                              ) # reconstructor for input 

    def forward(self, x):
        gx = self.g(x)
        reconstruction = self.f(gx)
        return reconstruction


# def approx_ndcg_loss(f,s):
#
#     sigm = torch.nn.Sigmoid()
#
#     numerator = torch.pow(2,s) - 1.0
#
#     denominator = torch.log2( 1.0 +  (sigm( f.unsqueeze(dim=1)*torch.ones([1,len(f)]) - (f.unsqueeze(dim=1)*torch.ones([1,len(f)])).transpose(0,1)  )).sum(dim=1)  )
#
#     return (numerator / denominator).sum()


# def get_idcg(ref_scores, p=2):
#     if isinstance(ref_scores, np.ndarray):
#         ref_scores = torch.from_numpy(ref_scores)
#
#     idcg = np.sum((np.power(p, ref_scores.numpy()) - 1.0) / np.log2(rankdata(-ref_scores) + 1.0))
#     # print("IDCG:", idcg)
#     return idcg


def approx_ndcg_loss(f, s, p=2, scale=1):
    # if isinstance(f, np.ndarray):
    #     f = torch.from_numpy(f)
    #     s = torch.from_numpy(s)

    sigm = torch.nn.Sigmoid()

    numerator = torch.pow(p, s) - 1.0
    # print(s)
    # print(numerator)
    denominator = torch.log2(1.0 + (sigm(-scale * (f.unsqueeze(dim=1) * torch.ones([1, len(f)]) -
                                                   (f.unsqueeze(dim=1) * torch.ones([1, len(f)])).transpose(0,
                                                                                                            1)))).sum(
        dim=1))

    value = (numerator / denominator).sum()
    # ndcg_norm = get_idcg(s, p)
    return value


def cor_group_criterion(X_hat, X, S, AE_scores):
    '''
     This reguralizer promotes correlation of reconstruction error per-group with corrsponding unregularize autoencoder OD scores_local
    '''    

    ind_1 = [i for i,s_i in enumerate(list(S)) if s_i == 1 ]
    ind_0 = [i for i,s_i in enumerate(list(S)) if s_i == 0 ]

    ell = torch.pow(X_hat - X, 2).sum( dim =1)

    ell_1 = ell[ind_1]
    ell_0 = ell[ind_0]

    a_1 =  AE_scores[ind_1]
    a_0 = AE_scores[ind_0]

    female_ndcg = (1.0 / ndcg_norm_females) * approx_ndcg_loss(ell_1,a_1, p=64, scale=100)
    male_ndcg = (1.0/ ndcg_norm_males) * approx_ndcg_loss(ell_0,a_0,p=64, scale=100)
    return_val = (1-female_ndcg) + (1-male_ndcg)
    return return_val


def cor_group_train(alpha, beta, gamma):
    model.train()
    training_loss = 0
    recon_loss = 0
    sens_loss = 0
    group_loss = 0
    for batch_idx, (X_i, S_i, A_i) in enumerate(trainloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        cor_group_optimizer.zero_grad()
        recon_output = model(X_i)
        r_loss = recon_criterion(recon_output, X_i)
        s_loss = cor_sens_criterion(recon_output, X_i, S_i)
        g_loss = cor_group_criterion(recon_output, X_i, S_i, A_i)
        loss = alpha * r_loss + beta * s_loss + gamma * g_loss

        loss.backward()
        cor_group_optimizer.step()

        recon_loss += r_loss.item()
        sens_loss += s_loss.item()
        group_loss += g_loss.item()
        training_loss += loss.item()
    
    recon_loss /= len(trainloader)
    sens_loss /= len(trainloader)
    group_loss /= len(trainloader)
    training_loss /= len(trainloader)
    return recon_loss, sens_loss, training_loss, group_loss
    

def cor_group_test(alpha, beta, gamma):
    model.eval()
    sens_loss = 0
    recon_loss = 0    
    testing_loss = 0
    group_loss = 0

    for batch_idx, (X_i, S_i, A_i) in enumerate(testloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        recon_output = model(X_i)
        recon_loss =  recon_criterion(recon_output, X_i)
        sens_loss =  cor_sens_criterion(recon_output, X_i, S_i)
        group_loss = cor_group_criterion(recon_output, X_i, S_i, A_i)        
        loss = alpha *recon_loss + beta * sens_loss + gamma * group_loss
        recon_loss += recon_loss.item()
        sens_loss += sens_loss.item()      
        group_loss += group_loss.item()          
        testing_loss += loss.item()

    recon_loss /= len(trainloader.dataset)
    sens_loss /= len(trainloader.dataset)        
    group_loss /= len(trainloader.dataset)        
    testing_loss /= len(testloader.dataset)
    return recon_loss, sens_loss, testing_loss

def column_wise_norm(x):
    x_normed = (x - x.min(0, keepdim=True)[0]) / (x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0])
    return x_normed


'''
Main running part of this code
'''
# torch.manual_seed(0)
# ds_identifier = name of dataset file to be processed
ds_identifier = str(sys.argv[-1]) #

base_path = os.path.join(script_dir, '../../data/')
print('base path:', base_path)

with open(base_path + 'X' + ds_identifier + '.pkl', 'rb') as f:
    X = pickle.load(f)
X_copy = np.copy(X)

with open(base_path + 'pv' + ds_identifier + '.pkl', 'rb') as f:
    S = pickle.load(f)
with open(base_path + 'y' + ds_identifier + '.pkl', 'rb') as f:
    y = pickle.load(f)

score_dir = os.path.join(script_dir, '../../model_data/')
print('score path:', score_dir)


S_ = S.copy()

y_true = y.tolist()
true_outliers = set([i for i, l in enumerate(y_true) if l == 1.])

y_true_male = y[S == 0]
y_true_female = y[S == 1]

n_male = int(sum(S == 0))
n_female = int(sum(S == 1))
print('#Male:', n_male, '#Femal:', n_female)

n_male_o = int(sum((S == 0) & (y == 1)))
n_female_o = int(sum((S == 1) & (y == 1)))
print('#Male Outliers:', n_male_o, '#Female Outliers:', n_female_o)

# next 3 variables controls for how long is going to run
N_splits = 1 # used 5 for the paper
N_iters = 1  # used 100 for the paper
epochs = 1 # used 200 for the paper
# n_hidden = 8
n_hidden = int(sys.argv[-2])

# N_o = 100 # = N_flagged
N_o = sum(y == 1)
N_flagged = N_o

alpha = 0.5


# gammas = [1.0, 2, 5, 10.0, 100.0, 500.0, 1000]
# gammas = [100.0, 500.0, 10000]
gammas = [0.01, 1, 10, 100, 10000]

alphas = [0.01, 0.1, 0.5, 0.9, 0.99]
# alphas = [0.01, 0.5, 0.9]
# alphas = [0]
params = []
for a_ in alphas:
    for g_ in gammas:
        params.append((a_, g_))

ranks = {'RAW': [y_true] }
ranks['RAW'].append( S_.tolist() )  # [0]*n_male + [1]*n_female
flag_rates = { 'RAW': np.zeros( [N_iters, 2 ] ) }
group_precision = { 'RAW': np.zeros( [N_iters, 2 ] ) }
group_recall = { 'RAW': np.zeros( [N_iters, 2 ] ) }
group_AP = { 'RAW': np.zeros( [N_iters, 2 ] ) }
AP = { 'RAW': np.zeros( [N_iters, 1 ] ) }
K_tau = {  }

total_loss = {'RAW': np.zeros(epochs)}
construction_loss = {'RAW': np.zeros(epochs)}
protected_loss = {'RAW': np.zeros(epochs)}
ranking_loss = {'RAW': np.zeros(epochs)}

# added now
scores_over_epoch = {'RAW': []}

# for gamma_ in gammas:
for param in params:
    ranks[param] = [y_true]
    ranks[param].append( S_.tolist() )
    flag_rates[param] = np.zeros( [N_iters, 2 ] )
    group_precision[param] = np.zeros( [N_iters, 2 ] )
    group_recall[param] = np.zeros( [N_iters, 2 ] )
    group_AP[param] = np.zeros( [N_iters, 2 ] )
    AP[param] = np.zeros( [N_iters, 1 ] )
    K_tau[param] = np.zeros( [N_iters, 2 ] )

    total_loss[param] = np.zeros(epochs)
    construction_loss[param] = np.zeros(epochs)
    protected_loss[param] = np.zeros(epochs)
    ranking_loss[param] = np.zeros(epochs)

    scores_over_epoch[param] = []


for iter_ in range(N_iters):

    print(' ')
    print('Iter: ', iter_ + 1)
    print(' ')

    # data_ = np.load('../../data/samples/sample_'+str(iter_) + '.npy' ).item()

    # X, S = data_['X'],data_['S']

    X = column_wise_norm(torch.FloatTensor(X))
    S = torch.LongTensor(S).flatten()

    X_copy = column_wise_norm(torch.FloatTensor(X_copy))
    X_copy = X_copy.to(device)

    dataset = TensorDataset(X, S)
    n_samples = len(dataset)
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size

    best_score = 1e10



    # load scores_local for RAW
    scores = np.load(os.path.join(score_dir, str(iter_) + '_scores.npy'), allow_pickle=True)

    AE_scores = torch.Tensor(scores).flatten()

    print('Original data AP: ', AP_score(y_true=y_true, y_score= scores))

    ranks['RAW'].append( list(np.argsort(-scores)))
    ranks['RAW'].append(scores)

    top_k = np.argsort(-scores)[:N_flagged]
    bottom_k = np.argsort(scores)[:N_flagged]

    # male_o = [ r for r in top_k if r < n_male ]
    # female_o = [ r for r in top_k if r >= n_male ]
    male_o = [r for r in top_k if S_[r] == 0.]
    female_o = [r for r in top_k if S_[r] == 1.]

    pr_m_o = len(male_o)/N_flagged
    pr_f_o = len(female_o)/N_flagged

    try:
        print( 'Original Flag rate ratio (M/F): ', (pr_m_o/pr_f_o) * (n_female/n_male))

        flag_rates['RAW'][iter_,0]  = pr_m_o * N_flagged / n_male
        flag_rates['RAW'][iter_,1]  = pr_f_o * N_flagged / n_female


        group_precision['RAW'][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / len(male_o )
        group_precision['RAW'][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / len(female_o )

        group_recall['RAW'][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / n_male_o
        group_recall['RAW'][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / n_female_o

        group_AP['RAW'][iter_,0]  = AP_score( y_true_male, scores[S == 0] )
        group_AP['RAW'][iter_,1]  = AP_score( y_true_female , scores[S == 1] )

        AP['RAW'][iter_] = AP_score( y_true, scores )
    except:
        print('Zero encountered')

    # original_male_scores, original_female_scores = scores_local[:n_male], scores_local[n_male:]
    original_male_scores = scores[S_ == 0.]
    original_female_scores = scores[S_ == 1.]


    # COR exps
    ############################################################################################

    dataset = TensorDataset(X, S, AE_scores)
    n_samples = len(dataset)
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size

    ndcg_norm_males = np.sum( (np.power(2.0, AE_scores.numpy()[S_ == 0] ) - 1.0)/np.log2(rankdata(-AE_scores.numpy()[S_ == 0] ) + 1.0) )
    ndcg_norm_females = np.sum( (np.power(2.0, AE_scores.numpy()[S_ == 1]) - 1.0)/np.log2(rankdata(-AE_scores.numpy()[S_ == 1]) + 1.0) )

    cnt = 0
    # for gamma in gammas:
    for param in params:
        alpha, gamma = param
        beta = 1.0 - alpha
        # beta = 0

        # n_hidden = 2

        best_test_loss = 1e10

        for split_ in range(N_splits):

            trainset, testset = random_split(dataset, [train_size, test_size])
            trainset = dataset

            trainloader = DataLoader(trainset, batch_size = 128, shuffle=True)
            testloader = DataLoader(testset, batch_size = len(testset), shuffle=False)


            # instantiate model, loss and optimizer

            model = FairRepresentationLearner(layer_dims=[X.shape[1], n_hidden, n_hidden])
            model.load_state_dict(torch.load(os.path.join(score_dir, str(iter_) + 'weights_only.pth')))
            model.to(device)
            recon_criterion = nn.MSELoss()
            #recon_criterion = nn.L1Loss()
    #                sens_criterion = nn.CrossEntropyLoss()
            cor_group_optimizer = optim.Adam(model.parameters(), lr=0.0005)


            for epoch in range(epochs):
                recon_train, sens_train, train_loss, group_loss = cor_group_train(alpha, beta, gamma)
                recon_test, sens_test, test_loss = cor_group_test(alpha, beta, gamma)

                construction_loss[param][epoch] += recon_train
                protected_loss[param][epoch] += sens_train
                total_loss[param][epoch] += train_loss
                ranking_loss[param][epoch] += group_loss

                run_pred_ = model(X_copy)
                scores_over_epoch[param].append(np.linalg.norm(X_copy.cpu().detach().numpy() -
                                                               run_pred_.cpu().detach().numpy(), axis=1))

                if epoch % 10 == 0:
                    print('[Epoch {}] Recon Train {:.8f}, Sens Train {:.8f}, Test Loss: {:.8f}'.format(epoch, recon_train, sens_train, test_loss ))
                # if recon_train < 0.001:
                #     break

            model.eval()
            X = X.to(device)

            this_X_pred = model(X)

            this_residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(), axis=1)

            if test_loss < best_test_loss:
                X_pred = this_X_pred
                residuals = this_residuals
                best_test_loss = test_loss



        # Interpret COR Reconstruction Erros as OD scores_local
        ############################################################################################

        scores = None
        try:
            scores = residuals
        except:
            scores = this_residuals

        print('COR data AP: ', AP_score(y_true=y_true, y_score= scores) )

        # ranks[gamma].append( list(np.argsort(-scores_local)))
        ranks[param].append(list(np.argsort(-scores)))
        ranks[param].append(scores)

        top_k = np.argsort(-scores)[:N_flagged]
        bottom_k = np.argsort(scores)[:N_flagged]

        # male_o = [ r for r in top_k if r < n_male ]
        # female_o = [ r for r in top_k if r >= n_male ]
        male_o = [r for r in top_k if S_[r] == 0.]
        female_o = [r for r in top_k if S_[r] == 1.]

        pr_m_o = len(male_o)/N_flagged
        pr_f_o = len(female_o)/N_flagged

        try:
            print('COR Flag rate ratio (M/F): ', (pr_m_o/pr_f_o) * (n_female/n_male))

            flag_rates[param][iter_,0]  = pr_m_o * N_flagged / n_male
            flag_rates[param][iter_,1]  = pr_f_o * N_flagged / n_female


            group_precision[param][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / len(male_o )
            group_precision[param][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / len(female_o )

            group_recall[param][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / n_male_o
            group_recall[param][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / n_female_o

            group_AP[param][iter_, 0] = AP_score(y_true_male, scores[S == 0])
            group_AP[param][iter_, 1] = AP_score(y_true_female, scores[S == 1])

            AP[param][iter_] = AP_score(y_true, scores)

            cor_male_scores, cor_female_scores = scores[S == 0], scores[S == 1]

            # Store Kendals tau of the before/after rankings at each row: [male, female]
            K_tau[param][iter_, 0] = kendalltau(rankdata(original_male_scores), rankdata(cor_male_scores))[0]
            K_tau[param][iter_,1] = kendalltau(rankdata(original_female_scores), rankdata(cor_female_scores))[0]
        except:
            print('Zero encountered.')


for key in total_loss:
    construction_loss[key] = construction_loss[key]/(N_iters*N_splits)
    protected_loss[key] = protected_loss[key]/(N_iters*N_splits)
    total_loss[key] = total_loss[key]/(N_iters*N_splits)
    ranking_loss[key] = ranking_loss[key]/(N_iters*N_splits)

# Store all results
store_path = os.path.join(script_dir, 'cor_grid_vs_AE_res/')
os.makedirs(store_path+'params', exist_ok=True)
np.save( os.path.join(store_path, 'params/flag_rates'), flag_rates)
np.save( os.path.join(store_path, 'params/group_precision'), group_precision)
np.save( os.path.join(store_path, 'params/group_recall'), group_recall)
np.save( os.path.join(store_path, 'params/group_AP'), group_AP)
np.save( os.path.join(store_path, 'params/AP'), AP)
np.save( os.path.join(store_path, 'params/ranks'), np.array(ranks))
np.save( os.path.join(store_path, 'params/total_loss'), np.array(total_loss))
np.save( os.path.join(store_path, 'params/construction_loss'), np.array(construction_loss))
np.save( os.path.join(store_path, 'params/protected_loss'), np.array(protected_loss))
np.save( os.path.join(store_path, 'params/ranking_loss'), np.array(ranking_loss))

np.save( os.path.join(store_path, 'params/scores_over_epoch'), np.array(scores_over_epoch))
