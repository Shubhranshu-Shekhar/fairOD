import sys
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import expon
from sklearn.metrics import average_precision_score as AP_score
import numpy.random as rnd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score as AP_score
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import os
import sys

# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# np.random.seed(0)

script_dir = os.path.dirname(__file__)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '~/twitteraae/code')
# import predict, twokenize

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


def column_wise_norm(x):
    x_normed = (x - x.min(0, keepdim=True)[0]) / (x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0])
    return x_normed


def cor_sens_criterion(X_hat, X, S):
    # torch.manual_seed(0)
    ell = torch.pow(X_hat - X, 2).sum( dim =1)

    l_centered = ell - ell.mean()

    S_centered = S.type(torch.FloatTensor) - S.type(torch.FloatTensor).mean()

    return torch.abs(torch.dot( l_centered, S_centered )/torch.sqrt(torch.dot(l_centered, l_centered) * torch.dot(S_centered, S_centered)))

def sp(scores, groups):
    mean_s, mean_g = np.mean(scores), np.mean(groups)

    # centering of the scores and sensitive attribute
    scores_centered = scores - mean_s
    groups_centered = groups - mean_g

    sp = np.abs(np.dot(scores_centered, groups_centered) / np.sqrt(
        np.dot(scores_centered, scores_centered) * np.dot(groups_centered, groups_centered)))
    return sp




def cor_train(alpha, beta):
    # torch.manual_seed(0)
    model.train()
    training_loss = 0
    recon_loss = 0
    sens_loss = 0
    for batch_idx, (X_i, S_i) in enumerate(trainloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        cor_optimizer.zero_grad()
        recon_output = model(X_i)
        recon_loss =  recon_criterion(recon_output, X_i)
        sens_loss =  cor_sens_criterion(recon_output, X_i, S_i)
        loss = alpha * recon_loss + beta * sens_loss
        loss.backward()
        cor_optimizer.step()
        recon_loss += recon_loss.item()
        sens_loss += sens_loss.item()
        training_loss += loss.item()
    
    recon_loss /= len(trainloader.dataset)
    sens_loss /= len(trainloader.dataset)
    training_loss /= len(trainloader.dataset)
    return recon_loss, sens_loss, training_loss
    

def cor_test(alpha, beta):
    # torch.manual_seed(0)
    model.eval()
    sens_loss = 0
    recon_loss = 0    
    testing_loss = 0

    for batch_idx, (X_i, S_i) in enumerate(testloader):
        X_i, S_i = X_i.to(device), S_i.to(device)
        recon_output = model(X_i)
        recon_loss =  recon_criterion(recon_output, X_i)
        sens_loss =  cor_sens_criterion(recon_output, X_i, S_i)
        loss = alpha *recon_loss + beta * sens_loss
        recon_loss += recon_loss.item()
        sens_loss += sens_loss.item()        
        testing_loss += loss.item()

    recon_loss /= len(trainloader.dataset)
    sens_loss /= len(trainloader.dataset)        
    testing_loss /= len(testloader.dataset)
    return recon_loss, sens_loss, testing_loss



'''
Main running part of this code
'''
# torch.manual_seed(0)
# ds_identifier = '_m1f5'  # '_m5f5'  # '_m1f5', '_m5f1'
br = str(sys.argv[-1])

base_path = os.path.join(script_dir, '../data/')
print('base path:', base_path)

with open(base_path + 'X' + br + '.pkl', 'rb') as f:
    X = pickle.load( f)
X_copy = np.copy(X)

with open(base_path + 'pv' + br + '.pkl', 'rb') as f:
    S = pickle.load(f)
with open(base_path + 'y' + br + '.pkl', 'rb') as f:
    y = pickle.load(f)

score_dir = os.path.join(script_dir, '../model_data/')
print('score path:', score_dir)

S_ = S.copy()

y_true = y.tolist()
true_outliers = set([i for i, l in enumerate(y_true) if l == 1. ])

y_true_male = y[S == 0]
y_true_female = y[S == 1]

n_male = int(sum(S == 0))
n_female = int(sum(S == 1))
print('#Male:', n_male, '#Femal:', n_female)

n_male_o = int(sum((S == 0) & (y == 1)))
n_female_o = int(sum((S == 1) & (y == 1)))
print('#Male Outliers:', n_male_o, '#Femal Outliers:', n_female_o)


N_splits = 1
N_iters = 1 #100
epochs = 200
# n_hidden = 4
n_hidden = int(sys.argv[-2])
print("Embedding=", n_hidden)

# N_o = 100 # = N_flagged
N_o = sum(y == 1)

# alphas = [0.01, 0.1, 0.5, 0.9, 0.99]
alphas = [0.01, 0.5, 0.9]

ranks = {'RAW': [y_true] }
ranks['RAW'].append( S_.tolist() )  # ranks['RAW'].append( [0]*n_male + [1]*n_female )
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

for alpha_ in alphas:
    ranks[alpha_] = [y_true]
    ranks[alpha_].append(  S_.tolist() )     # [0]*n_male + [1]*n_female
    flag_rates[alpha_] = np.zeros( [N_iters, 2 ] )
    group_precision[alpha_] = np.zeros( [N_iters, 2 ] )
    group_recall[alpha_] = np.zeros( [N_iters, 2 ] )
    group_AP[alpha_] = np.zeros( [N_iters, 2 ] )
    AP[alpha_] = np.zeros( [N_iters, 1 ] )
    K_tau[alpha_] = np.zeros( [N_iters, 2 ] )

    total_loss[alpha_] = np.zeros(epochs)
    construction_loss[alpha_] = np.zeros(epochs)
    protected_loss[alpha_] = np.zeros(epochs)
    ranking_loss[alpha_] = np.zeros(epochs)

    scores_over_epoch[alpha_] = []


for iter_ in range(N_iters):

    print(' ')
    print('Iter: ', iter_ + 1)
    print(' ')

    # LOAD DATA

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


    # Run OD with simple autoencoder (a == 1)
    #############################################################################################

    alpha, beta = 1.0, 0.0

    # n_hidden = 2

    best_test_loss = 1e10
    best_model = None
    for split_ in range(N_splits):

        trainset, testset = random_split(dataset, [train_size, test_size])
        trainset = dataset

        trainloader = DataLoader(trainset, batch_size = 128, shuffle=True)
        testloader = DataLoader(testset, batch_size = len(testset), shuffle=False)


        # instantiate model, loss and optimizer

        model = UncorrelatedRepresentationLearner(layer_dims=[X.shape[1], n_hidden, n_hidden])
        model.to(device)
        recon_criterion = nn.MSELoss()
#                sens_criterion = nn.CrossEntropyLoss()
        cor_optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            recon_train, sens_train, train_loss = cor_train(alpha, beta)
            recon_test, sens_test, test_loss = cor_test(alpha, beta)
            construction_loss['RAW'][epoch] += recon_train
            protected_loss['RAW'][epoch] += sens_train
            total_loss['RAW'][epoch] += train_loss

            run_pred_ = model(X_copy)
            scores_over_epoch['RAW'].append(np.linalg.norm(X_copy.cpu().detach().numpy() -
                                                           run_pred_.cpu().detach().numpy(), axis=1))

            if True and (epoch+1) % 10 == 0:
                print('[Epoch {}] Recon Train {:.8f}, Sens Train {:.8f}, Test Loss: {:.8f}'.format(epoch, recon_train, sens_train, test_loss ))
            
            if recon_train < 0.00001:
                break

        model.eval()
        X = X.to(device)

        this_X_pred = model(X)

        this_residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),axis=1)

        # we should also keep track of best model...which I'm not doing currently
        best_model = model  # this should inside the following if condition--ignoring for now
        if test_loss < best_test_loss:
            X_pred = this_X_pred
            residuals = this_residuals
            best_test_loss = test_loss



    # Interpret AE Reconstruction Erros as OD scores_local
    ############################################################################################

    # save the best model cross-validated-- though currently I'm only saving the latest model
    # saving state_dict() of 'model' -- containing only the weights
    os.makedirs(score_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(score_dir, str(iter_) + 'weights_only.pth'))

    scores = None
    try:
        scores = residuals
    except:
        scores = this_residuals

    # save scores_local to be read by other models
    os.makedirs(score_dir, exist_ok=True)
    np.save(os.path.join(score_dir, str(iter_) + '_scores'), scores)

    print('Original data AP: ', AP_score(y_true=y_true, y_score= scores) )

    ranks['RAW'].append( list(np.argsort(-scores)))
    ranks['RAW'].append(scores)

    top_k = np.argsort(-scores)[:N_o]

    male_o = [ r for r in top_k if S_[r] == 0.]
    female_o = [r for r in top_k if S_[r] == 1.]

    pr_m_o = len(male_o)/N_o
    pr_f_o = len(female_o)/N_o

    # try:
    fr_m = pr_m_o * N_o / n_male
    fr_f = pr_f_o * N_o / n_female
    print('Flag_rates', fr_m, fr_f)
    print( 'Original Flag rate ratio (W/AA): ', (pr_m_o/pr_f_o) * (n_female/n_male), fr_m/fr_f)

    flag_rates['RAW'][iter_,0]  = pr_m_o * N_o / n_male
    flag_rates['RAW'][iter_,1]  = pr_f_o * N_o / n_female

    group_precision['RAW'][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / len(male_o )
    group_precision['RAW'][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / len(female_o )

    group_recall['RAW'][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / n_male_o
    group_recall['RAW'][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / n_female_o

    group_AP['RAW'][iter_, 0] = AP_score(y_true_male, scores[S == 0])
    group_AP['RAW'][iter_, 1] = AP_score(y_true_female, scores[S == 1])

    AP['RAW'][iter_] = AP_score( y_true, scores )
    # except Exception as e:
    #     print (str(e))

    original_male_scores, original_female_scores = scores[S == 0], scores[S == 1]


    # COR exps
    ############################################################################################

    for alpha in alphas:

        beta = 1.0 - alpha

        # n_hidden = 2

        best_test_loss = 1e10

        for split_ in range(N_splits):

            trainset, testset = random_split(dataset, [train_size, test_size])
            trainset = dataset

            trainloader = DataLoader(trainset, batch_size = 128, shuffle=True)
            testloader = DataLoader(testset, batch_size = len(testset), shuffle=False)


            # instantiate model, loss and optimizer

            model = UncorrelatedRepresentationLearner(layer_dims=[X.shape[1], n_hidden, n_hidden])
            model.to(device)
            recon_criterion = nn.MSELoss()
    #                sens_criterion = nn.CrossEntropyLoss()
            cor_optimizer = optim.Adam(model.parameters(), lr=0.001)


            for epoch in range(epochs):
                recon_train, sens_train, train_loss = cor_train(alpha, beta)
                recon_test, sens_test, test_loss = cor_test(alpha, beta)

                construction_loss[alpha][epoch] += recon_train
                protected_loss[alpha][epoch] += sens_train
                total_loss[alpha][epoch] += train_loss

                if True and epoch % 10 == 0:
                    print('[Epoch {}] Recon Train {:.8f}, Sens Train {:.8f}, Test Loss: {:.8f}'.format(epoch, recon_train, sens_train, test_loss ))

            model.eval()
            X = X.to(device)

            this_X_pred = model(X)

            this_residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),axis=1)

            if test_loss < best_test_loss:
                X_pred = this_X_pred
                residuals = this_residuals
                best_test_loss = test_loss



        # Interpret COR Reconstruction Erros as OD scores_local
        ############################################################################################

        scores = residuals

        print('COR data AP: ', AP_score(y_true=y_true, y_score= scores) )

        ranks[alpha].append( list(np.argsort(-scores)))
        ranks[alpha].append(scores)

        top_k = np.argsort(-scores)[:N_o]

        # male_o = [ r for r in top_k if r < n_male ]
        # female_o = [ r for r in top_k if r >= n_male ]

        male_o = [r for r in top_k if S_[r] == 0.]
        female_o = [r for r in top_k if S_[r] == 1.]

        pr_m_o = len(male_o)/N_o
        pr_f_o = len(female_o)/N_o

        try:
            print( 'COR Flag rate ratio (M/F): ', (pr_m_o/pr_f_o) * (n_female/n_male))

            flag_rates[alpha][iter_,0]  = pr_m_o * N_o / n_male
            flag_rates[alpha][iter_,1]  = pr_f_o * N_o / n_female


            group_precision[alpha][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / len(male_o )
            group_precision[alpha][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / len(female_o )

            group_recall[alpha][iter_,0]  = len([ i for i in male_o if i in true_outliers]) / n_male_o
            group_recall[alpha][iter_,1]  = len([ i for i in female_o if i in true_outliers]) / n_female_o

            group_AP[alpha][iter_, 0] = AP_score(y_true_male, scores[S == 0])
            group_AP[alpha][iter_, 1] = AP_score(y_true_female, scores[S == 1])

            AP[alpha][iter_] = AP_score( y_true, scores )

            cor_male_scores, cor_female_scores = scores[S == 0], scores[S == 1]

            # Store Kendals tau of the before/after rankings at each row: [male, female]
            K_tau[alpha][iter_, 0] = kendalltau(rankdata(original_male_scores), rankdata(cor_male_scores))[0]
            K_tau[alpha][iter_,1] = kendalltau(rankdata(original_female_scores), rankdata(cor_female_scores))[0]
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


