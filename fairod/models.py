import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from scipy.stats import rankdata
import os

from utility import column_wise_norm, load_data, group_train, group_test, cor_train, cor_test

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class RepresentationLearner(nn.Module):
    '''
    Creates the autoencoder network for anomaly detection.
    Depending on different training methods, the network parameters will be learned.
    '''

    def __init__(self, layer_dims):
        '''
        Initializing the autoencoder network structure.
        '''
        super(RepresentationLearner, self).__init__()
        layer_structures = list(zip(layer_dims, layer_dims[1:]))
        g = []

        for layer_id, (input_dim, output_dim) in enumerate(layer_structures):
            g.append(nn.Linear(input_dim, output_dim, bias=True))
            g.append(nn.ReLU())
        self.g = nn.Sequential(*g)  # produces the latent vector

        f = []
        reverse_layer_structure = list(zip(layer_dims[1:], layer_dims))[::-1][-1]

        for layer_id, (input_dim, output_dim) in enumerate(reverse_layer_structure):
            f.append(nn.Linear(input_dim, output_dim, bias=True))
            f.append(nn.ReLU())
        f.append(nn.Linear(output_dim, layer_dims[0], bias=True))  # add linear from send last to input dim

        self.f = nn.Sequential(*f)  # reconstructor for input

    def forward(self, x):
        gx = self.g(x)
        reconstruction = self.f(gx)
        return reconstruction


def get_model_object(model_type, layer_dims):
    # model type is not used. However, in future, we may intend to invoke different structures,
    # where the model type can help in creating respective network structures
    return RepresentationLearner(layer_dims)


class FTrainer:
    def __init__(self, config):
        '''
        Keys in the config:
        model_type: takes values in ["base", "fairL", "fairC", "fairOD"]
        recon_criterion: loss function for reconstruction error
        sens_criterion: loss function for correlation based statistical parity
        group_criterion: loss function for group fidelity. See Eq. 13 in the paper.
        data: directory path to dataset. Directory contains X.pkl, y.pkl, pv.pkl.
                The .pkl files should be saved from an input dataset.
        epochs: training epochs for the learner
        lr: learning rate used in adam optimizer
        model_path: path to save the best model
        base_model_weights: path to saved parameters from the base autoencoder model
        base_model_scores: anomaly scores from base model. ndarray.
        niters: number of times repeating the experiment to reduce variation over random initializations
        n_splits: cross-validation splits
        n_hidden: number of nodes in a hidden layer
        params: list of parameter tuples
        '''
        self.model_type = config["model_type"]
        self.recon_criterion = config["recon_criterion"]
        self.sens_criterion = config["sens_criterion"]
        self.group_criterion = config["group_criterion"]
        self.data_path = config["data"]
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.model_path = config["model_path"]  # if not None, then use this path to save the model
        self.base_model_weights = config[
            "base_model_weights"]  # final network weights of base model_type (base_weights.pth)
        self.base_model_scores = config[
            "base_model_scores"]  # anomaly scores for each instance by base model

        # how many re-runs with random initializations of weights for AE
        # for fair AE load the final AE weights corresponding to each iteration
        self.niters = config["niters"]

        self.n_splits = config["n_splits"]  # for recording cross-validation
        self.n_hidden = config["n_hidden"]  # network hidden layer nodes

        # list of tuples (alpha, gamma). See paper for definition of alpha and gamma
        # first entry in the list is a string "base" to denote no regularization
        self.params = config["params"]

        # storing anomaly scores
        self.scores = None

    def train(self):
        """
        Full training logic
        """
        # returns anomaly scores from best model
        residuals = None

        # load dataset -- stored as pickled files
        X, S, y = load_data(self.data_path)

        X = column_wise_norm(torch.FloatTensor(X))
        S = torch.LongTensor(S).flatten()

        # load scores for each instance from base model as a Tensor
        AE_scores = torch.Tensor(self.base_model_scores).flatten()

        # create tensor dataset
        dataset = TensorDataset(X, S, AE_scores)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        n_samples = len(dataset)
        train_size = int(0.8 * n_samples)  # using 80% split in each random cross-validation
        test_size = n_samples - train_size

        # if model type is FairOD, the use ndcg normalizer
        if self.model_type == "fairOD":
            ndcg_norm_maj = np.sum(
                (np.power(2.0, AE_scores.numpy()[S == 0]) - 1.0) / np.log2(rankdata(-AE_scores.numpy()[S == 0]) + 1.0))
            ndcg_norm_min = np.sum(
                (np.power(2.0, AE_scores.numpy()[S == 1]) - 1.0) / np.log2(rankdata(-AE_scores.numpy()[S == 1]) + 1.0))

        else:
            ndcg_norm_maj, ndcg_norm_min = None, None

        # record errors for each parameter across iterations
        total_loss = {}
        construction_loss = {}
        protected_loss = {}
        ranking_loss = {}
        for param in self.params:
            total_loss[param] = np.zeros(self.epochs)
            construction_loss[param] = np.zeros(self.epochs)
            protected_loss[param] = np.zeros(self.epochs)
            ranking_loss[param] = np.zeros(self.epochs)

        for iter_ in range(self.niters):
            # best cross validated loss across params
            best_test_loss = 1e10

            for param in self.params:
                alpha, gamma = param
                beta = 1.0 - alpha

                avg_split_loss = 0
                for split_ in range(self.n_splits):
                    trainset, testset = random_split(dataset, [train_size, test_size])
                    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
                    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

                    # instantiate model_type weights from the base models
                    model = get_model_object(self.model_type, layer_dims=[X.shape[1], self.n_hidden,
                                                                          self.n_hidden])
                    model.load_state_dict(torch.load(self.base_model_weights))
                    model.to(device)

                    # model optimizer
                    optimizer = optim.Adam(model.parameters(), lr=self.lr)

                    for epoch in range(self.epochs):
                        # train the model_type for given number of epochs
                        _, _, _, _ = group_train(model=model,
                                                 dataloader=trainloader,
                                                 device=device,
                                                 recon_criterion=self.recon_criterion,
                                                 sens_croterion=self.sens_criterion,
                                                 group_criterion=self.group_criterion,
                                                 optimizer=optimizer,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 gamma=gamma,
                                                 ndcg_norm_maj=ndcg_norm_maj,
                                                 ndcg_norm_min = ndcg_norm_min
                        )
                    # record test loss for this split
                    avg_split_loss += group_test(model=model,
                                                 dataloader=testloader,
                                                 device=device,
                                                 recon_criterion=self.recon_criterion,
                                                 sens_criterion=self.sens_criterion,
                                                 group_criterion=self.group_criterion,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 gamma=gamma,
                                                 ndcg_norm_maj=ndcg_norm_maj,
                                                 ndcg_norm_min=ndcg_norm_min
                                                 )

                # average validation loss for this param
                avg_split_loss = avg_split_loss / self.n_splits

                # if this param has best loss then train on full data and store the model and record anomaly scores
                if avg_split_loss < best_test_loss:
                    # set new best as avg loss for the current best parameter
                    best_test_loss = avg_split_loss

                    # instantiate model now that we know the best parameter so far
                    model = get_model_object(self.model_type, layer_dims=[X.shape[1], self.n_hidden,
                                                                                              self.n_hidden])
                    model.load_state_dict(torch.load(self.base_model_weights))
                    model.to(device)

                    # model optimizer
                    optimizer = optim.Adam(model.parameters(), lr=self.lr)

                    # train on full dataset
                    _, _, _, _ = group_train(model=model,
                                             dataloader=dataloader,  # full dataset
                                             device=device,
                                             recon_criterion=self.recon_criterion,
                                             sens_croterion=self.sens_criterion,
                                             group_criterion=self.group_criterion,
                                             optimizer=optimizer,
                                             alpha=alpha,
                                             beta=beta,
                                             gamma=gamma,
                                             ndcg_norm_maj=ndcg_norm_maj,
                                             ndcg_norm_min=ndcg_norm_min
                                             )

                    # save the best model_type
                    if self.model_path:
                        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                        torch.save(model.state_dict(), self.model_path)

                    # compute anomaly scores
                    model.eval()
                    this_X_pred = model(X.to(device))
                    residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),
                                               axis=1)

        self.scores = residuals


class BTrainer:
    def __init__(self, config):
        '''
        Keys in the config:
        model_type: takes values in ["base", "fairL"]
        recon_criterion: loss function for reconstruction error
        sens_criterion: loss function for correlation based statistical parity
        data: directory path to dataset. Directory contains X.pkl, y.pkl, pv.pkl.
                The .pkl files should be saved from an input dataset.
        epochs: training epochs for the learner
        lr: learning rate used in adam optimizer
        model_path: path to save the best model
        base_model_weights: path to saved parameters from the base autoencoder model if model_type="fairL"
        niters: number of times repeating the experiment to reduce variation over random initializations
        n_splits: cross-validation splits
        n_hidden: number of nodes in a hidden layer
        params: list of parameter values. parameter = alpha
        '''
        self.model_type = config["model_type"]
        self.recon_criterion = config["recon_criterion"]
        self.sens_criterion = config["sens_criterion"]

        self.data_path = config["data"]
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.model_path = config["model_path"]  # if not None, then use this path to save the model
        self.base_model_weights = config[
            "base_model_weights"]  # final network weights of base model_type (base_weights.pth)

        # how many re-runs with random initializations of weights for AE
        self.niters = config["niters"]

        self.n_splits = config["n_splits"]  # for recording cross-validation
        self.n_hidden = config["n_hidden"]  # network hidden layer nodes

        # list of tuples (alpha, gamma). See paper for definition of alpha and gamma
        # first entry in the list is a string "base" to denote no regularization
        self.params = config["params"]

        # storing anomaly scores
        self.scores = None

    def train(self):
        """
        Full training logic
        """
        # returns anomaly scores from best model
        residuals = None

        # load dataset -- stored as pickled files
        X, S, y = load_data(self.data_path)

        X = column_wise_norm(torch.FloatTensor(X))
        S = torch.LongTensor(S).flatten()

        # create tensor dataset
        dataset = TensorDataset(X, S)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        n_samples = len(dataset)
        train_size = int(0.8 * n_samples)  # using 80% split in each random cross-validation
        test_size = n_samples - train_size

        # record errors for each parameter across iterations
        total_loss = {}
        construction_loss = {}
        protected_loss = {}
        for param in self.params:
            total_loss[param] = np.zeros(self.epochs)
            construction_loss[param] = np.zeros(self.epochs)
            protected_loss[param] = np.zeros(self.epochs)

        for iter_ in range(self.niters):
            # best cross validated loss across params
            best_test_loss = 1e10

            for param in self.params: # here params will only have alpha parameter
                alpha = param
                beta = 1.0 - alpha

                avg_split_loss = 0
                for split_ in range(self.n_splits):
                    trainset, testset = random_split(dataset, [train_size, test_size])
                    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
                    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

                    # instantiate model_type weights from the base models
                    model = get_model_object(self.model_type, layer_dims=[X.shape[1], self.n_hidden,
                                                                          self.n_hidden])

                    # get model optimizer
                    optimizer = optim.Adam(model.parameters(), lr=self.lr)

                    if self.base_model_weights:
                        model.load_state_dict(torch.load(self.base_model_weights))
                    model.to(device)
                    # recon_criterion = nn.MSELoss()
                    # sens_criterion = nn.CrossEntropyLoss()


                    for epoch in range(self.epochs):
                        # train the model_type for given number of epochs
                        _, _, _ = cor_train(model=model,
                                            dataloader=trainloader,
                                            device=device,
                                            recon_criterion=self.recon_criterion,
                                            sens_criterion=self.sens_criterion,
                                            optimizer=optimizer,
                                            alpha=alpha,
                                            beta=beta)
                    # record test loss for this split
                    avg_split_loss += cor_test(model=model,
                                               dataloader=testloader,
                                               device=device,
                                               recon_criterion=self.recon_criterion,
                                               sens_criterion=self.sens_criterion,
                                               alpha=alpha,
                                               beta=beta)

                # average validation loss for this param
                avg_split_loss = avg_split_loss / self.n_splits

                # if this param has best loss then store the model and record anomaly scores
                if avg_split_loss < best_test_loss:
                    # set new best as avg loss for the current best parameter
                    best_test_loss = avg_split_loss

                    # instantiate model now that we know the best parameter so far
                    model = get_model_object(self.model_type, layer_dims=[X.shape[1], self.n_hidden,
                                                                                              self.n_hidden])
                    optimizer = optim.Adam(model.parameters(), lr=self.lr)
                    if self.base_model_weights:
                        model.load_state_dict(torch.load(self.base_model_weights))
                    model.to(device)

                    # train on full dataset
                    _, _, _ = cor_train(model=model,
                                             dataloader=dataloader, # full dataset
                                             device=device,
                                             recon_criterion=self.recon_criterion,
                                             sens_criterion=self.sens_criterion,
                                             optimizer=optimizer,
                                             alpha=alpha,
                                             beta=beta)

                    # save the best model_type
                    if self.model_path:
                        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                        torch.save(model.state_dict(), self.model_path)

                    # compute anomaly scores
                    model.eval()
                    this_X_pred = model(X.to(device))
                    residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),
                                               axis=1)

        self.scores = residuals
