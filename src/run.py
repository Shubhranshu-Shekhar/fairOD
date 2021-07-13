from utility import cor_sens_criterion, group_corr_criterion, group_ndcg_criterion, evaluate
from models import *


def run_demo():
    # edit config to train
    # 1. let's train and record base autoencoder. It requires alpha = 1
    config = {"group_criterion": None, "data": "../data/synthetic/1/", "epochs": 1, "lr": 0.001,
              "base_model_weights": None, "base_model_scores": None, "niters": 1, "n_splits": 2, "n_hidden": 2,
              "model_type": "base", "recon_criterion": nn.MSELoss(),
              "sens_criterion": cor_sens_criterion,  # it doesn't use it, but needs the function
              "model_path": "../model_data/synthetic/1/model_weights.pt", "params": [1.]}

    base_trainer = BTrainer(config=config)
    base_trainer.train()
    base_scores = base_trainer.scores

    # evaluate the base performance
    _, S, y_true = load_data(config["data"])
    base_metrics = evaluate(y_true=y_true, S=S, scores=base_scores)
    print(base_metrics)

    # 2. Train and evaluate FairOD-L (L for laziness).
    config = {"model_type": "fairL", "recon_criterion": nn.MSELoss(), "sens_criterion": cor_sens_criterion,
              "data": "../data/synthetic/1/", "epochs": 1, "lr": 0.001,
              "model_path": None, "base_model_weights": "../model_data/synthetic/1/model_weights.pt",
              "niters": 1, "n_splits": 2, "n_hidden": 2,
              "params": [0.01, 0.5, 0.9]}
    fair_correlation_trainer = BTrainer(config=config)  # fairOD-L
    fair_correlation_trainer.train()
    fair_corr_scores = fair_correlation_trainer.scores
    fair_corr_metrics = evaluate(y_true=y_true, S=S, scores=fair_corr_scores)
    print(fair_corr_metrics)

    # 3. Train and evaluate FairOD-C
    # a group fairness based detector
    config = {"model_type": "fairC", "recon_criterion": nn.MSELoss(), "sens_criterion": cor_sens_criterion,
              "group_criterion": group_corr_criterion,
              "data": "../data/synthetic/1/", "epochs": 1, "lr": 0.001,
              "model_path": None,
              "base_model_weights": "../model_data/synthetic/1/model_weights.pt",
              "base_model_scores": base_scores,
              "niters": 1, "n_splits": 2, "n_hidden": 2,
              "params": [(a, g) for a in [0.01, 0.5, 0.9] for g in [0.01, 1, 10, 100, 10000]]}

    # Train requires alpha, gamma parameters
    fair_group_trainer = FTrainer(config=config)
    fair_group_trainer.train()
    fair_group_scores = fair_group_trainer.scores
    fair_group_metrics = evaluate(y_true=y_true, S=S, scores=fair_group_scores)
    print(fair_group_metrics)

    # 4. tain and evaluate FairOD. The proposed method.
    # a group fairness based detector, uses ndcg based criterion
    config = {"model_type": "fairOD", "recon_criterion": nn.MSELoss(), "sens_criterion": cor_sens_criterion,
              "group_criterion": group_ndcg_criterion,
              "data": "../data/synthetic/1/", "epochs": 1, "lr": 0.001,
              "model_path": None,
              "base_model_weights": "../model_data/synthetic/1/model_weights.pt",
              "base_model_scores": base_scores,
              "niters": 1, "n_splits": 2, "n_hidden": 2,
              "params": [(a, g) for a in [0.01, 0.9] for g in [0.01, 1]]}

    # Train requires alpha, gamma parameters
    fairOD_trainer = FTrainer(config=config)
    fairOD_trainer.train()
    fairOD_scores = fairOD_trainer.scores
    fairOD_metrics = evaluate(y_true=y_true, S=S, scores=fairOD_scores)
    print(fairOD_metrics)


if __name__ == '__main__':
    run_demo()
