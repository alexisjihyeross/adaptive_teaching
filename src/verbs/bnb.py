import numpy as np
import torch
import torch.distributions as dist
import pdb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.special import gammaln, psi, beta, betaln

from src.verbs.verb_helpers import *


def get_beta_log_pdf(alpha1, alpha2, x):
    return (
        np.log(1)
        - betaln(alpha1, alpha2)
        + (alpha1 - 1) * np.log(x)
        + (alpha2 - 1) * np.log(1 - x)
    )


class BayesianNaiveBayes:
    def __init__(self, alphas, betas, lemmas, categories, seed):
        """
        Initialize the Bayesian Naive Bayes model. Used for both getting the target concept for the teacher and for creating student models.

        :param alphas: torch.tensor, a vector for the Dirichlet prior on the class labels, size (n_classes, )
        :param betas: torch.tensor, a matrix for the Beta prior on the features, size (n_classes, n_features, 2)
        :param lemmas: list[str]: used for fitting the input featurizer
        :param categories: list[str]: used for fitting the output featurizer
        :param seed: int, seed for random generator
        """
        self.feature_extractor = LemmaNgramExtractor()
        self.input_vectorizer = CountVectorizer(
            binary=True, tokenizer=lambda txt: txt.split()
        )
        self.output_vectorizer = preprocessing.LabelEncoder()

        # Data is used for fitting the vectorizer
        self.input_vectorizer.fit(self.feature_extractor.transform(lemmas))
        self.output_vectorizer.fit(categories)

        n_features = len(self.input_vectorizer.get_feature_names())
        n_classes = len(categories)
        if alphas is not None:
            assert n_classes == alphas.shape[0]

        self.n_classes = n_classes
        self.n_features = n_features

        # Random generator used for sampling predictions from probabilities
        self.seed = seed
        self.generator = torch.Generator()
        # TODO: this won't necessarily ensure consistency across teacher/student if predict at diff times
        self.generator.manual_seed(seed)

        # Initialize the counts of feature/class combinations in the data
        self.feature_counts = torch.zeros((n_classes, n_features))
        self.label_counts = torch.zeros(n_classes)

        # Initialize the posterior Dirichlet and Beta distribution parameters for each feature

        if alphas is None:
            self.alpha_prior_parameters = torch.ones(n_classes)
        else:
            self.alpha_prior_parameters = alphas
            assert self.alpha_prior_parameters.shape[0] == (n_classes), (
                f"self.alpha_prior_parameters.shape[0] ="
                f"{self.alpha_prior_parameters.shape[0]}; should = {n_classes}"
            )
        self.alpha_post_parameters = self.alpha_prior_parameters.clone()

        # Initialize the betas to all be 1
        # self.beta_prior_parameters[:, :, 0] corresponds to B_0 params
        # self.beta_prior_parameters[:, :, 1] corresponds to B_1 params
        if betas is None:
            self.beta_prior_parameters = torch.ones((n_classes, n_features, 2))
        else:
            self.beta_prior_parameters = betas
            assert self.beta_prior_parameters.shape == (n_classes, n_features, 2)
        self.beta_post_parameters = self.beta_prior_parameters.clone()

        # Used for reducing calls to get_beta_log_pdf()
        self.beta_pdf_cache = {}
        self.transformed_inputs = {}

    def featurize(self, lemmas):
        return self.feature_extractor.transform(lemmas)

    def transform_inputs(self, lemmas):
        """Returns tensor (n_lemmas x n_features)"""

        key = tuple(lemmas)
        if key in self.transformed_inputs:
            return self.transformed_inputs[key]
        else:
            transformed = self.input_vectorizer.transform(self.featurize(lemmas))
            tensor = torch.as_tensor(transformed.toarray())
            self.transformed_inputs[key] = tensor
            return tensor

    def transform_outputs(self, labels):
        return torch.as_tensor(self.output_vectorizer.transform(labels))

    def get_label_names(self):
        return self.output_vectorizer.classes_

    def get_label_idx(self, label):
        return list(self.get_label_names()).index(label)

    def print_label_counts(self, label):
        """For a given label, prints
        features with nonzero counts"""
        idx = self.get_label_idx(label)
        feature_counts = self.feature_counts[idx, :]
        features = self.get_feature_names()
        for feat, count in zip(features, feature_counts):
            if count != 0:
                print(f"{feat}:\t{count}")

    def print_feature_counts(self, feat):
        """For a given feature, prints
        labels with nonzero counts"""
        idx = self.get_feature_idx(feat)
        label_counts = self.feature_counts[:, idx]
        labels = self.get_label_names()
        for label, count in zip(labels, label_counts):
            if count != 0:
                print(f"{label}:\t{count}")

    def get_feature_idx(self, feature):
        return list(self.get_feature_names()).index(feature)

    def get_feature_names(self):
        return self.input_vectorizer.get_feature_names()

    def increment_counts(self, X, y):
        """
        Increment the counts of feature/class combinations in the data.

        :param X: torch.tensor, the training data of size (n_samples, n_features)
        :param y: torch.tensor, the class labels of size (n_samples, )
        """
        # TODO: is this inefficient (because cloning the big matrices?)
        feature_counts, label_counts = self.compute_counts(X, y)
        self.feature_counts = feature_counts
        self.label_counts = label_counts

    def compute_counts(self, X, y):
        """
        Compute the new counts of feature/class combinations in the data.

        Helper function for increment_counts() and used by the teacher in computing best datapoints

        :param X: torch.tensor, the training data of size (n_samples, n_features)
        :param y: torch.tensor, the class labels of size (n_samples, )

        """

        feature_counts = self.feature_counts.clone()
        label_counts = self.label_counts.clone()

        for i in range(self.n_classes):
            class_mask = y == i
            """
            X[class_mask] will have shape (k x n_features) where k = num examples with class i.
            Want to get num counts of each feature in X[class_mask], so sum counts across dim 0
            Now self.counts[i] is a n_features-dimensional vector with number of counts for each feature occurring with class i
            """
            feature_counts[i, :] += torch.sum(X[class_mask], dim=0)

            num_class = class_mask.sum()
            label_counts[i] += num_class
            assert num_class == X[class_mask].shape[0]
        return feature_counts, label_counts

    def compute_posterior(self, feature_counts=None, label_counts=None):
        """
        Compute the posterior parameters of the Dirichlet and Beta distributions for each feature.

        :param feature_counts: torch.tensor, feature_counts to use when computing the posterior (will usually be the output of compute_counts())
            If None, default to self.feature_counts
        :param label_counts: torch.tensor, label_counts to use when computing the posterior (will usually be the output of compute_counts())
            If None, default to self.feature_counts
        """

        if label_counts is None:
            label_counts = self.label_counts

        if feature_counts is None:
            feature_counts = self.feature_counts

        # for alpha parameters, add all occurrences of classes
        alpha_post_parameters = self.alpha_prior_parameters.clone()
        alpha_post_parameters += label_counts

        beta_post_parameters = self.beta_prior_parameters.clone()
        beta_post_parameters[:, :, 0] += feature_counts
        beta_post_parameters[:, :, 1] += (
            label_counts.unsqueeze(1).expand(-1, feature_counts.shape[1])
            - feature_counts
        )
        return alpha_post_parameters, beta_post_parameters

    def transform_and_simulate_fit(self, lemmas, labels, return_counts=False):
        """
        Analogous to transform_and_fit, except doesn't update any object parameters/states and instead returns the posterior parameters

        - Transforms inputs and outputs
        - Calls compute_counts() to compute new feature/label counts (analogous to increment_counts)
        - Calls compute_posterior() to compute posterior parameters given these counts (analogous to update_posterior)
        No actual updating of the NB object. Used as a helper function for the Bruteforce teacher

        :param lemmas: list, the training data of size n_samples
        :param labels: list, the class labels

        Returns:
            alpha_post_parameters: torch.Tensor, posterior parameters for the Dirichlet distribution
            beta_post_parameters: torch.Tensor, posterior parameters for the Beta distributions
        """
        X = self.transform_inputs(lemmas)
        y = self.transform_outputs(labels)

        feature_counts, label_counts = self.compute_counts(X, y)
        alpha_post_parameters, beta_post_parameters = self.compute_posterior(
            feature_counts, label_counts
        )
        if return_counts:
            return (
                alpha_post_parameters,
                beta_post_parameters,
                label_counts,
                feature_counts,
            )
        return alpha_post_parameters, beta_post_parameters

    def update_posterior(self):
        """
        Update the posterior parameters of the Dirichlet and Beta distributions for each feature.
        """
        alpha_post_parameters, beta_post_parameters = self.compute_posterior()
        self.alpha_post_parameters = alpha_post_parameters
        self.beta_post_parameters = beta_post_parameters

    # TODO: look into: can exceed 1 bc PDF?
    #    @profile
    def get_posterior_log_prob(
        self, pi, phi, alpha_post_parameters=None, beta_post_parameters=None
    ):
        """
        Get posterior log prob of pi/phi estimates
        :param pi: posterior mean for Dirichlet distrib
        :param phi: posterior mean for Beta distribs
        :param alpha_post_parameters: If None, use self.alpha_post_parameters
        :param beta_post_parameters: If None, default to self.beta_post_parameters
        """

        if alpha_post_parameters is None:
            alpha_post_parameters = self.alpha_post_parameters
        if beta_post_parameters is None:
            beta_post_parameters = self.beta_post_parameters

        # pi should have shape n_classes
        assert pi.shape == alpha_post_parameters.shape
        assert pi.shape[0] == self.n_classes

        # phi should have shape (n_classes, n_features)
        assert phi.shape == beta_post_parameters.shape[:-1]
        assert phi.shape == (self.n_classes, self.n_features)

        posterior_dirichlet = dist.dirichlet.Dirichlet(alpha_post_parameters)
        prob = posterior_dirichlet.log_prob(pi)
        for i in range(beta_post_parameters.shape[0]):
            for j in range(beta_post_parameters.shape[1]):
                beta_0 = beta_post_parameters[i, j, 0]
                beta_1 = beta_post_parameters[i, j, 1]
                k = (beta_0, beta_1, phi[i, j])
                if k in self.beta_pdf_cache:
                    prob += self.beta_pdf_cache[k]
                else:
                    val = get_beta_log_pdf(beta_0, beta_1, phi[i, j])
                    prob += val
                    self.beta_pdf_cache[k] = val
        return prob

    def fit(self, X, y):
        """
        Fit the model to the training data.

        :param X: torch.tensor, the training data of size (n_samples, n_features)
        :param y: torch.tensor, the class labels of size (n_samples, )
        """
        self.increment_counts(X, y)
        self.update_posterior()

    def transform_and_fit(self, lemmas, labels):
        """
        Fit the model to the training data.

        :param lemmas: list, the training data of size n_samples
        :param labels: list, the class labels
        """
        X = self.transform_inputs(lemmas)
        y = self.transform_outputs(labels)
        self.fit(X, y)

    def get_posterior_mean(
        self,
        beta_post_parameters=None,
        alpha_post_parameters=None,
        label_counts=None,
        feature_counts=None,
    ):
        """
        Get posterior mean parameters, i.e. estimates for pi/phi
        """

        if beta_post_parameters is None:
            beta_post_parameters = self.beta_post_parameters
        if alpha_post_parameters is None:
            alpha_post_parameters = self.alpha_post_parameters
        if label_counts is None:
            label_counts = self.label_counts
        if feature_counts is None:
            feature_counts = self.feature_counts

        # k-dimensional vector
        dir_post_mean = alpha_post_parameters / (
            self.alpha_prior_parameters.sum() + label_counts.sum()
        )
        beta_0 = self.beta_prior_parameters[:, :, 0]
        beta_1 = self.beta_prior_parameters[:, :, 1]

        # k x p matrix

        beta_post_mean = feature_counts + beta_0
        expanded_label_counts = label_counts.unsqueeze(1).expand(
            (label_counts.shape[0], self.n_features)
        )
        beta_post_mean /= expanded_label_counts + beta_0 + beta_1

        return dir_post_mean, beta_post_mean

    def predict_proba(
        self, lemmas, dir_post_mean=None, beta_post_mean=None, return_log=False
    ):
        """
        Predict the probability of each class label for the given samples.

        :param X: torch.tensor, the test data of size (n_samples, n_features)
        :param return_log: bool
            if True, return *unnormalized* log prob (used by teacher in computing best population)

        :returns: torch.tensor, a matrix of size (n_samples, n_classes)
            containing the probability of each class label per sample
        """
        X = self.transform_inputs(lemmas)
        # alpha_parameters should already be alpha_priors + counts

        # beta_post_mean is theta, should be k x p matrix
        if dir_post_mean is None or beta_post_mean is None:
            # dir_post_mean should be k-dimensional vector
            dir_post_mean, beta_post_mean = self.get_posterior_mean()

        assert dir_post_mean.shape[0] == self.n_classes

        # copy X for each class
        # X is now a b x p x k matrix
        X_expanded = X.unsqueeze(-1).expand((X.shape[0], X.shape[1], self.n_classes))
        # X is now a b x k x p matrix
        X_expanded = X_expanded.swapaxes(1, 2)
        assert X_expanded.shape[0] == X.shape[0]
        assert X_expanded.shape[1] == self.n_classes
        assert X_expanded.shape[2] == self.n_features

        # TODO: better var name
        temp = torch.where(X_expanded == 1, beta_post_mean, 1 - beta_post_mean)
        temp = torch.log(temp)
        temp = temp.sum(dim=-1)
        # temp should have shape b x k
        assert temp.shape == (
            X.shape[0],
            self.n_classes,
        ), f"temp.shape: {temp.shape} != ({X.shape[0]}, {self.n_classes})"

        log_posterior = torch.log(dir_post_mean) + temp

        if return_log:
            return log_posterior

        # trick to avoid numerical issues
        log_posterior = log_posterior - log_posterior.max()

        # Normalize the log-posterior to get the posterior probabilities
        posterior = torch.exp(log_posterior)
        total = posterior.sum(dim=-1)
        posterior = torch.div(posterior, total.reshape(-1, 1))

        return posterior

    def predict(self, lemmas, do_sample=True, proba=None):
        """
        Predict the class label for the given samples.
        :param lemmas: list of n_samples examples

        :param proba = None: predicted probabilities (for efficient computation if already computed)

        :returns: torch.tensor, a vector containing the predicted class labels per sample
        """
        if proba is None:
            proba = self.predict_proba(lemmas)

        if do_sample:
            sampled_indices = torch.multinomial(
                proba, 1, replacement=False, generator=self.generator
            )
            preds = sampled_indices.squeeze(dim=1)
        else:
            preds = torch.argmax(proba, dim=1)
        labels = self.get_label_names()
        label_preds = [labels[idx] for idx in preds]
        return label_preds
