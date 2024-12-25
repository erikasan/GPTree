import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import resample
from sklearn.base import clone

from numpy.random import binomial

from typing import Callable, Optional, Type

from copy import deepcopy

from tqdm import tqdm

import joblib

import sys
import time

from scipy.optimize import differential_evolution, minimize
from scipy.linalg import cho_solve, cholesky

def _differential_evolution(obj_func: Callable, initial_theta: np.ndarray, bounds: tuple):
    pass
    #return theta_opt, func_min

class Default_GPR(GaussianProcessRegressor):
        """ The default GaussianProcessRegressor of GPTree """
        def __init__(self, kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
            self.kernel = Matern()
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts_optimizer = n_restarts_optimizer
            self.normalize_y = normalize_y
            self.copy_X_train = copy_X_train
            self.n_targets = n_targets
            self.random_state = random_state



class GPNode(Node):
    def __init__(self,
                 *args,
                 my_GPR: GaussianProcessRegressor,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.my_GPR = my_GPR
        self.parent = None
        self.children = None
        
        self.is_left = None
        self.is_leaf = False
        
        self.num_training_points = 0

        self.my_X_data = None
        self.my_y_data = None
        self.n_features = None

        self.j = 0
        self.s = 0.0
        self.o = 0.001

    # Override the "value" attribute of Node parent class 
    @property
    def num_training_points(self):
        return self.value
    
    # such that the value of a node is the number of training points
    @num_training_points.setter
    def num_training_points(self, value):
        self.value = value

    def init_training_set(self, n_features: int, n_targets: int):
        """ Initialize the training set of the node. """
        self.my_X_data = np.array([]).reshape((0, n_features))
        self.my_y_data = np.array([]).reshape((0, n_targets))
        self.n_features = n_features

    def generate_children(self, GPR: Type[GaussianProcessRegressor], n_features: int, n_targets: int):
        """ Grow the GPtree by adding two GPNodes as children of the current GPNode. """
        self.left = GPNode(0, my_GPR=deepcopy(GPR))
        self.right = GPNode(0, my_GPR=deepcopy(GPR))
        self.left.is_left = True
        self.right.is_left = False

        self.children = [self.left, self.right]

        for child in self.children:
            child.parent = self
            child.init_training_set(n_features, n_targets)



    def add_training_data(self, x: np.ndarray, y: float):
        """ Add a single training sample to the training set of the node. """
        self.my_X_data = np.append(self.my_X_data, x, axis=0)
        self.my_y_data = np.append(self.my_y_data, y, axis=0)
        self.num_training_points += 1
        
        #self.value = self.num_training_points

    def split_training_data(self):
        """ Assign the training samples of a node to its child nodes. """
        for x, y in zip(self.my_X_data, self.my_y_data):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape(1, -1)
            child = self.children[int(binomial(1, self.prob_func(x)))]
            child.add_training_data(x, y)
        
    def delete_training_data(self):
        del self.my_X_data, self.my_y_data

    def compute_my_GPR(self):
        """ Fit the GP of the node with sklearn. """
        self.my_GPR.fit(self.my_X_data, self.my_y_data)



    def compute_s_and_o(self, theta: float): ## TODO: This needs a better name lmao
        """ Find the position s of the dividing hyperplane and the overlapping region o. """
        w = np.zeros(self.n_features)
        for i in range(self.n_features):
            w[i] = np.max(self.my_X_data[:, i]) - np.min(self.my_X_data[:, i])
        self.j = np.argmax(w)

        self.s = np.mean(self.my_X_data[:, self.j])
        self.o = theta*w[self.j]

    def prob_func(self, x: np.array):
        """ The default probability function as suggested in the DLGP article. """
        prob = (x[:, self.j] - self.s)/self.o + 0.5
        prob[prob < 0] = 0
        prob[prob > 1] = 1

        prob.shape = (x.shape[0], 1)

        return prob
    
    """ def prob_func(self, x: np.array):
        return np.array(x.shape[0]*[0.5]).reshape((x.shape[0], 1)) """

    def marg_prob(self, x: np.ndarray):
        """ Compute the marginal probability that a test point x belongs to this node. """
        ptilde = np.ones(shape=(x.shape[0], 1))
        node = self
        while node.parent:
            is_left = node.is_left
            node = node.parent
            if is_left:
                ptilde *= (1 - node.prob_func(x))
            else:
                ptilde *= node.prob_func(x)
            
        return ptilde
        


        

class GPTree:
    """ Class for regression with Dividing Local Gaussian Processes.
    
        Attributes
        ----------
        Nbar: Optional[int] = 100
            Maximum number of training points that each node can have.

        theta: Optional[float] = 0.0001
            Parameter in probability function when assigning training samples to nodes.

        Methods
        -------
    """
    def __init__(self,
                 GPR: Optional[GaussianProcessRegressor] = Default_GPR(),
                 Nbar: Optional[int] = 100,
                 theta: Optional[float] = 0.0001):
        
        self.GPR = GPR
        self.root = GPNode(0, my_GPR=GPR)  # Initialize root node of the GPTree
        
        self.Nbar = Nbar
        self.theta = theta

        self.n_training_points = 0
        self.n_features = 0
        self.n_targets = 0


    def updateTree(self, x: np.ndarray, y: float):
        """ Essentially Algorithm 1 in DLGP article. """
        node = self.root  # Start from root node
        while node.children:  # Then follow a random branch until a leaf node is reached
            node = node.children[int(binomial(1, node.prob_func(x)))]
        

        if node.num_training_points == self.Nbar:   # If the node is full
            node.generate_children(self.GPR, self.n_features, self.n_targets) # generate child nodes
            
            
            node.compute_s_and_o(self.theta)        # Calculate parameters in probability function

            node.split_training_data()              # Assign training samples to children
            """ for child in node.children:         # and update their GPs
                child.compute_my_GPR()
                pass """

            node.delete_training_data()             # Training data of non-leaf nodes is not needed

            node = node.children[int(binomial(1, node.prob_func(x)))]
        
        node.add_training_data(x, y)
        #node.compute_my_GPR()

    def build_binary_tree(self, X_train: np.ndarray, y_train: np.ndarray, 
            show_progress: Optional[bool]=False, shuffle: Optional[bool]=True):
        
        self.n_features = X_train.shape[1]
        self.n_targets = y_train.shape[1]
        self.n_training_points = X_train.shape[0]
        self.root.init_training_set(self.n_features, self.n_targets)

        if shuffle:
            X_train, y_train = resample(X_train, y_train, replace=False)

        for x, y in tqdm(zip(X_train, y_train), total=self.n_training_points, disable=not show_progress, desc="Building binary tree"):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, self.n_targets))
            self.updateTree(x, y)

        for leaf in self.root.leaves:
            leaf.is_leaf = True
            leaf.my_GPR.kernel_ = clone(leaf.my_GPR.kernel)
            leaf.my_GPR.X_train_ = leaf.my_X_data
            leaf.my_GPR.y_train_ = leaf.my_y_data

        return

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            show_progress: Optional[bool]=False, shuffle: Optional[bool]=True, inherit_GPR=False,
            share_hyperparams: Optional[bool]=False):
        """
        Build the binary tree by assigning training samples to nodes and train the GP of the leaf nodes.

        Arguments
        ----------
        X_train: np.ndarray
            The training data in feature space. Has shape=(N_train, n_features).

        y_train: np.ndarray
            The training data in target space. Has shape=(N_train, n_targets).

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        shuffle: Optional[bool]=True
            Shuffle the training set to avoid an unbalanced tree.
        """

        self.build_binary_tree(X_train=X_train, y_train=y_train, show_progress=show_progress, shuffle=shuffle)      

        if share_hyperparams: # Training with shared hyperparameters
            self.fit_shared_hyperparams()
        
        else: # Train each leaf node independently on their local training set.
            for i, leaf in tqdm(enumerate(self.root.leaves), total=len(self.root.leaves), disable=not show_progress, desc="Training"):
                leaf.compute_my_GPR()
                if inherit_GPR and i != len(self.root.leaves) - 1:
                    self.root.leaves[i+1].my_GPR = deepcopy(leaf.my_GPR)

            
            """ kernel = leaf.my_GPR.kernel_
            with open("hyperparameters.txt", 'a') as infile:
                infile.write(f"Leaf node {i}")
                infile.write("##############")
                for hyperparameter, hyperparameter_value in zip(kernel.hyperparameters, kernel.theta):
                    infile.write(f"{hyperparameter} {np.exp(hyperparameter_value)} \n") """
        return  
    
    
    def predict(self, X_test: np.ndarray, recursive_search: Optional[bool]=True, show_progress: Optional[bool]=False):
        """ 
        Predict the target function at selected points in feature space. 
        
        Arguments
        ---------

        X_test: np.ndarray
            The points in feature space where we'd like to predict the target function. Has shape=(N_test, n_features).

        recursive_search: Optional[bool]=True
            Prediction with recursive search of leafs with nonzero marginal probabilities.
            The method recursive_predict() is called instead.

            If false the predictions of leafs with zero marginal probabilities are also computed. 

            recursive_search should be set to True when N >> Nbar. Otherwise, this approach is actually slower. 

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        Returns
        -------

        mean_DLGP: np.ndarray
            The posterior mean used to predict the target function. Has shape=(N_test, 1).
        
        std_DLGP: np.ndarray
            The posterior standard deviation used to quantify the uncertainty in the prediction. Has shape=(N_test, 1).
        """
        if recursive_search: # Recursive search
            mean_DLGP, std_DLGP = self.recursive_predict(X_test, show_progress)
        
        else:   # Loop over all leaf nodes, even those with zero weights. 
            mean_DLGP = np.zeros((X_test.shape[0], self.n_targets))
            var_DLGP = np.zeros((X_test.shape[0], self.n_targets))

            for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Predicting"):
                
                mu_leaf, sigma_leaf = leaf.my_GPR.predict(X_test, return_std=True)

                mu_leaf = mu_leaf.reshape(mean_DLGP.shape)
                sigma_leaf = sigma_leaf.reshape(mean_DLGP.shape)

                ptilde = leaf.marg_prob(X_test)
                #ptilde = ptilde.reshape(mean_DLGP.shape)
                ptilde = ptilde.reshape(X_test.shape[0], 1)

                mean_DLGP += ptilde*mu_leaf

                var_DLGP += ptilde*(sigma_leaf*sigma_leaf + mu_leaf*mu_leaf)
            
            var_DLGP += -mean_DLGP*mean_DLGP
            std_DLGP = np.sqrt(var_DLGP)

        return mean_DLGP, std_DLGP
    

    def collect_leaves(self, x: np.ndarray):
        """ Collect the leaves that are active at x. """

        def _collect_leaves(x: np.ndarray, current_node: GPNode, current_prob: float):
            """ Recursive function to collect contributing leaves for prediction at a test point x.  """

            #global sum_probs, collection_done

            """ if collection_done or current_prob <= 0:
                return """
            
            if current_prob <= 0:
                return
            
            # Return if we have reached a leaf node
            if current_node.is_leaf:
                leaves.append(current_node)
                pred_leaf_probs.append(current_prob)

                """ sum_probs += current_prob
                if sum_probs >= 1:
                    collection_done = True   """             
                return

            # Ok, not a leaf node. Now, for both child nodes:
            # - compute the probability
            # - call this function again

            new_p = current_node.prob_func(x)

            p0 = current_prob*(1 - new_p)
            if p0 > 0:
                _collect_leaves(x, current_node.left, p0)

            p1 = current_prob*new_p
            if p1 > 0:
                _collect_leaves(x, current_node.right, p1)

            # Done
            return
        
        leaves = []
        pred_leaf_probs = []

        _collect_leaves(x, self.root, 1)

        return leaves, pred_leaf_probs


    def recursive_predict(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        """ 
        Second predict function with recursive search of leaves with nonzero marginal probabilities.  
        
        Arguments
        ---------

        X_test: np.ndarray
            The points in feature space where we'd like to predict the target function. Has shape=(N_test, n_features).

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        Returns
        -------

        mean_DLGP: np.ndarray
            The posterior mean used to predict the target function. Has shape=(N_test, 1).
        
        std_DLGP: np.ndarray
            The posterior standard deviation used to quantify the uncertainty in the prediction. Has shape=(N_test, 1).
        """        
        
        mean_DLGP = np.zeros((X_test.shape[0], self.n_targets))
        var_DLGP = np.zeros((X_test.shape[0], self.n_targets))
        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))

            leaves, pred_leaf_probs = self.collect_leaves(x)
        
            for leaf, ptilde in zip(leaves, pred_leaf_probs):
                
                mu_leaf, sigma_leaf = leaf.my_GPR.predict(x, return_std=True)

                # This is the stupid part
                mean_DLGP[i] += ptilde[0]*mu_leaf[0]
                var_DLGP[i] += ptilde[0]*(sigma_leaf[0]*sigma_leaf[0] + mu_leaf[0]*mu_leaf[0])
            
            var_DLGP[i] += -mean_DLGP[i]*mean_DLGP[i]
        
        return mean_DLGP, np.sqrt(var_DLGP)
    

    def sample_predict(self, X_test: np.ndarray, num_samples: Optional[int]=100, show_progress: Optional[bool]=False):

        """ If X_test has shape (num_test_points, num_features),
            
            Returns an array "sample_array" with shape (num_test_points, num_targets, num_samples)

            The samples are drawn from the mixture of experts predictive distribution at each test point.
        """

        num_test_points = X_test.shape[0]

        sample_array = np.zeros((num_test_points, self.n_targets, num_samples))

        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))

            # Lists with the predictive mean and standard deviation of each expert at the point x
            means = [] 
            stds = []

            leaves, pred_leaf_probs = self.collect_leaves(x)

            leaf_indices = [j for j in range(len(leaves))]

            # Because of mistakes I made when I was young and stupid
            pred_leaf_probs = [pred_leaf_probs[p][0][0] for p in range(len(pred_leaf_probs))]

            for leaf in leaves:
                mu_leaf, sigma_leaf = leaf.my_GPR.predict(x, return_std=True)

                if self.n_targets == 1:
                    mu_leaf = mu_leaf.reshape(-1, 1)
                    sigma_leaf = sigma_leaf.reshape(-1, 1)
                
                means.append(mu_leaf)
                stds.append(sigma_leaf)

            """ for target_index in range(self.n_targets):
                for k in range(num_samples):

                    # Pick a leaf
                    leaf_index = np.random.choice(leaf_indices, size=1, p=pred_leaf_probs)
                    leaf_index = leaf_index[0]

                    sample = np.random.normal(loc=means[leaf_index][target_index], scale=stds[leaf_index][target_index])

                    print("sample.shape=", sample.shape)
                    print("sample=", sample)

                    sample_array[i, target_index, k] = sample """
            
            for k in range(num_samples):
                # Pick a leaf
                leaf_index = np.random.choice(leaf_indices, size=1, p=pred_leaf_probs)
                leaf_index = leaf_index[0]

                sample = np.random.normal(loc=means[leaf_index], scale=stds[leaf_index], size=(1, self.n_targets))

                sample_array[i, :, k] = sample

        return sample_array

    def predict_piecewise(self, x_test):
        """ Get the prediction of each leaf node respectively.  """
        res = []
        for leaf in self.root.leaves:
            mu_leaf, sigma_leaf = leaf.my_GPR.predict(x_test, return_std=True)
            res.append((mu_leaf, sigma_leaf))
        return res
    
    def save(self, path: str):
        joblib.dump(self, path)

    

    def sum_negative_log_marginal_likelihoods(self, hyperparams, eval_gradient=False):
        """ Return the sum of the negative log marginal likelihoods of the leaf nodes 
            each evaluated with the same hyperparameters.

            If eval_gradient is True it also returns the gradient with respect to 
            the natural logs of the hyperparameters.

            NOTE: Only works with scikit-learn's GaussianProcessRegressor.    
         """

        if eval_gradient:
            sum_log_marginal_likelihoods = 0
            sum_log_marginal_likelihoods_gradient = 0
            for leaf in self.root.leaves:
                lml, lml_gradient = leaf.my_GPR.log_marginal_likelihood(theta=hyperparams, eval_gradient=True)
                sum_log_marginal_likelihoods += lml
                sum_log_marginal_likelihoods_gradient += lml_gradient
            
            return -sum_log_marginal_likelihoods, -sum_log_marginal_likelihoods_gradient
        
        else:
            sum_log_marginal_likelihoods = 0
            for leaf in self.root.leaves:
                lml = leaf.my_GPR.log_marginal_likelihood(theta=hyperparams, eval_gradient=False)
                sum_log_marginal_likelihoods += lml

            return -sum_log_marginal_likelihoods
    

    def _obj_func(self, hyperparams, eval_gradient=False):
        """ The objective function to be minimized. Sets the (log transformed) hyperparameters of the leaf nodes
            to hyperparams at each call.

            Returns the sum negative log marginal likelihood.
        """

        for leaf in self.root.leaves:
            #leaf.my_GPR.kernel_.theta = np.log(hyperparams)
            leaf.my_GPR.kernel_.theta = hyperparams

        return self.sum_negative_log_marginal_likelihoods(hyperparams=hyperparams, eval_gradient=eval_gradient)
    

    def fit_shared_hyperparams(self):
        """ Train the leaf nodes with shared hyperparmeters using L-BFGS.
        """

        # Initialize kernel_, X_train_, y_train_ attributes, which otherwise doesn't exist unless gp.fit() is called.
        for leaf in self.root.leaves:
            leaf.my_GPR.kernel_ = clone(leaf.my_GPR.kernel)
            #leaf.my_GPR.kernel_ = leaf.my_GPR.kernel
            leaf.my_GPR.X_train_ = leaf.my_X_data
            leaf.my_GPR.y_train_ = leaf.my_y_data

            shape_y_stats = (leaf.my_y_data.shape[1],) if leaf.my_y_data.ndim == 2 else 1
            leaf.my_GPR._y_train_mean = np.zeros(shape=shape_y_stats)
            leaf.my_GPR._y_train_std = np.ones(shape=shape_y_stats)

        
        initial_hyperparams = leaf.my_GPR.kernel.theta
        bounds = leaf.my_GPR.kernel.bounds

        # Would be cool to have an animation in the future
        print("Training leaf nodes with shared hyperparameters...")

        result = minimize(
            fun=lambda x: self._obj_func(x, eval_gradient=True),
            x0=initial_hyperparams,
            bounds=bounds,
            method='L-BFGS-B',
            jac=True,
        )

        if not result.success:
            print("Warning: Optimization failed")

        
        # Precompute quantities required for predictions which are independent of actual query points
        for leaf in self.root.leaves:
            X_train_ = leaf.my_GPR.X_train_
            K = leaf.my_GPR.kernel_(X_train_)
            K[np.diag_indices_from(K)] += leaf.my_GPR.alpha

            try:
                leaf.my_GPR.L_ = cholesky(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError as exc:
                exc.args = (
                    (
                        f"The kernel, {self.kernel_}, is not returning a positive "
                        "definite matrix. Try gradually increasing the 'alpha' "
                        "parameter of your GaussianProcessRegressor estimator."
                    ),
                ) + exc.args
                raise

            leaf.my_GPR.alpha_ = cho_solve(
                (leaf.my_GPR.L_, True),
                leaf.my_GPR.y_train_,
                check_finite=False,
                )


        return result.x, self._obj_func(result.x, eval_gradient=False)
