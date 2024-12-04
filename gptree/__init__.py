import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import resample

from numpy.random import binomial

from typing import Callable, Optional, Type, Union

from copy import deepcopy

from tqdm import tqdm

import joblib

from scipy.optimize import differential_evolution

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
        
        self.is_root = False
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

    def init_training_set(self, n_features: int):
        """ Initialize the training set of the node. """
        self.my_X_data = np.array([]).reshape((0, n_features))
        self.my_y_data = np.array([]).reshape((0, 1))
        self.n_features = n_features

    def generate_children(self, GPR: Type[GaussianProcessRegressor], n_features: int):
        """ Grow the GPtree by adding two GPNodes as children of the current GPNode. """
        self.left = GPNode(0, my_GPR=deepcopy(GPR))
        self.right = GPNode(0, my_GPR=deepcopy(GPR))
        self.left.is_left = True
        self.right.is_left = False

        self.children = [self.left, self.right]

        for child in self.children:
            child.parent = self
            child.init_training_set(n_features)



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
            y = y.reshape((1, 1))
            child = self.children[int(binomial(1, self.prob_func(x)))]
            child.add_training_data(x, y)
        
    def delete_training_data(self):
        del self.my_X_data, self.my_y_data

    def compute_my_GPR(self):
        """ Fit the GP of the node with sklearn. """
        self.my_GPR.fit(self.my_X_data, self.my_y_data)

    def compute_s_and_o(self, theta: float): ## TODO: This needs a better name lmao
        """ Find the position s of the dividing hyperplane and the overlapping region o. """
        w = np.empty(self.n_features)
        for i in range(self.n_features):
            w[i] = np.max(self.my_X_data[:, i]) - np.min(self.my_X_data[:, i])
        self.j = np.argmax(w)

        self.s = np.mean(self.my_X_data[:, self.j])
        self.o = theta*w[self.j]

    def prob_func(self, x: np.array):
        """ The default probability function as suggested in the DLGP article.
            The probability that a point (x, y) is sent to the right child node. """
        

        # TEMPORARY
        #self.s = 99.0

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
    """ Class for GPTree regression (only scalar target functions implemented).
    
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
        self.root.is_root = True
        
        self.Nbar = Nbar
        self.theta = theta

        self.n_features = 0


    def updateTree(self, x: np.ndarray, y: float):
        """ Essentially Algorithm 1 in DLGP article. """
        node = self.root  # Start from root node
        while node.children:  # Then follow a random branch until a leaf node is reached
            node = node.children[int(binomial(1, node.prob_func(x)))]

        if node.num_training_points == self.Nbar:   # If the node is full
            node.generate_children(self.GPR, self.n_features) # generate child nodes
            
            
            node.compute_s_and_o(self.theta)        # Calculate parameters in probability function

            node.split_training_data()              # Assign training samples to children
            """ for child in node.children:         # and update their GPs
                child.compute_my_GPR()
                pass """

            node.delete_training_data()             # Training data of non-leaf nodes is not needed

            node = node.children[int(binomial(1, node.prob_func(x)))]
        
        node.add_training_data(x, y)
        #node.compute_my_GPR()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, show_progress: Optional[bool]=False, shuffle: Optional[bool]=True, inherit_GPR=False):
        """
        Build the binary tree by assigning training samples to nodes and train the GP of the leaf nodes.

        Arguments
        ----------
        X_train: np.ndarray
            The training data in feature space. Has shape=(N_train, n_features).

        y_train: np.ndarray
            The training data in target space. Has shape=(N_train, 1) (only scalar targets implemented).

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        shuffle: Optional[bool]=True
            Shuffle the training set to avoid an unbalanced tree.
        """
        self.n_features = X_train.shape[1]
        N = X_train.shape[0]
        self.root.init_training_set(self.n_features)

        if shuffle:
            X_train, y_train = resample(X_train, y_train, replace=False)

        for x, y in tqdm(zip(X_train, y_train), total=N, disable=not show_progress, desc="Building binary tree"):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, 1))
            self.updateTree(x, y)
        
        for i, leaf in tqdm(enumerate(self.root.leaves), total=len(self.root.leaves), disable=not show_progress, desc="Training"):
            leaf.is_leaf = True
            leaf.compute_my_GPR()
            if inherit_GPR and i != len(self.root.leaves) - 1:
                self.root.leaves[i+1].my_GPR = deepcopy(leaf.my_GPR)

            
            """ kernel = leaf.my_GPR.kernel_
            with open("hyperparameters.txt", 'a') as infile:
                infile.write(f"Leaf node {i}")
                infile.write("##############")
                for hyperparameter, hyperparameter_value in zip(kernel.hyperparameters, kernel.theta):
                    infile.write(f"{hyperparameter} {np.exp(hyperparameter_value)} \n") """
                
                
    
    
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
            mean_DLGP = np.zeros((X_test.shape[0], 1))
            var_DLGP = np.zeros((X_test.shape[0], 1))

            for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Predicting"):
                
                mu_leaf, sigma_leaf = leaf.my_GPR.predict(X_test, return_std=True)
                mu_leaf = mu_leaf.reshape(mean_DLGP.shape)
                sigma_leaf = sigma_leaf.reshape(mean_DLGP.shape)

                ptilde = leaf.marg_prob(X_test)
                ptilde = ptilde.reshape(mean_DLGP.shape)

                mean_DLGP += ptilde*mu_leaf

                var_DLGP += ptilde*(sigma_leaf*sigma_leaf + mu_leaf*mu_leaf)
            
            var_DLGP += -mean_DLGP*mean_DLGP
            std_DLGP = np.sqrt(var_DLGP)

        return mean_DLGP, std_DLGP
    

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
        

        # Unneccesary to keep track of whether the marginal probabilities of the currently collected leaf nodes add up to one.
        #global sum_probs, collection_done

        
        def collect_leaves(x: np.ndarray, current_node: GPNode, current_prob: float):
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
                collect_leaves(x, current_node.left, p0)

            p1 = current_prob*new_p
            if p1 > 0:
                collect_leaves(x, current_node.right, p1)

            # Done
            return
        
        mean_DLGP = np.zeros((X_test.shape[0], 1))
        var_DLGP = np.zeros((X_test.shape[0], 1))
        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))

            """ sum_probs = 0
            collection_done = False """

            leaves = []
            pred_leaf_probs = []

            collect_leaves(x, self.root, 1)
        
            for leaf, ptilde in zip(leaves, pred_leaf_probs):
                
                mu_leaf, sigma_leaf = leaf.my_GPR.predict(x, return_std=True)

                # This is the stupid part
                mean_DLGP[i] += ptilde[0]*mu_leaf[0]
                var_DLGP[i] += ptilde[0]*(sigma_leaf[0]*sigma_leaf[0] + mu_leaf[0]*mu_leaf[0])
            
            var_DLGP[i] += -mean_DLGP[i]*mean_DLGP[i]
        
        return mean_DLGP, np.sqrt(var_DLGP)
    

    def recursive_predict_algortihm_2(self, X_test):
        """ Third predict function if you can believe it.
         
            Literal implementation of Algorithm 2 in Gaussian Process-Based Real-Time Learning for Safety Critical Applications by Lederer et al.
            Seems to be somewhat slower than recursive_predict when N >> N_bar, so recursive_predict is preferred.
        """

        def ____predict(self, x: np.ndarray, n: GPNode):
            if n.is_leaf:
                mu_n, sigma_n = n.my_GPR.predict(x, return_std=True)
                return [mu_n], [sigma_n*sigma_n], [1]
            
            else:
                mu_list = []; var_list = []; omega_list = []

                p = n.prob_func(x)
                if  p > 0:
                    mu_hat, var_hat, omega_hat = ____predict(self, x, n.right)
                    mu_list = mu_list + mu_hat; var_list = var_list + var_hat;
                    
                    omega_list = omega_list + [omega*p for omega in omega_hat]
                if 1 - p > 0:
                    mu_hat, var_hat, omega_hat = ____predict(self, x, n.left)
                    mu_list = mu_list + mu_hat; var_list = var_list + var_hat;
                    
                    omega_list = omega_list + [omega*(1-p) for omega in omega_hat]

                if not n.is_root:
                    return mu_list, var_list, omega_list
                
                else:
                    mu_DLGP = 0
                    var_DLGP = 0

                    for mu, var, omega in zip(mu_list, var_list, omega_list):
                        mu_DLGP += mu[0]*omega[0]

                        var_DLGP += (mu[0]*mu[0] + var[0])*omega[0]
                    
                    var_DLGP -= mu_DLGP[0]*mu_DLGP[0]

                    return mu_DLGP, np.sqrt(var_DLGP)

        mean_DLGP = np.zeros((X_test.shape[0], 1))
        std_DLGP = np.zeros((X_test.shape[0], 1))

        for i, x in enumerate(X_test):
            x = x.reshape((1, x.shape[0]))
            mean_DLGP[i], std_DLGP[i] = ____predict(self, x, self.root)
        
        return mean_DLGP, std_DLGP
    


    def recursive_predict_algortihm_3(self, X_test, show_progress=True):
        """ Third predict function if you can believe it.
         
            Implementation of Algorithm 2 in Gaussian Process-Based Real-Time Learning for Safety Critical Applications by Lederer et al.
            Seems to be somewhat slower than recursive_predict when N >> N_bar, so recursive_predict is preferred.
        """

        def _collect_leaves(x: np.ndarray, n: GPNode):
            if n.is_leaf:
                return [n], [1]
            
            else:
                leaves = []; omega_list = []

                p = n.prob_func(x)
                if  p > 0:
                    ell_hat,  omega_hat = _collect_leaves(x, n.right)
                    leaves = leaves + ell_hat; omega_list = omega_list + [omega*p for omega in omega_hat]
                    
                if 1 - p > 0:
                    ell_hat,  omega_hat = _collect_leaves(x, n.left)
                    leaves = leaves + ell_hat; omega_list = omega_list + [omega*(1-p) for omega in omega_hat]

            return leaves, omega_list
        

        mean_DLGP = np.zeros((X_test.shape[0], 1))
        var_DLGP = np.zeros((X_test.shape[0], 1))
        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))

            leaves, pred_leaf_probs = _collect_leaves(x, self.root)
        
            for leaf, ptilde in zip(leaves, pred_leaf_probs):
                
                mu_leaf, sigma_leaf = leaf.my_GPR.predict(x, return_std=True)

                # This is the stupid part
                mean_DLGP[i] += ptilde[0]*mu_leaf[0]
                var_DLGP[i] += ptilde[0]*(sigma_leaf[0]*sigma_leaf[0] + mu_leaf[0]*mu_leaf[0])
            
            var_DLGP[i] += -mean_DLGP[i]*mean_DLGP[i]
        
        return mean_DLGP, np.sqrt(var_DLGP)

    def predict_piecewise(self, x_test):
        """ Get the prediction of each leaf node respectively.  """
        res = []
        for leaf in self.root.leaves:
            mu_leaf, sigma_leaf = leaf.my_GPR.predict(x_test, return_std=True)
            res.append((mu_leaf, sigma_leaf))
        return res
    
    def save(self, path: str):
        joblib.dump(self, path)



class GPForest:
    def __init__(self,
                 GPR: Optional[Union[GaussianProcessRegressor, list]] = Default_GPR(),
                 Nbar: Optional[Union[int, list]] = 100,
                 theta: Optional[Union[float, list]] = 0.0001):
        
        self.GPR = GPR
        self.Nbar = Nbar
        self.theta = theta

        self.GPTrees = []

        
        if type(self.Nbar) == int:
            self.Nbar = [self.Nbar]

        if type(self.theta) == int:
            self.theta = [self.theta]
        
        assert len(self.Nbar) == len(self.theta), "Nbar and theta must have equally many elements"

        self.num_GPTrees = len(self.Nbar)

        if type(self.GPR) == GaussianProcessRegressor or type(self.GPR) == Default_GPR:
            self.GPR = self.num_GPTrees*[self.GPR]


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, show_progress: Optional[bool]=False):

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Building forest'):
            self.GPTrees.append(GPTree(self.GPR[i], self.Nbar[i], self.theta[i]))
        
        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Training GPTrees'):
            self.GPTrees[i].fit(X_train, y_train, shuffle=True)

    def predict(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        mean = np.zeros((X_test.shape[0], 1))
        std = np.zeros((X_test.shape[0], 1))

        mean_list = []
        std_list = []

        alpha = []
        T = []

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Predicting'):
            mean_i, sigma_i = self.GPTrees[i].predict(X_test)

            mean_list.append(mean_i)
            std_list.append(sigma_i)

            T.append(1./(sigma_i*sigma_i))

            sigma_prior = np.diag(self.GPTrees[i].GPR.kernel(X_test)).reshape(-1, 1)

            alpha.append(0.5*(np.log(sigma_prior) - np.log(sigma_i)))

        sum_alpha = np.sum(alpha)
        
        for i in range(self.num_GPTrees):

            mean += mean_list[i]*alpha[i]*T[i]/sum_alpha

            std += alpha[i]*T[i]/sum_alpha
            
            
        std = 1./std
        
        mean *= std

        return mean, std

    def save(self, path: str):
        joblib.dump(self, path)


        
        

