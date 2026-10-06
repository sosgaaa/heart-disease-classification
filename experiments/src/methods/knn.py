import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=7, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind


    def _compute_distances(self, test_data):
        """
        Compute pairwise distances between test data and training data.
        
        Arguments:
            test_data (np.array): test data of shape (N_test, D)
        
        Returns:
            distances (np.array): distances of shape (N_test, N_train)
        """
        n_test = test_data.shape[0]
        n_train = self.training_data.shape[0]
        distances = np.zeros((n_test, n_train))
        
            # Default to Manhattan distance
        for i in range(n_test):
            distances[i] = np.sqrt(np.sum((self.training_data - test_data[i])**2, axis=1))
        
        return distances

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
                # copies
        self.training_data = np.copy(training_data)
        self.training_labels = np.copy(training_labels)
        
        # start with random values
        pred_labels = np.ones_like(training_labels)
        
        for i in range(training_data.shape[0]):
            
            # compute distances between point i and all other points
            distances = np.sqrt(np.sum((self.training_data - training_data[i])**2, axis=1))
      
            # get indices of k nearest neighbors
            k_nearest_ind = np.argsort(distances, kind='quicksort')[:self.k]
            k_nearest_labels = self.training_labels[k_nearest_ind]
            
            # map of label -> occurences
            occ = {k: 0 for k in np.unique(k_nearest_labels)}
            for j in k_nearest_labels:
                occ[j]+=1

            # sorting by occurences the map
            occ = dict(sorted(occ.items(), key=lambda item: item[1]))

            # take the biggest one
            label = list(occ.items())[len(occ.values()) - 1][0]
            pred_labels[i] = label
            
                
        return pred_labels
    

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        # pairwise distances between test data and training data
        distances = self._compute_distances(test_data)
        
        n_samples = test_data.shape[0]
        test_labels = np.zeros(n_samples)
        
        for i in range(n_samples):
            # get indices of k nearest neighbors
            k_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = self.training_labels[k_indices]
            
            # weight by inverse distance (add small epsilon to avoid division by zero)
            k_distances = distances[i, k_indices]
            weights = 1.0 / (k_distances + 1e-8)
            
            # get weighted votes with their labels
            label_weights = {}
            for label, w in zip(k_nearest_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + w
            
            # find the label with the highest weight
            test_labels[i] = max(label_weights.items(), key=lambda x: x[1])[0]
                
        return test_labels







 