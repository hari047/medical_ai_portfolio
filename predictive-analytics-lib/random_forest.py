import numpy as np
from collections import namedtuple, Counter

# Lightweight, immutable node structure (Kept from your original)
KDT_Node = namedtuple('KDT_Node', ['location', 'split_dimension', 'left_child', 'right_child'])

class RandomForest:
    """
    Custom Random Forest implementation using K-D Trees.
    
    Features:
    - Bootstrap Aggregating (Bagging)
    - K-D Tree Base Learners for fast spatial partitioning
    - Ensemble Voting Mechanism
    """
    def __init__(self, n_trees=10, subsample_ratio=0.8):
        self.n_trees = n_trees
        self.subsample_ratio = subsample_ratio
        self.forest = []

    def _build_tree(self, dataset, depth, total_features):
        """
        Recursively constructs a K-d Tree.
        (Refactored to remove Student ID logic and print statements)
        """
        if dataset.shape[0] == 0:
            return None

        # Determine split axis (Round Robin strategy)
        split_axis = depth % total_features

        # Sort and find median
        sorted_indices = dataset[:, split_axis].argsort()
        sorted_dataset = dataset[sorted_indices]
        
        median_index = sorted_dataset.shape[0] // 2
        pivot_point = sorted_dataset[median_index]
        median_val = pivot_point[split_axis]

        # Partition remaining points
        # Logic: Values <= median go left, > median go right
        remaining = np.concatenate((sorted_dataset[:median_index], sorted_dataset[median_index + 1:]), axis=0)
        
        left_mask = remaining[:, split_axis] <= median_val
        left_points = remaining[left_mask]
        right_points = remaining[~left_mask]

        return KDT_Node(
            location=pivot_point,
            split_dimension=split_axis,
            left_child=self._build_tree(left_points, depth + 1, total_features),
            right_child=self._build_tree(right_points, depth + 1, total_features)
        )

    def _find_nearest(self, node, target, current_best):
        """
        Recursively searches for the nearest neighbor.
        """
        if node is None:
            return current_best

        best_point, best_dist = current_best
        
        # Calculate Euclidean distance
        # Assuming last column is label, so we exclude it from distance calc
        dist = np.linalg.norm(node.location[:-1] - target)

        if dist < best_dist:
            best_dist = dist
            best_point = node.location

        split_dim = node.split_dimension
        
        # Determine search order
        if target[split_dim] < node.location[split_dim]:
            primary, secondary = node.left_child, node.right_child
        else:
            primary, secondary = node.right_child, node.left_child

        # Recurse down primary branch
        best_point, best_dist = self._find_nearest(primary, target, (best_point, best_dist))

        # Check secondary branch if needed (Backtracking)
        dist_to_plane = abs(target[split_dim] - node.location[split_dim])
        if dist_to_plane < best_dist:
            best_point, best_dist = self._find_nearest(secondary, target, (best_point, best_dist))

        return best_point, best_dist

    def fit(self, X, y):
        """
        Trains the forest on provided data.
        Args:
            X: Feature matrix
            y: Labels
        """
        # Combine X and y to keep labels attached to points
        training_data = np.column_stack((X, y))
        n_samples, n_total_cols = training_data.shape
        num_features = n_total_cols - 1 # Last col is label
        
        subsample_size = int(n_samples * self.subsample_ratio)
        self.forest = []

        for i in range(self.n_trees):
            # 1. Bootstrap Sampling (Random choice with replacement)
            indices = np.random.choice(n_samples, size=subsample_size, replace=True)
            sampled_data = training_data[indices]
            
            # 2. Build Tree
            # Start depth at 'i' to vary the starting split dimension for diversity
            root = self._build_tree(sampled_data, depth=i, total_features=num_features)
            self.forest.append(root)

    def predict(self, X):
        """
        Predicts class labels for input X.
        """
        predictions = []
        for query_vector in X:
            tree_votes = []
            
            for root in self.forest:
                if root is None: continue
                
                # Find nearest neighbor in this tree
                best_point, _ = self._find_nearest(root, query_vector, (None, float('inf')))
                
                if best_point is not None:
                    # Label is the last element
                    tree_votes.append(int(best_point[-1]))
            
            # Majority Vote
            if tree_votes:
                # Use Counter to find the most common label
                vote_counts = Counter(tree_votes)
                most_common = vote_counts.most_common(1)[0][0]
                predictions.append(most_common)
            else:
                predictions.append(0) # Default fallback
                
        return np.array(predictions)