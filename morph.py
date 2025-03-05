# import numpy as np
# import matplotlib.pyplot as plt
# from aeon.distances import dtw_distance
# import importlib
# tsmorph = importlib.import_module('tsmorph-xai.tsmorph.tsmorph')
# TSmorph = tsmorph.TSmorph
# from models import Models


# class Morph:
#     def __init__(self, X : np.array, y : np.array, target_class = 1): # apply on test data 
#         self.X = X
#         self.y = y
#         self.target_class = target_class
#         self.class1_X = self.X[self.y == target_class]
#         self.class1_y = self.y[self.y == target_class]

#         self.class0_X = self.X[self.y != target_class]
#         self.class0_y = self.y[self.y != target_class]

#         self.distances = []
#         self.borderline_pairs = {}
#         return
    
#     def get_DTWGlobalBorderline(self, perc_samples : float) -> None:   
#         distances = [] # DTW distances
#         indices = [] # (class0, class1)  

#         # get distance matrix
#         for i, sample0 in enumerate(self.class0_X):
#             for j, sample1 in enumerate(self.class1_X):
#                 distances.append(dtw_distance(sample0, sample1))
#                 indices.append((i,j))

#         sorted_neighbors = np.argsort(distances)
#         n_samples = round(len(sorted_neighbors) * perc_samples)

#         #print("Sorted neighbors: ", len(sorted_neighbors))
#         #print("Number of pairs to consider: ", n_samples)

#         if n_samples > len(sorted_neighbors):
#             n_samples = len(sorted_neighbors)

#         self.distances = np.array(distances[:n_samples])

#         for idx in sorted_neighbors[:n_samples]:
#             self.borderline_pairs[indices[idx]] = distances[idx]
            
#         return


#     def Binay_MorphingCalculater(self, model : Models, granularity=100, verbose=False):
#         morphs_perc = []
#         morphs = {}
#         preds = {}
#         results = {}
#         metrics = {}

#         acc_count = 0 # count correctly classified pairs

#         if verbose:
#             print("Pair-Wise Results:")
#         for pair in self.borderline_pairs.keys():
#             # desired shape (1, n_features)

#             source_c0 = self.class0_X[pair[0]].reshape(1,-1)
#             source_c0_y = self.class0_y[pair[0]]

#             target_c1 = self.class1_X[pair[1]].reshape(1,-1)
#             target_c1_y = self.class1_y[pair[1]]

#             # apply morphing
#             morph = TSmorph(S=source_c0, T=target_c1, granularity=granularity+2).transform()
#             morphing = np.array(morph.T, dtype=np.float64)

#             # Predict new labels using selected model
#             if model.model_name == 'lstm':
#                 if len(morphing.shape) != 3:  # Ensure it has 3 dimensions (samples, sequence_length, features)
#                     morphing = morphing.reshape(morphing.shape[0], 1, morphing.shape[1])
#                 #print(morphing.shape)
#                 pred,_ = model.predict(morphing)
#             else:
#                 pred,_ = model.predict(morphing)
            
#             if(pred[0]!=source_c0_y or pred[-1]!=target_c1_y):
#                 continue
#             else:
#                 acc_count+=1
#                 # calculate morphing percentage
#                 change_idx = 1
#                 for i in range(1, len(pred)-1):  # account for both original series
#                     if pred[i] != source_c0_y:
#                         change_idx = i
#                         break
#                 perc = 1/granularity * change_idx
#                 morphs_perc.append(perc)

#                 morphs[pair] = morph
#                 preds[pair] = pred
#                 results[pair] = round(perc, 2)
#                 if verbose:
#                     print(f"Pair: {pair} -> Morphing percentage: {perc:.2f}")

#         # Calculate metrics only if morphs list is not empty
#         if morphs_perc:
#             metrics['mean'] = float(np.mean(morphs_perc))
#             metrics['std'] = float(np.std(morphs_perc))
#         else:
#             metrics['mean'] = 0.0
#             metrics['std'] = 0.0

#         if verbose:
#             print("-------------------------------------------------")
#             print(f"Mean morphing percentage: {metrics['mean']:.2f}")
#             print(f"Standard deviation of morphing percentage: {metrics['std']:.2f}")
#             print("-------------------------------------------------")
#             print(f"Correctly Classified Pairs: {acc_count}/{len(self.borderline_pairs)}")
        
#         return morphs, preds, results, metrics
    

#     def plot_morph(self, pair: tuple, morphs, preds) -> None:
#         if pair not in morphs:
#             print("Pair not found")
#             return
        
#         morph = morphs[pair] # pandas DataFrame

#         start_color = '#61E6AA'  
#         end_color = '#5722B1'    

#         # Create the plot
#         plt.figure(figsize=(12, 6))

#         # Plot intermediate morphed series
#         for idx, column in enumerate(morph.columns):
#             linew = 2 if idx==0 or idx==len(morph.columns)-1 else 1
#             color = start_color if preds[pair][idx] == 0 else end_color
            
#             # Set label based on whether it's start, end, or intermediate series
#             if idx == 0:
#                 label = 'Class 0'
#             elif idx == len(morph.columns)-1:
#                 label = 'Class 1'
#             else:
#                 label = None  # Don't show intermediate series in legend

#             plt.plot(morph.index, morph[column], color=color, label=label, 
#                     linewidth=linew, alpha=0.5)
        
#         # Customize the plot
#         plt.title('Morphed Time Series', fontsize=14, pad=15)
#         plt.xlabel('Time', fontsize=12)
#         plt.ylabel('Value', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend(loc='upper left')

#         plt.tight_layout()
#         plt.show()
#         return
    
    
import importlib
tsmorph = importlib.import_module('tsmorph-xai.tsmorph.tsmorph')
TSmorph = tsmorph.TSmorph
import numpy as np
import matplotlib.pyplot as plt
from aeon.distances import dtw_distance
import importlib
from typing import Dict, Tuple, List, Any
from numba import njit, prange
from models import Models
from tqdm import tqdm

class Morph:
    def __init__(self, X: np.ndarray, y: np.ndarray, target_class: int = 1):
        """
        Initialize Morph class for morphing analysis
        
        Parameters:
        -----------
        X : np.ndarray
            Input feature array
        y : np.ndarray
            Target labels
        target_class : int, optional
            Class to focus morphing analysis on (default is 1)
        """
        self.X = X
        self.y = y
        self.target_class = target_class
        
        # Efficient class separation
        self.class1_mask = self.y == target_class
        self.class0_mask = self.y != target_class

        self.class1_X = self.X[self.class1_mask]
        self.class1_y = self.y[self.class1_mask]
        self.class0_X = self.X[self.class0_mask]
        self.class0_y = self.y[self.class0_mask]
        
        self.distances = []
        self.borderline_pairs = {}
    
    @staticmethod
    @njit(parallel=True)
    def compute_dtw_distances(class0_X: np.ndarray, class1_X: np.ndarray) -> np.ndarray:
        """
        Compute DTW distances between two classes using Numba for acceleration
        
        Parameters:
        -----------
        class0_X : np.ndarray
            Samples from class 0
        class1_X : np.ndarray
            Samples from class 1
        
        Returns:
        --------
        np.ndarray
            Array of DTW distances
        """
        n0, n1 = len(class0_X), len(class1_X)
        distances = np.zeros(n0 * n1, dtype=np.float64)
        
        for i in prange(n0):
            for j in range(n1):
                distances[i * n1 + j] = dtw_distance(class0_X[i], class1_X[j])
        
        return distances
    
    def get_DTWGlobalBorderline(self, perc_samples: float) -> None:
        """
        Compute global borderline pairs based on DTW distances
        
        Parameters:
        -----------
        perc_samples : float
            Percentage of samples to consider for borderline pairs
        """
        distances = self.compute_dtw_distances(self.class0_X, self.class1_X)
        
        # Sort distances and select top pairs
        sorted_indices = np.argsort(distances)
        n_samples = min(round(len(sorted_indices) * perc_samples), len(sorted_indices))
        
        # Store distances and pair indices
        self.distances = distances[sorted_indices[:n_samples]]
        
        # Reconstruct pair indices
        indices = [
            (i, j) 
            for i in range(len(self.class0_X)) 
            for j in range(len(self.class1_X))
        ]
        
        self.borderline_pairs = {
            indices[idx]: distances[idx] 
            for idx in sorted_indices[:n_samples]
        }
    
    def Binary_MorphingCalculater(
        self, 
        models: Tuple[Models], 
        granularity: int = 100, 
        verbose: bool = False
    ) -> Dict:
        results = {}
        
        for pair, _ in tqdm(self.borderline_pairs.items()):
            source_c0 = self.class0_X[pair[0]].reshape(1, -1)
            source_c0_y = self.class0_y[pair[0]]
            target_c1 = self.class1_X[pair[1]].reshape(1, -1)
            target_c1_y = self.class1_y[pair[1]]
            
            # Apply morphing
            morph = TSmorph(S=source_c0, T=target_c1, granularity=granularity+2).transform()
            morphing = np.array(morph.T, dtype=np.float64)
            
            for model in models:
                # Initialize model-specific results if not exists
                if model.model_name not in results:
                    results[model.model_name] = {
                        'morphs': {},
                        'model_preds': {},
                        'pair_results': {},
                        'morphs_perc': []
                    }
                
                # Predict new labels using selected model
                if model.model_name == 'lstm':
                    if len(morphing.shape) != 3:  # Ensure it has 3 dimensions (samples, sequence_length, features)
                        morphing = morphing.reshape(morphing.shape[0], 1, morphing.shape[1])
                    pred,_ = model.predict(morphing)
                else:
                    pred,_ = model.predict(morphing)


                # Ensure valid morphing pairs
                if pred[0] == source_c0_y and pred[-1] == target_c1_y:
                    
                    # Find where label changes
                    change_idx = 1
                    for i in range(1, len(pred)-1):  # account for both original series
                        if pred[i] != source_c0_y:
                            change_idx = i
                            break
            
                    # Calculate morphing percentage 
                    perc = 1/granularity * change_idx
                    
                    results[model.model_name]['morphs'][pair] = morph
                    results[model.model_name]['model_preds'][pair] = pred
                    results[model.model_name]['pair_results'][pair] = round(perc, 2)
                    results[model.model_name]['morphs_perc'].append(perc)
                    
                    if verbose:
                        print(f"Model {model} Pair: {pair} -> Morphing percentage: {perc:.2f}")

        # Compute metrics for each model
        for model_name, model_results in results.items():
            morphs_perc = model_results['morphs_perc']
            model_results['metrics'] = {
                'mean': float(np.mean(morphs_perc)) if morphs_perc else 0.0,
                'std': float(np.std(morphs_perc)) if morphs_perc else 0.0
            }
        
        return results