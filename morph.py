import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aeon.distances import dtw_distance
import importlib
tsmorph = importlib.import_module('tsmorph-xai.tsmorph.tsmorph')
TSmorph = tsmorph.TSmorph
from models import Models

class Morph:
    def __init__(self, X : np.array, y : np.array): # apply on test data 
        self.X = X
        self.y = y
        self.class0_X = self.X[self.y == 0]
        self.class1_X = self.X[self.y == 1]
        self.distances = []
        self.borderline_pairs = None

        # Pair-wise results
        self.morphs = {}
        self.preds = {}
        self.results = {}

        # Global metrics
        self.metrics = {}
        return
    
    def get_DTWGlobalBorderline(self, n_samples : int) -> None:   
        # check if y in binarry 
        if np.unique(self.y).shape[0] != 2:
            print("Error: Target must be binary")

        distances = [] # DTW distances
        indices = [] # (class0, class1)  

        # get distance matrix
        for i, sample0 in enumerate(self.class0_X):
            for j, sample1 in enumerate(self.class1_X):
                distances.append(dtw_distance(sample0, sample1))
                indices.append((i,j))

        closer_neighbors = np.argsort(distances)
        if n_samples > len(closer_neighbors):
            n_samples = len(closer_neighbors)

        self.distances = np.array(distances[:n_samples])

        self.borderline_pairs = [(indices[idx], distances[idx]) for idx in closer_neighbors[:n_samples]]
        return


    def Binay_MorphingCalculater(self, model : Models, granularity=100, verbose=False) -> None:
        morphs_perc = []
        acc_count = 0 # count correctly classified pairs

        if verbose:
            print("Pair-Wise Results:")
        for i, pair in enumerate(self.borderline_pairs):
            # desired shape (1, n_features)
            source_c0 = self.class0_X[pair[0][0]].reshape(1,-1)
            target_c1 = self.class1_X[pair[0][1]].reshape(1,-1)

            # apply morphing
            morph = TSmorph(S=source_c0, T=target_c1, granularity=granularity+2).transform()
            morphing = np.array(morph.T, dtype=np.float64)

            # Predict new labels using selected model
            if model.model_name == 'lstm':
                if len(morphing.shape) != 3:  # Ensure it has 3 dimensions (samples, sequence_length, features)
                    morphing = morphing.reshape(morphing.shape[0], 1, morphing.shape[1])
                #print(morphing.shape)
                pred,_ = model.predict(morphing)
            else:
                pred,_ = model.predict(morphing)
            
            if(pred[0]!=0 or pred[-1]!=1):
                continue
            else:
                acc_count+=1
                # calculate morphing percentage
                change_idx = 1
                for i in range(1, len(pred)-1):  # account for both original series
                    if pred[i] != 0:
                        change_idx = i
                        break
                perc = 1/granularity * change_idx
                morphs_perc.append(perc)

                self.morphs[pair[0]] = morph
                self.preds[pair[0]] = pred
                self.results[pair[0]] = round(perc, 2)
                if verbose:
                    print(f"Pair: {pair[0]} -> Morphing percentage: {perc:.2f}")

        # Calculate metrics only if morphs list is not empty
        if morphs_perc:
            self.metrics['mean'] = float(np.mean(morphs_perc))
            self.metrics['std'] = float(np.std(morphs_perc))
        else:
            self.metrics['mean'] = 0.0
            self.metrics['std'] = 0.0

        if verbose:
            print("-------------------------------------------------")
            print(f"Mean morphing percentage: {self.metrics['mean']:.2f}")
            print(f"Standard deviation of morphing percentage: {self.metrics['std']:.2f}")
            print("-------------------------------------------------")
            print(f"Correctly Classified Pairs: {acc_count}/{len(self.borderline_pairs)}")
        
        return self.morphs, self.preds, self.results, self.metrics
    

    def plot_morph(self, pair: tuple) -> None:
        if pair not in self.morphs:
            print("Pair not found")
            return
        
        morph = self.morphs[pair] # pandas DataFrame

        start_color = '#61E6AA'  
        end_color = '#5722B1'    

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot intermediate morphed series
        for idx, column in enumerate(morph.columns):
            linew = 2 if idx==0 or idx==len(morph.columns)-1 else 1
            color = start_color if self.preds[pair][idx] == 0 else end_color
            
            # Set label based on whether it's start, end, or intermediate series
            if idx == 0:
                label = 'Class 0'
            elif idx == len(morph.columns)-1:
                label = 'Class 1'
            else:
                label = None  # Don't show intermediate series in legend

            plt.plot(morph.index, morph[column], color=color, label=label, 
                    linewidth=linew, alpha=0.5)
        
        # Customize the plot
        plt.title('Morphed Time Series', fontsize=14, pad=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
        return