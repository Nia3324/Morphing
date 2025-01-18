from aeon.distances import dtw_distance
import numpy as np
from models import Models
tsmorph = importlib.import_module('tsmorph-xai.tsmorph.tsmorph')
TSmorph = tsmorph.TSmorph
import numpy as np
import tensorflow as tf
import importlib

class Morph:
    def __init__(self, X, y): # applyed test data 
        self.X = X
        self.y = y
        self.class0_X = self.X[self.y == 0]
        self.class1_X = self.X[self.y == 1]
        self.borderline_neig = None
        self.morphs = None
        return
    
    def get_DTWGlobalBorderline(self, n_samples) -> None:   
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

        distances = np.array(distances)
        closer_neighbors = np.argsort(distances)[:n_samples]

        self.borderline_neig = [(indices[idx], distances[idx]) for idx in closer_neighbors]
        return


    def Binay_MorphingCalculater(self, model, granularity=100):
        morphs = []
        results = {}
        metrics = {}

        for sample in self.borderline_neig:
            # desired shape (1, n_features)
            source_ts = self.X[sample[0]].reshape(1,-1)
            target_ts = self.X[sample[1]].reshape(1,-1)
            #print(source_ts.shape)

            # apply morphing
            self.morphs = TSmorph(S=source_ts, T=target_ts, granularity=granularity+2).transform()
            self.morphs = self.morphs.T

            # Predict new labels using selected model
            if model.model_name == 'lstm':
                morphing = morphing.to_numpy()
                if len(morphing.shape) != 3:  # Ensure it has 3 dimensions (samples, sequence_length, features)
                    morphing = morphing.reshape(morphing.shape[0], 1, morphing.shape[1])

                print(morphing.shape)
                pred = model.predict(morphing)
   

            elif catch22_features:
                features = compute_catch22_features(morphing)
                pred = model.predict(features)

            elif rocket_kernels is not None:  # Fixed variable name from kernels to rocket_kernels
                morphing = np.array(morphing, dtype=np.float64)
                morphing_transform = apply_kernels(morphing, rocket_kernels)
                pred = model.predict(morphing_transform)

            # check if model predictions align with ground truth
            if (y_morph[sample[0]] == pred[0] and y_morph[sample[1]] == pred[-1]):
                # calculate morphing percentage
                change_idx = 1
                for i in range(1, len(pred)-1):  # account for both original series
                    if pred[0] != pred[i]:
                        change_idx = i
                        break
                morph_perc = 1/granularity * change_idx
                morphs.append(morph_perc)

                results[(sample[0], sample[1])] = round(morph_perc, 2)
            else:
                continue

        # Calculate metrics only if morphs list is not empty
        if morphs:
            metrics['mean'] = float(np.mean(morphs))
            metrics['std'] = float(np.std(morphs))
        else:
            metrics['mean'] = 0.0
            metrics['std'] = 0.0

        return results, metrics