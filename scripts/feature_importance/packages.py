import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer, make_regression, make_classification
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import silhouette_samples, r2_score, mean_squared_error, f1_score, roc_auc_score, mean_absolute_error, accuracy_score
from collections import defaultdict
from functools import reduce
from lime import lime_tabular, submodular_pick
from sklearn.datasets import make_regression, make_classification
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, date
import shap
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tarfile
from matplotlib.ticker import MaxNLocator
import pickle
import random
import uuid
import matplotlib.pyplot as plt