import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import spacy
from scipy.sparse import save_npz, load_npz
from sklearn.decomposition import PCA
import optuna
import joblib

import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from typing import Union, Optional, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import PreTrainedTokenizerBase
from transformers import DistilBertTokenizer, BigBirdTokenizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string
import textstat
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, roc_auc_score
from textblob import TextBlob
from collections import Counter
from joblib import dump, load
import pickle

nltk.data.path.append('nltk_data')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pytorch_lightning as pl
import torch
import torch.optim as optim
import logging
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchmetrics import AUROC, F1Score, Accuracy
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, BigBirdForSequenceClassification

from sklearn.model_selection import train_test_split, GroupShuffleSplit

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized from the model checkpoint .*")

sns.set_style("whitegrid")