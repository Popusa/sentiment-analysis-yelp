import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import collections
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import requests
import os
import pickle
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec