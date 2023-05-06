# -*- coding: utf-8 -*-
"""
rnn.py

# RNN Model

"""
#!pip install pandas numpy nltk scikit-learn wordcloud seaborn gensim tensorflow imblearn xgboost matplotlib patool > /dev/null

from nlp_ai_utils_vast import *
READY_DATA_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/ready_for_models.rar'
CLASS_WEIGHTS_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/class_weights.pickle'
TF_ENABLE_ONEDNN_OPTS = 0
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 15

if os.path.exists("data_files"):
    os.mkdir("data_files")

get_chunks(READY_DATA_URL,0,1,'ready_for_models','data_files/','.rar',False)

patoolib.extract_archive('data_files/ready_for_models.rar', outdir = 'data_files')

get_chunks(CLASS_WEIGHTS_URL,0,1,'class_weights','data_files/','.pickle',False)

class_weights = pickle.load(open("data_files/class_weights.pickle", "rb"))

train_set_padded = pickle.load(open("data_files/train_set_padded.pickle", "rb"))
  
train_labels = pickle.load(open("data_files/train_labels.pickle", "rb"))

valid_set_padded = pickle.load(open("data_files/valid_set_padded.pickle", "rb"))

validation_labels = pickle.load(open("data_files/validation_labels.pickle", "rb"))

test_set_padded = pickle.load(open("data_files/test_set_padded.pickle", "rb"))

test_labels = pickle.load(open("data_files/test_labels.pickle", "rb"))

EMBEDDING_LAYER = pickle.load(open("data_files/EMBEDDING_LAYER.pickle", "rb"))


tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
rnn_model = tf.keras.models.Sequential([
    EMBEDDING_LAYER,
    tf.keras.layers.SimpleRNN(60),
    tf.keras.layers.Dense(512, activation = "relu"),
    tf.keras.layers.Dense(5, activation = "softmax")
])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

rnn_model.compile(optimizer=Adam(), loss = SparseCategoricalCrossentropy(), metrics=['accuracy',custom_f1_score])
rnn_model.fit(train_set_padded, train_labels,validation_data = (valid_set_padded,validation_labels),\
              batch_size=BATCH_SIZE, epochs=EPOCHS,class_weight=dict(enumerate(class_weights)),\
              callbacks=[tensorboard_callback,EarlyStopping(patience=3),ReduceLROnPlateau(factor=0.1, patience=1)])

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

rnn_model.save_weights('saved_models/rnn_model.h5')