# -*- coding: utf-8 -*-
"""sarcasm_rnn_base_acc_66.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k9LqsTuBWgvE51kIwXKKfMyjV9fyBh20
"""

VARIANT_NUM = 0

"""# RNN Model for Sarcasm Detection

#### Goal of This File:

##### 1. Import Libraries, Helper Functions, and Constants ==> Data Sourcing and Munging ==> Utility Functions ==> Loading the Data ==> Merging all Individual Files

##### 2. Data Processing

##### 3. RNN

##### 4. Conclusion ==> Results ==> Tensorboard

## 1. Import Libraries, Helper Functions and Load Constants
"""

!pip install pandas numpy nltk scikit-learn wordcloud\
seaborn gensim tensorflow imblearn xgboost matplotlib unrar pyunpack more-itertools patool keras-tqdm > /dev/null

VAST = True

if VAST:
    !sudo apt-get install unrar
    !sudo apt-get install rar

GDRIVE = True

UTILS_URL = 'https://f005.backblazeb2.com/file/gp-support-files/archived_nlp_ai_utils.py'
UPDATING_VALUES_URL = 'https://f005.backblazeb2.com/file/gp-support-files/archived_updating_values.py'
ALL_LIBS_URL = 'https://f005.backblazeb2.com/file/gp-support-files/archived_all_libs_dl.py'
CHUNKS_URLS_FILE_URL = 'https://f005.backblazeb2.com/file/gp-support-files/chunks_urls.py'

UTILS_FILE_NAME = 'archived_nlp_ai_utils'
UPDATING_VALUES_FILE_NAME = 'archived_updating_values'
ALL_LIBS_FILE_NAME = 'archived_all_libs_dl'
CHUNKS_URLS_FILE_NAME = 'sd_chunks_urls'

DEP_FILE_EXT = '.py'

import requests
import os

def get_dependencies(url,file_name,file_extension):
    if os.path.exists(file_name + file_extension):
        return print(file_name + " already exists.")
    else:
        print(f"downloading {file_name}...")
        r = requests.get(url)
        with open(file_name + file_extension, 'wb') as fd:
            for chunk in r.iter_content():
                fd.write(chunk)

get_dependencies(UTILS_URL,UTILS_FILE_NAME,DEP_FILE_EXT)
get_dependencies(UPDATING_VALUES_URL,UPDATING_VALUES_FILE_NAME,DEP_FILE_EXT)
get_dependencies(ALL_LIBS_URL,ALL_LIBS_FILE_NAME,DEP_FILE_EXT)
get_dependencies(CHUNKS_URLS_FILE_URL,CHUNKS_URLS_FILE_NAME,DEP_FILE_EXT)

from archived_nlp_ai_utils import *
from sd_chunks_urls import SD_CHUNKS_URLS
from archived_updating_values import *

TF_ENABLE_ONEDNN_OPTS = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_URLS = SD_CHUNKS_URLS
GLOVE_URL = 'https://f005.backblazeb2.com/file/glove-embeddings-dims/glove.6B.100d.txt'
LIMIT = 10
TRAINED_MODELS_COUNT = TRAINED_MODELS

if GDRIVE:
    PICKLES_DIR = 'sd_pickle_files'
    DATA_PATH = 'datasets'
    GLOVE_FILES_DIR = 'glove_files'
    SAVED_MODELS_DIR = 'sd_saved_models'
else:
    PICKLES_DIR = '../sd_pickle_files'
    DATA_PATH = '../datasets'
    GLOVE_FILES_DIR = '../glove_files'
    SAVED_MODELS_DIR = '../sd_saved_models'

PREPROCESSED_CHUNKS_PATH = DATA_PATH + "/preprocessed_sd_data_chunks/"
PROCESSED_DATA_DIR = DATA_PATH + "/processed_data/"
BASE_FILE_NAME = "sd_chunk_"
FILE_FORMAT = ".csv"
DATA_IN_CHUNK = 99744
if LIMIT == 10:
    ACTUAL_DATA_SHAPE = DATA_IN_CHUNK * LIMIT - 2
else:
    ACTUAL_DATA_SHAPE = DATA_IN_CHUNK * LIMIT
RANDOM_STATE = CONST_RANDOM_STATE

"""### 1.1 Data Sourcing and Munging

#### 1.1.1 Loading The Data
"""

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(PREPROCESSED_CHUNKS_PATH):
    os.mkdir(PREPROCESSED_CHUNKS_PATH)
get_chunks(DATA_URLS,LIMIT,1,BASE_FILE_NAME,PREPROCESSED_CHUNKS_PATH,FILE_FORMAT)

"""#### 1.1.2 Merging all Individual Files"""

#get all names of downloaded files
all_file_names = get_all_file_names(BASE_FILE_NAME,LIMIT)

#read all chunks into a list
list_dfs = read_chunks(all_file_names,PREPROCESSED_CHUNKS_PATH,FILE_FORMAT)

#concatenate all chunks into a singular df
df = group_up_chunks(list_dfs)

#check how much of the data was actually downloaded
percent_loaded = check_no_missing_data(df.shape[0],ACTUAL_DATA_SHAPE)
percent_loaded

df.reset_index(inplace = True)
df.drop(['index'],axis = 1,inplace = True)

df.isnull().sum()

df.dropna(inplace = True)

df['text'].replace('', np.nan, inplace=True)
df.dropna(inplace = True)

"""## 2. Data Processing"""

X = df['text']
y = df['labels']

X = pd.Series([str(text) for text in X])

"""### 2.2 GloVe + Hyperparameters"""

if not os.path.exists(GLOVE_FILES_DIR):
    os.mkdir(GLOVE_FILES_DIR)

get_chunks([GLOVE_URL],0,1,"glove.6B.100d",GLOVE_FILES_DIR + '/','.txt',False)

print('Getting Unique Words...')
UNIQUE_WORDS = set(' '.join(X).split())

"""### Hyperparameters"""

VOCAB_SIZE = len(UNIQUE_WORDS)
RNN_UNITS = 256
ATTENTION_UNITS = 64
DENSE_UNITS = 1024
LABELS_COUNT = 1
EMBEDDING_DIM = 100
MAX_TEXT_LEN = 150
CONV_FILTERS = 60
CONV_KERNEL_SIZE = 3
DROPOUT_VAL = 0.2
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"
BATCH_SIZE = 128
EPOCHS = 25

"""##### ==> here is a quick explaination of how the dataset will be split using a smaller sample example.
##### ==> dataset => 100
##### ==> train_set => tr_s (example: 80)
##### ==> valid_set => vs (example: 10)
##### ==> test_set => te_s (example: 10)
##### ==> t = tr_s (80) + vs (10)
##### ==> train_set = x[:80]
##### ==> valid_set = x[80:t]
##### ==> test_set = x[t:] why t? because => vs = ts

##### use this guideline if you are confused about how the train-validation-test split was done. Also, this is a future guide for me as well in case I forget.

* train_set_size = 6,990,280 * 0.8 = 5,592,224
* valid_set_size = 6,990,280 * 0.1 = 699,028
* train_plus_valid = 5,592,224 + 699,028 = 6,291,252

==> To Confirm: test_size = 6,990,280 - 6,291,252 = 699,028

* train_set = [0:5,592,224]
* train_labels = [0:5,592,224]
* validation_set = [5,592,224:6,291,252] ==> 699,028
* validation_labels = [5,592,224:6,291,252] ==> 699,028
* test_set = [6,291,252,6,990,280] ==> 699,028
* test_labels = [6,291,252,6,990,280] ==> 699,028
"""

TRAIN_PERCENT = 0.8
VALID_TEST_PERCENT = 0.1
TRAIN_SIZE = int(ACTUAL_DATA_SHAPE * TRAIN_PERCENT)
VALID_TEST_SIZE = int(ACTUAL_DATA_SHAPE * VALID_TEST_PERCENT)
TOTAL_TEST_SIZE = TRAIN_SIZE + VALID_TEST_SIZE
train_set = X[:TRAIN_SIZE]
train_labels = y[:TRAIN_SIZE]
validation_set = X[TRAIN_SIZE:TOTAL_TEST_SIZE]
validation_labels = y[TRAIN_SIZE:TOTAL_TEST_SIZE]
test_set = X[TOTAL_TEST_SIZE:]
test_labels = y[TOTAL_TEST_SIZE:]

validation_set.reset_index(drop=True,inplace=True)
test_set.reset_index(drop=True,inplace=True)

if not os.path.exists(PROCESSED_DATA_DIR):
    os.mkdir(PROCESSED_DATA_DIR)

# NOTE: THIS CELL TAKES A WHILE TO RUN.
if not os.path.exists(PROCESSED_DATA_DIR + "train_set_padded.pickle"):
    print("Tokenizing the Dataset...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_set)
    words_to_index = tokenizer.word_index

# NOTE: THIS CELL TAKES A WHILE TO RUN.
if not os.path.exists(PROCESSED_DATA_DIR + "train_set_padded.pickle"):
    print("Padding the Training Set.")
    train_sequences = tokenizer.texts_to_sequences(train_set)
    train_set_padded = pad_sequences(train_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
if not os.path.exists(PROCESSED_DATA_DIR + "valid_set_padded.pickle"):
    print("Padding the Validation Set.")
    valid_sequences = tokenizer.texts_to_sequences(validation_set)
    valid_set_padded = pad_sequences(valid_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
if not os.path.exists(PROCESSED_DATA_DIR + "test_set_padded.pickle"):
    print("Padding the Testing Set.")
    test_sequences = tokenizer.texts_to_sequences(test_set)
    test_set_padded = pad_sequences(test_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

if not os.path.exists(PROCESSED_DATA_DIR + "train_set_padded.pickle"):
    print("Re-formatting Train Set Shape.")
    train_set_padded = np.array(train_set_padded)
    
if not os.path.exists(PROCESSED_DATA_DIR + "train_labels.pickle"):
    print("Re-formatting Train Labels Shape.")
    train_labels = np.array(train_labels)
    
if not os.path.exists(PROCESSED_DATA_DIR + "valid_set_padded.pickle"): 
    print("Re-formatting Validation Set Shape.")
    valid_set_padded = np.array(valid_set_padded)

if not os.path.exists(PROCESSED_DATA_DIR + "validation_labels.pickle"):
    print("Re-formatting Validation Labels Shape.")
    validation_labels = np.array(validation_labels)

if not os.path.exists(PROCESSED_DATA_DIR + "test_set_padded.pickle"):
    print("Re-formatting Testing Set Shape.")
    test_set_padded = np.array(test_set_padded)

if not os.path.exists(PROCESSED_DATA_DIR + "test_labels.pickle"):
    print("Re-formatting Testing Labels Shape.")
    test_labels = np.array(test_labels)

word_to_vec_map = read_glove_vector(GLOVE_FILES_DIR + '/glove.6B.100d.txt')

if not os.path.exists(PROCESSED_DATA_DIR + "EMBEDDING_LAYER.pickle") :
    print("Generating Embedding From Scratch.")
    vocab_mapping = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_mapping, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector

    EMBEDDING_LAYER = Embedding(input_dim=vocab_mapping,\
                                output_dim=embed_vector_len, input_length=MAX_TEXT_LEN, weights = [emb_matrix], trainable=True)

if os.path.exists(PROCESSED_DATA_DIR + "train_set_padded.pickle") :
    print("Using Pickle File!")
    train_set_padded = pickle.load(open(PROCESSED_DATA_DIR + "train_set_padded.pickle", "rb"))
else:
    pickle_out = open(PROCESSED_DATA_DIR + "train_set_padded.pickle",'wb')
    pickle.dump(train_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(PROCESSED_DATA_DIR + "train_labels.pickle") :
    print("Using Pickle File!")
    train_labels = pickle.load(open(PROCESSED_DATA_DIR + "train_labels.pickle", "rb"))
else:
    pickle_out = open(PROCESSED_DATA_DIR + "train_labels.pickle",'wb')
    pickle.dump(train_labels,pickle_out)
    pickle_out.close()

if os.path.exists(PROCESSED_DATA_DIR + "valid_set_padded.pickle") :
    print("Using Pickle File!")
    valid_set_padded = pickle.load(open(PROCESSED_DATA_DIR + "valid_set_padded.pickle", "rb"))
else:
    pickle_out = open(PROCESSED_DATA_DIR + "valid_set_padded.pickle",'wb')
    pickle.dump(valid_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(PROCESSED_DATA_DIR + "validation_labels.pickle") :
    print("Using Pickle File!")
    validation_labels = pickle.load(open(PROCESSED_DATA_DIR + "validation_labels.pickle", "rb"))
else:
    pickle_out = open(PROCESSED_DATA_DIR + "validation_labels.pickle",'wb')
    pickle.dump(validation_labels,pickle_out)
    pickle_out.close()

if os.path.exists(PROCESSED_DATA_DIR + "test_set_padded.pickle") :
    print("Using Pickle File!")
    test_set_padded = pickle.load(open(PROCESSED_DATA_DIR + "test_set_padded.pickle", "rb"))
else:
    pickle_out = open(PROCESSED_DATA_DIR + "test_set_padded.pickle",'wb')
    pickle.dump(test_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(PROCESSED_DATA_DIR + "test_labels.pickle") :
    print("Using Pickle File!")
    test_labels = pickle.load(open(PROCESSED_DATA_DIR + "test_labels.pickle", "rb"))
else:
    pickle_out = open(PROCESSED_DATA_DIR + "test_labels.pickle",'wb')
    pickle.dump(test_labels,pickle_out)
    pickle_out.close()

if os.path.exists(PROCESSED_DATA_DIR + "EMBEDDING_LAYER.pickle") :
    print("Using Pickle File!")
    EMBEDDING_LAYER = pickle.load(open(PROCESSED_DATA_DIR + "EMBEDDING_LAYER.pickle", "rb"))
else:
    print("Pickling Embedding Layer!")
    pickle_out = open(PROCESSED_DATA_DIR + "EMBEDDING_LAYER.pickle",'wb')
    pickle.dump(EMBEDDING_LAYER,pickle_out)
    pickle_out.close()

"""## 3. RNN

The requirements to use the cuDNN implementation are:

* activation == tanh
* recurrent_activation == sigmoid
* recurrent_dropout == 0
* unroll is False
* use_bias is True
* Inputs, if use masking, are strictly right-padded.
* Eager execution is enabled in the outermost context.
"""

configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
inputs = tf.keras.Input(shape=(MAX_TEXT_LEN,))
x = EMBEDDING_LAYER(inputs)
x = tf.keras.layers.Normalization()(x)
x = tf.keras.layers.Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same', activation='tanh')(x)
x = tf.keras.layers.GlobalMaxPooling1D(keepdims=True)(x)
x = tf.keras.layers.SimpleRNN(int(RNN_UNITS),activation='tanh')(x)
x = tf.keras.layers.Dense(DENSE_UNITS, activation='tanh')(x)
outputs = tf.keras.layers.Dense(LABELS_COUNT, activation="sigmoid")(x)

rnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

rnn_model.summary()

log_dir = "sd_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

rnn_model.compile(optimizer=Adam(learning_rate = 0.01,amsgrad=True), loss = BinaryCrossentropy(), metrics=['accuracy'])
history = rnn_model.fit(train_set_padded, train_labels,validation_data = (valid_set_padded,validation_labels),\
              batch_size=BATCH_SIZE, epochs=EPOCHS,\
              callbacks=[tensorboard_callback,EarlyStopping(patience=6),ReduceLROnPlateau(factor=0.1, patience=3)])

if not os.path.exists(SAVED_MODELS_DIR):
    os.mkdir(SAVED_MODELS_DIR)

TRAINED_MODELS_COUNT = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if VARIANT_NUM == 0:
    rnn_model.save_weights(SAVED_MODELS_DIR + "/sd_rnn_model_base" + ".h5")
else:
    rnn_model.save_weights(SAVED_MODELS_DIR + "/sd_rnn_model_variant_" + str(VARIANT_NUM) + ".h5")

if VARIANT_NUM == 0:
    pickle_out = open(SAVED_MODELS_DIR + "/sd_rnn_model_params_base" + ".pickle",'wb')
else:
    pickle_out = open(SAVED_MODELS_DIR + "/sd_rnn_model_params_" + str(VARIANT_NUM) + ".pickle",'wb')
pickle.dump({'EMBEDDING_DIM':EMBEDDING_DIM,'MAX_TEXT_LEN':MAX_TEXT_LEN,'BATCH_SIZE':BATCH_SIZE,'EPOCHS':EPOCHS,\
             'train_set_size':len(train_set),'optimizer':str(rnn_model.optimizer),\
             'learning_rate':str(rnn_model.optimizer.learning_rate),\
             'conv_filters':CONV_FILTERS,'conv_kernel':CONV_KERNEL_SIZE,\
             'chunks_used':(len(train_set) + len(validation_set) + len(test_set)) // DATA_IN_CHUNK,\
            'RNN_UNITS':RNN_UNITS,'DENSE_UNITS':DENSE_UNITS,'LABELS_COUNT':LABELS_COUNT},pickle_out)
pickle_out.close()

np.save(SAVED_MODELS_DIR + "/" + 'sd_rnn_model_' + str(VARIANT_NUM) + '.npy',history.history)

"""## 4. Conclusion"""

if VAST:
    !tar -czf sd_rnn_base_logs.tar.gz logs

results = rnn_model.evaluate(test_set_padded, test_labels, batch_size=BATCH_SIZE)
print("test loss, test acc:", results)

"""### 4.1 Results

### 4.2 Tensorboard
"""

# %load_ext tensorboard
# %tensorboard --logdir logs/fit