# -*- coding: utf-8 -*-
"""rnn_3labels_variant_six_acc_76_chunks_18.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V_m3_IexVRiq9mrqypWVO1EDZdoGL5zD
"""

VARIANT_NUM = 4

"""# RNN Model

#### Goal of This File:

##### 1. Import Libraries, Helper Functions, and Constants ==> Data Sourcing and Munging ==> Utility Functions ==> Loading the Data ==> Merging all Individual Files

##### 2. Imbalanced Data Solution ==> Class Weights ==> Glove + Hyperparameters

##### 3. Simple RNN

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

UTILS_URL = 'https://f005.backblazeb2.com/file/gp-support-files/nlp_ai_utils.py'
UPDATING_VALUES_URL = 'https://f005.backblazeb2.com/file/gp-support-files/updating_values.py'
ALL_LIBS_URL = 'https://f005.backblazeb2.com/file/gp-support-files/all_libs_dl.py'
CHUNKS_URLS_FILE_URL = 'https://f005.backblazeb2.com/file/gp-support-files/chunks_urls.py'

UTILS_FILE_NAME = 'nlp_ai_utils'
UPDATING_VALUES_FILE_NAME = 'updating_values'
ALL_LIBS_FILE_NAME = 'all_libs_dl'
CHUNKS_URLS_FILE_NAME = 'chunks_urls'

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

from nlp_ai_utils import *
from chunks_urls import CHUNKS_URLS
from updating_values import *

TF_ENABLE_ONEDNN_OPTS = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LIMIT = 60
DATA_URLS = CHUNKS_URLS[:LIMIT]

GLOVE_URL = 'https://f005.backblazeb2.com/file/glove-embeddings-dims/glove.6B.100d.txt'
READY_DATA_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/ready_for_models.rar'
CLASS_WEIGHTS_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/class_weights.pickle'
UNIQUE_WORDS_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/unique_words.pickle'
LIMIT = len(DATA_URLS)
TRAINED_MODELS_COUNT = TRAINED_MODELS

if GDRIVE:
    PICKLES_DIR = "pickle_files"
    READY_DATASET_PATH = "larger_dataset/ready_for_models/"
    LARGER_DATASET_PATH = "larger_dataset"
    GLOVE_FILES_DIR = "glove_files"
    SAVED_MODELS_DIR = 'saved_models'
else:
    PICKLES_DIR = "../pickle_files"    
    LARGER_DATASET_PATH = "../larger_dataset"
    READY_DATASET_PATH = "../larger_dataset/ready_for_models/"
    GLOVE_FILES_DIR = "../glove_files"
    SAVED_MODELS_DIR = '../saved_models'
    
PREPROCESSED_CHUNKS_PATH = LARGER_DATASET_PATH + "/preprocessed_data_chunks/"
BASE_FILE_NAME = "chunk_"
FILE_FORMAT = ".csv"
DATA_IN_CHUNK = 116505
if LIMIT == 60:
    ACTUAL_DATA_SHAPE = DATA_IN_CHUNK * LIMIT + 20
else:
    ACTUAL_DATA_SHAPE = DATA_IN_CHUNK * LIMIT
RANDOM_STATE = CONST_RANDOM_STATE
DELETE_PICKLES_AFTER_TRAINING = True
USE_READY_DATA = False
BALANCE_DATA = True
USE_MODIFIED_LABELS = True

"""### 1.1 Data Sourcing and Munging

#### 1.1.1 Loading The Data
"""

if not USE_READY_DATA:
    if not os.path.exists(LARGER_DATASET_PATH):
        os.mkdir(LARGER_DATASET_PATH)
    if not os.path.exists(PREPROCESSED_CHUNKS_PATH):
        os.mkdir(PREPROCESSED_CHUNKS_PATH)
    get_chunks(DATA_URLS,LIMIT,1,BASE_FILE_NAME,PREPROCESSED_CHUNKS_PATH,FILE_FORMAT)

"""#### 1.1.2 Merging all Individual Files"""

#get all names of downloaded files
if not USE_READY_DATA:
    all_file_names = get_all_file_names(BASE_FILE_NAME,LIMIT)

#read all chunks into a list
if not USE_READY_DATA:
    list_dfs = read_chunks(all_file_names,PREPROCESSED_CHUNKS_PATH,FILE_FORMAT)

#concatenate all chunks into a singular df
if not USE_READY_DATA:
    df = group_up_chunks(list_dfs)

#check how much of the data was actually downloaded
if not USE_READY_DATA:
    percent_loaded = check_no_missing_data(df.shape[0],ACTUAL_DATA_SHAPE)
    percent_loaded

if not USE_READY_DATA:
    review_data = df[['text', 'stars']].copy()

if not USE_READY_DATA:
    review_data.reset_index(inplace = True)
    review_data.drop(['index'],axis = 1,inplace = True)

if not USE_READY_DATA:
    review_data.rename(columns = {'text':'full_review_text','stars':'star_rating'}, inplace = True)

if not USE_READY_DATA:
    review_data.isnull().sum()

if not USE_READY_DATA:
    review_data.dropna(inplace = True)

if not USE_READY_DATA:
    review_data['full_review_text'].replace('', np.nan, inplace=True)
    review_data.dropna(inplace = True)

"""## 2. Imbalanced Data Solution"""

if BALANCE_DATA:
    label_1 = review_data[review_data['star_rating'] == 1]
    label_2 = review_data[review_data['star_rating'] == 2]
    label_3 = review_data[review_data['star_rating'] == 3]
    label_4 = review_data[review_data['star_rating'] == 4]
    label_5 = review_data[review_data['star_rating'] == 5]

    minority_class = min([label_1.shape[0],label_2.shape[0],label_3.shape[0],label_4.shape[0],label_5.shape[0]])

if BALANCE_DATA:
    if USE_MODIFIED_LABELS:
        review_data_label_1 = label_1[:int(minority_class / 2)]
        review_data_label_2 = label_2[:int(minority_class / 2)]
        review_data_label_3 = label_3[:int(minority_class)]
        review_data_label_4 = label_4[:int(minority_class / 2)]
        review_data_label_5 = label_5[:int(minority_class / 2)]

    else:
        review_data_label_1 = label_1[:int(minority_class)]
        review_data_label_2 = label_2[:int(minority_class)]
        review_data_label_3 = label_3[:int(minority_class)]
        review_data_label_4 = label_4[:int(minority_class)]
        review_data_label_5 = label_5[:int(minority_class)]

    review_data = pd.concat([review_data_label_1,review_data_label_2,review_data_label_3,review_data_label_4,review_data_label_5])

    print(review_data.shape)

    print(review_data['star_rating'].value_counts())

if not USE_READY_DATA:
    X = review_data['full_review_text']
    y = review_data['star_rating']

if not USE_READY_DATA:
    X = pd.Series([str(text) for text in X])

if USE_MODIFIED_LABELS:
    translated_labels = translate_labels(y)
    y = pd.Series(encode_sent(translated_labels))

print(y.value_counts())

if not USE_READY_DATA:
    y = y - 1

"""#### Classes' shapes before and after (if USE_MODIFIED_LABELS = True):

BEFORE: 1 ==> Very Bad ==> AFTER: 2 ==> Negative Sentiment

BEFORE: 2 ==> Bad ==> AFTER: 2  ==> Negative Sentiment

BEFORE: 3 ==> Ok ==> AFTER: 3  ==> Neutral Sentiment

BEFORE: 4 ==> Good ==> AFTER: 1  ==> Positive Sentiment

BEFORE: 5 ==> Very Good ==> AFTER: 1  ==> Positive Sentiment

THEN: 1 is subtracted from each label. So, the Labels go FROM: 2,3,1 TO: 1,2,0

#### Classes' shapes before and after (if USE_MODIFIED_LABELS = False):

1 is subtracted from each label. So, the Labels go FROM: 1,2,3,4,5 TO: 0,1,2,3,4

Why is 1 subracted from each label? To bring the data into the preferred shape of the class weights in both sklearn and keras (starting label is 0, not 1).

### 2.1 Class Weights
"""

if not os.path.exists(PICKLES_DIR):
    os.mkdir(PICKLES_DIR)

if USE_READY_DATA:
    get_chunks([CLASS_WEIGHTS_URL],0,1,'class_weights',PICKLES_DIR + '/','.pickle',False)

if USE_READY_DATA:
    class_weights = pickle.load(open(PICKLES_DIR + "/class_weights.pickle", "rb"))

if not USE_READY_DATA:
    if BALANCE_DATA:
        class_weights = False
    else:
        class_weights = compute_class_weight(class_weight = "balanced",classes = np.unique(y),y=y)
        class_weights = dict(zip(np.unique(y), class_weights))

"""### 2.2 GloVe + Hyperparameters"""

if not os.path.exists(GLOVE_FILES_DIR):
    os.mkdir(GLOVE_FILES_DIR)

if not USE_READY_DATA:
    get_chunks([GLOVE_URL],0,1,"glove.6B.100d",GLOVE_FILES_DIR + '/','.txt',False)

if not USE_READY_DATA:
    print('Getting Unique Words...')
    UNIQUE_WORDS = set(' '.join(X).split())

if USE_READY_DATA:
    get_chunks([UNIQUE_WORDS_URL],0,1,'unique_words',PICKLES_DIR + '/','.pickle',False)

if USE_READY_DATA:
    UNIQUE_WORDS = pickle.load(open(PICKLES_DIR + "/unique_words.pickle", "rb"))

"""### Hyperparameters"""

VOCAB_SIZE = len(UNIQUE_WORDS)
RNN_UNITS = 256
CONV_FILTERS = 96
CONV_KERNEL_SIZE = 3
DROPOUT_VAL = 0.5
DENSE_UNITS = 1024
LABELS_COUNT = len(y.unique())
EMBEDDING_DIM = 100
MAX_TEXT_LEN = 400
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"
BATCH_SIZE = 128
EPOCHS = 60

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

ACTUAL_DATA_SHAPE = len(X)

if BALANCE_DATA:
    temp = pd.DataFrame({'text':X,'labels':y})
    shuffled = temp.sample(frac=1,random_state=RANDOM_STATE)
    X = shuffled['text']
    y = shuffled['labels']

X.reset_index(drop=True,inplace=True)
y.reset_index(drop=True,inplace=True)

if not USE_READY_DATA:
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

train_set.reset_index(drop=True,inplace=True)
train_labels.reset_index(drop=True,inplace=True)
validation_set.reset_index(drop=True,inplace=True)
validation_labels.reset_index(drop=True,inplace=True)
test_set.reset_index(drop=True,inplace=True)
test_labels.reset_index(drop=True,inplace=True)

if not os.path.exists(LARGER_DATASET_PATH):
    os.mkdir(LARGER_DATASET_PATH)

if not os.path.exists(READY_DATASET_PATH):
    os.mkdir(READY_DATASET_PATH)

if USE_READY_DATA:
    get_chunks([READY_DATA_URL],0,1,'ready_for_models',READY_DATASET_PATH,'.rar',False)
    Archive(os.path.join(READY_DATASET_PATH,"ready_for_models.rar")).extractall(READY_DATASET_PATH)

if USE_READY_DATA:
    train_set_padded = pickle.load(open(READY_DATASET_PATH + "train_set_padded.pickle", "rb"))
    train_labels = pickle.load(open(READY_DATASET_PATH + "train_labels.pickle", "rb"))
    valid_set_padded = pickle.load(open(READY_DATASET_PATH + "valid_set_padded.pickle", "rb"))
    validation_labels = pickle.load(open(READY_DATASET_PATH + "validation_labels.pickle", "rb"))
    test_set_padded = pickle.load(open(READY_DATASET_PATH + "test_set_padded.pickle", "rb"))
    test_labels = pickle.load(open(READY_DATASET_PATH + "test_labels.pickle", "rb"))
    EMBEDDING_LAYER = pickle.load(open(READY_DATASET_PATH + "EMBEDDING_LAYER.pickle", "rb"))

# NOTE: THIS CELL TAKES A WHILE TO RUN.
if not os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle") and not USE_READY_DATA:
    print("Tokenizing the Dataset...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_set)
    words_to_index = tokenizer.word_index

# NOTE: THIS CELL TAKES A WHILE TO RUN.
if not os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle") and not USE_READY_DATA:
    print("Padding the Training Set.")
    train_sequences = tokenizer.texts_to_sequences(train_set)
    train_set_padded = pad_sequences(train_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
if not os.path.exists(READY_DATASET_PATH + "valid_set_padded.pickle") and not USE_READY_DATA:
    print("Padding the Validation Set.")
    valid_sequences = tokenizer.texts_to_sequences(validation_set)
    valid_set_padded = pad_sequences(valid_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
if not os.path.exists(READY_DATASET_PATH + "test_set_padded.pickle") and not USE_READY_DATA:
    print("Padding the Testing Set.")
    test_sequences = tokenizer.texts_to_sequences(test_set)
    test_set_padded = pad_sequences(test_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

if not os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle") and not USE_READY_DATA:
    print("Re-formatting Train Set Shape.")
    train_set_padded = np.array(train_set_padded)
    
if not os.path.exists(READY_DATASET_PATH + "train_labels.pickle") and not USE_READY_DATA:
    print("Re-formatting Train Labels Shape.")
    train_labels = np.array(train_labels)
    
if not os.path.exists(READY_DATASET_PATH + "valid_set_padded.pickle") and not USE_READY_DATA: 
    print("Re-formatting Validation Set Shape.")
    valid_set_padded = np.array(valid_set_padded)

if not os.path.exists(READY_DATASET_PATH + "validation_labels.pickle") and not USE_READY_DATA:
    print("Re-formatting Validation Labels Shape.")
    validation_labels = np.array(validation_labels)

if not os.path.exists(READY_DATASET_PATH + "test_set_padded.pickle") and not USE_READY_DATA:
    print("Re-formatting Testing Set Shape.")
    test_set_padded = np.array(test_set_padded)

if not os.path.exists(READY_DATASET_PATH + "test_labels.pickle") and not USE_READY_DATA:
    print("Re-formatting Testing Labels Shape.")
    test_labels = np.array(test_labels)

if not USE_READY_DATA:
    word_to_vec_map = read_glove_vector(GLOVE_FILES_DIR + '/glove.6B.100d.txt')

if not os.path.exists(READY_DATASET_PATH + "EMBEDDING_LAYER.pickle") and not USE_READY_DATA:
    print("Generating Embedding From Scratch.")
    vocab_mapping = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_mapping, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index - 1, :] = embedding_vector

    EMBEDDING_LAYER = Embedding(input_dim=vocab_mapping,\
                                output_dim=embed_vector_len, input_length=MAX_TEXT_LEN, weights = [emb_matrix], trainable=True)

if os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    train_set_padded = pickle.load(open(READY_DATASET_PATH + "train_set_padded.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "train_set_padded.pickle",'wb')
    pickle.dump(train_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(READY_DATASET_PATH + "train_labels.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    train_labels = pickle.load(open(READY_DATASET_PATH + "train_labels.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "train_labels.pickle",'wb')
    pickle.dump(train_labels,pickle_out)
    pickle_out.close()

if os.path.exists(READY_DATASET_PATH + "valid_set_padded.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    valid_set_padded = pickle.load(open(READY_DATASET_PATH + "valid_set_padded.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "valid_set_padded.pickle",'wb')
    pickle.dump(valid_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(READY_DATASET_PATH + "validation_labels.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    validation_labels = pickle.load(open(READY_DATASET_PATH + "validation_labels.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "validation_labels.pickle",'wb')
    pickle.dump(validation_labels,pickle_out)
    pickle_out.close()

if os.path.exists(READY_DATASET_PATH + "test_set_padded.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    test_set_padded = pickle.load(open(READY_DATASET_PATH + "test_set_padded.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "test_set_padded.pickle",'wb')
    pickle.dump(test_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(READY_DATASET_PATH + "test_labels.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    test_labels = pickle.load(open(READY_DATASET_PATH + "test_labels.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "test_labels.pickle",'wb')
    pickle.dump(test_labels,pickle_out)
    pickle_out.close()

if os.path.exists(READY_DATASET_PATH + "EMBEDDING_LAYER.pickle") and not USE_READY_DATA:
    print("Using Pickle File!")
    EMBEDDING_LAYER = pickle.load(open(READY_DATASET_PATH + "EMBEDDING_LAYER.pickle", "rb"))
else:
    print("Pickling Embedding Layer!")
    pickle_out = open(READY_DATASET_PATH + "EMBEDDING_LAYER.pickle",'wb')
    pickle.dump(EMBEDDING_LAYER,pickle_out)
    pickle_out.close()

"""## 3. RNN"""

configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
rnn_model = tf.keras.models.Sequential([
    EMBEDDING_LAYER,
    tf.keras.layers.Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same', activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(keepdims = True),
    tf.keras.layers.SimpleRNN(RNN_UNITS),
    tf.keras.layers.Dropout(DROPOUT_VAL),
    tf.keras.layers.Dense(DENSE_UNITS, activation = "relu"),
    tf.keras.layers.Dense(LABELS_COUNT, activation = "softmax")
])

rnn_model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",histogram_freq=1)

metrics_callback = MetricsCallback(test_data = valid_set_padded, y_true = validation_labels)

if not BALANCE_DATA:
    rnn_model.compile(optimizer=Adam(learning_rate = 1e-3,amsgrad=True), loss = SparseCategoricalCrossentropy(), metrics=['accuracy'])
    history = rnn_model.fit(train_set_padded, train_labels,validation_data = (valid_set_padded,validation_labels),\
                batch_size=BATCH_SIZE, epochs=EPOCHS,class_weight = class_weights,\
                callbacks=[tensorboard_callback,metrics_callback,EarlyStopping(patience=8,min_delta=0.005,mode='min',verbose = 1,restore_best_weights=True,monitor='val_loss'),ReduceLROnPlateau(monitor="val_loss",mode = 'min',factor=0.1,verbose = 1,min_delta=0.01,patience=3)])
else:
    rnn_model.compile(optimizer=Adam(learning_rate = 1e-3,amsgrad=True), loss = SparseCategoricalCrossentropy(), metrics=['accuracy'])
    history = rnn_model.fit(train_set_padded, train_labels,validation_data = (valid_set_padded,validation_labels),\
                    batch_size=BATCH_SIZE, epochs=EPOCHS,\
                    callbacks=[tensorboard_callback,metrics_callback,EarlyStopping(patience=8,min_delta=0.005,mode='min',verbose = 1,restore_best_weights=True,monitor='val_loss'),ReduceLROnPlateau(monitor="val_loss",mode = 'min',factor=0.1,verbose = 1,min_delta=0.01,patience=3)])

if not os.path.exists(SAVED_MODELS_DIR):
    os.mkdir(SAVED_MODELS_DIR)

rnn_model.save_weights(SAVED_MODELS_DIR + "/rnn_model_variant_" + str(VARIANT_NUM) + ".h5")

pickle_out = open(SAVED_MODELS_DIR + "/rnn_model_params_" + str(VARIANT_NUM) + ".pickle",'wb')
pickle.dump({'EMBEDDING_DIM':EMBEDDING_DIM,'MAX_TEXT_LEN':MAX_TEXT_LEN,'BATCH_SIZE':BATCH_SIZE,'EPOCHS':EPOCHS,\
             'train_set_size':len(train_set),'optimizer':str(rnn_model.optimizer),\
             'learning_rate':str(rnn_model.optimizer.learning_rate),\
             'conv_filters':CONV_FILTERS,'conv_kernel':CONV_KERNEL_SIZE,\
             'chunks_used':(len(train_set) + len(validation_set) + len(test_set)) // DATA_IN_CHUNK,\
            'RNN_UNITS':RNN_UNITS,'DENSE_UNITS':DENSE_UNITS,'LABELS_COUNT':LABELS_COUNT},pickle_out)
pickle_out.close()

"""## 4. Conclusion"""

if VAST:
    !tar -czf rnn_variant_6_logs.tar.gz logs

if DELETE_PICKLES_AFTER_TRAINING:
    if GDRIVE:
        !rm -r larger_dataset/ready_for_models
    else:
        !rm -r ../larger_dataset/ready_for_models

"""### 4.1 Results"""

results = rnn_model.evaluate(test_set_padded, test_labels, batch_size=BATCH_SIZE)
print("test loss, test acc:", results)

"""### 4.2 Tensorboard"""

# %load_ext tensorboard
# %tensorboard --logdir logs/fit