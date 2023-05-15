#!/usr/bin/env python
# coding: utf-8

# # Word2Vec vs GloVe

# # * Goal of This File:
# 
# ##### 1. Import Libraries, Helper Functions, and Constants ==> Data Sourcing and Munging ==> Utility Functions ==> Loading the Data ==> Merging all Individual Files
# 
# ##### 2. Comparison ==> Class Weights ==> Word2Vec + GloVe
# 
# ##### 3. RNN with Word2Vec 
# 
# ##### 4. RNN with GloVe 
# 
# ##### 5. Conclusion ==> Results ==> Tensorboard

# ## (1) Import Libraries, Helper Functions and Load Constants

# In[1]:


get_ipython().system('pip install pandas numpy nltk scikit-learn wordcloud seaborn gensim tensorflow imblearn xgboost matplotlib unrar pyunpack more-itertools patool keras-tqdm > /dev/null')


# In[2]:


VAST = True

if VAST:
    get_ipython().system('sudo apt-get install unrar')
    get_ipython().system('sudo apt-get install rar')
    
GDRIVE = True


# In[ ]:


UTILS_URL = 'https://f005.backblazeb2.com/file/gp-support-files/nlp_ai_utils.py'
UPDATING_VALUES_URL = 'https://f005.backblazeb2.com/file/gp-support-files/updating_values.py'
ALL_LIBS_URL = 'https://f005.backblazeb2.com/file/gp-support-files/all_libs_dl.py'
CHUNKS_URLS_FILE_URL = 'https://f005.backblazeb2.com/file/gp-support-files/chunks_urls.py'

UTILS_FILE_NAME = 'nlp_ai_utils'
UPDATING_VALUES_FILE_NAME = 'updating_values'
ALL_LIBS_FILE_NAME = 'all_libs_dl'
CHUNKS_URLS_FILE_NAME = 'chunks_urls'

DEP_FILE_EXT = '.py'


# In[ ]:


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


# In[ ]:


get_dependencies(UTILS_URL,UTILS_FILE_NAME,DEP_FILE_EXT)
get_dependencies(UPDATING_VALUES_URL,UPDATING_VALUES_FILE_NAME,DEP_FILE_EXT)
get_dependencies(ALL_LIBS_URL,ALL_LIBS_FILE_NAME,DEP_FILE_EXT)
get_dependencies(CHUNKS_URLS_FILE_URL,CHUNKS_URLS_FILE_NAME,DEP_FILE_EXT)


# In[3]:


from nlp_ai_utils import *
from chunks_urls import CHUNKS_URLS
from updating_values import *


# In[4]:


TF_ENABLE_ONEDNN_OPTS = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_URLS = CHUNKS_URLS
GLOVE_URL = 'https://f005.backblazeb2.com/file/glove-embeddings-dims/glove.6B.100d.txt'
READY_DATA_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/ready_for_models.rar'
CLASS_WEIGHTS_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/class_weights.pickle'
UNIQUE_WORDS_URL = 'https://f005.backblazeb2.com/file/yelp-dataset-ready-for-models/unique_words.pickle'
LIMIT = DATA_LIMIT
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
USE_READY_DATA = False
USE_MODIFIED_LABELS = True


# ### 1.1 Data Sourcing and Munging

# #### 1.1.1 Loading The Data

# In[5]:


if not USE_READY_DATA:
    if not os.path.exists(LARGER_DATASET_PATH):
        os.mkdir(LARGER_DATASET_PATH)
    if not os.path.exists(PREPROCESSED_CHUNKS_PATH):
        os.mkdir(PREPROCESSED_CHUNKS_PATH)
    get_chunks(DATA_URLS,LIMIT,1,BASE_FILE_NAME,PREPROCESSED_CHUNKS_PATH,FILE_FORMAT)


# #### 1.1.2 Merging all Individual Files

# In[6]:


#get all names of downloaded files
if not USE_READY_DATA:
    all_file_names = get_all_file_names(BASE_FILE_NAME,LIMIT)


# In[7]:


#read all chunks into a list
if not USE_READY_DATA:
    list_dfs = read_chunks(all_file_names,PREPROCESSED_CHUNKS_PATH,FILE_FORMAT)


# In[8]:


#concatenate all chunks into a singular df
if not USE_READY_DATA:
    df = group_up_chunks(list_dfs)


# In[9]:


#check how much of the data was actually downloaded
if not USE_READY_DATA:
    percent_loaded = check_no_missing_data(df.shape[0],ACTUAL_DATA_SHAPE)
    percent_loaded


# In[10]:


if not USE_READY_DATA:
    review_data = df[['text', 'stars']].copy()


# In[11]:


if not USE_READY_DATA:
    review_data.reset_index(inplace = True)
    review_data.drop(['index'],axis = 1,inplace = True)


# In[12]:


if not USE_READY_DATA:
    review_data.rename(columns = {'text':'full_review_text','stars':'star_rating'}, inplace = True)


# In[13]:


if not USE_READY_DATA:
    review_data.isnull().sum()


# In[14]:


if not USE_READY_DATA:
    review_data.dropna(inplace = True)


# In[15]:


if not USE_READY_DATA:
    review_data['full_review_text'].replace('', np.nan, inplace=True)
    review_data.dropna(inplace = True)


# ## (2) Comparison

# In[16]:


if not USE_READY_DATA:
    X = review_data['full_review_text']
    y = review_data['star_rating']


# In[17]:


if not USE_READY_DATA:
    X = pd.Series([str(text) for text in X])


# In[18]:


if USE_MODIFIED_LABELS:
    translated_labels = translate_labels(y)
    y = pd.Series(encode_sent(translated_labels))


# In[19]:


if not USE_READY_DATA:
    y = [label - 1 for label in y]


# ### 2.0 Classes' Shapes Explanation
# 
# #### Classes' shapes before and after (if USE_MODIFIED_LABELS = True):
# 
# BEFORE: 1 ==> Very Bad ==> AFTER: 2 ==> Negative Sentiment
# 
# BEFORE: 2 ==> Bad ==> AFTER: 2  ==> Negative Sentiment
# 
# BEFORE: 3 ==> Ok ==> AFTER: 3  ==> Neutral Sentiment
# 
# BEFORE: 4 ==> Good ==> AFTER: 1  ==> Positive Sentiment
# 
# BEFORE: 5 ==> Very Good ==> AFTER: 1  ==> Positive Sentiment
# 
# THEN: 1 is subtracted from each label. So, the Labels go FROM: 2,3,1 TO: 1,2,0
# 
# #### Classes' shapes before and after (if USE_MODIFIED_LABELS = False):
# 
# 1 is subtracted from each label. So, the Labels go FROM: 1,2,3,4,5 TO: 0,1,2,3,4
# 
# Why is 1 subracted from each label? To bring the data into the preferred shape of the class weights in both sklearn and keras (starting label is 0, not 1).

# ### 2.1 Class Weights

# In[20]:


if not os.path.exists(PICKLES_DIR):
    os.mkdir(PICKLES_DIR)


# In[21]:


if USE_READY_DATA:
    get_chunks([CLASS_WEIGHTS_URL],0,1,'class_weights',PICKLES_DIR + '/','.pickle',False)


# In[22]:


if USE_READY_DATA:
    class_weights = pickle.load(open(PICKLES_DIR + "/class_weights.pickle", "rb"))


# In[23]:


if not USE_READY_DATA:
    y = pd.Series(y)
    class_weights = compute_class_weight(class_weight = "balanced",classes = np.unique(y),y=y)
    class_weights = dict(zip(np.unique(y), class_weights))


# ### 2.2 Word2Vec

# In[24]:


if not os.path.exists(PICKLES_DIR + "/w2v_model.model"):
    print("Creating Embedding from scratch.")
    w2v_model = Word2Vec(sentences=[nltk.word_tokenize(text) for text in X], vector_size=100, window=5, min_count=1, workers=4)
    w2v_model.save(PICKLES_DIR + "/w2v_model.model")
else:
    print("Embedding found.")
    w2v_model = Word2Vec.load(PICKLES_DIR + "/w2v_model.model")


# In[25]:


if os.path.exists(PICKLES_DIR + "/w2v_model.model"):
    w2v_model = Word2Vec.load(PICKLES_DIR + "/w2v_model.model")


# In[26]:


word_index_gensim = {}

for i, word in enumerate(w2v_model.wv.key_to_index):
    word_index_gensim[word] = i


# In[27]:


vocab_size = len(word_index_gensim)
embedding_dim_gensim = w2v_model.vector_size

embedding_matrix_gensim = np.zeros((vocab_size, embedding_dim_gensim))

for word, i in word_index_gensim.items():
    if word in w2v_model.wv.key_to_index:
        embedding_matrix_gensim[i] = w2v_model.wv.get_vector(word)

EMBEDDING_LAYER_WORD2VEC = Embedding(vocab_size,
                            embedding_dim_gensim,
                            weights=[embedding_matrix_gensim],
                            trainable=True)


# ### 2.3 GloVe

# In[28]:


if not os.path.exists(GLOVE_FILES_DIR):
    os.mkdir(GLOVE_FILES_DIR)


# In[29]:


if not USE_READY_DATA:
    get_chunks([GLOVE_URL],0,1,"glove.6B.100d",GLOVE_FILES_DIR + '/','.txt',False)


# In[30]:


if not USE_READY_DATA:
    print('Getting Unique Words...')
    UNIQUE_WORDS = set(' '.join(X).split())


# In[31]:


if USE_READY_DATA:
    get_chunks([UNIQUE_WORDS_URL],0,1,'unique_words',PICKLES_DIR + '/','.pickle',False)


# In[32]:


if USE_READY_DATA:
    UNIQUE_WORDS = pickle.load(open(PICKLES_DIR + "/unique_words.pickle", "rb"))


# ### 2.4 Hyperparameters

# In[33]:


VOCAB_SIZE = len(UNIQUE_WORDS)
RNN_UNITS = 64
CONV_FILTERS = 48
CONV_KERNEL_SIZE = 3
DROPOUT_PERCENT = 0.2
DENSE_UNITS = 512
LABELS_COUNT = len(y.unique())
EMBEDDING_DIM = 100
MAX_TEXT_LEN = 200
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"
BATCH_SIZE = 64
EPOCHS = 25


# ### 2.5 Data Split
# 
# ##### ==> here is a quick explaination of how the dataset will be split using a smaller sample example.
# ##### ==> dataset => 100
# ##### ==> train_set => tr_s (example: 80)
# ##### ==> valid_set => vs (example: 10)
# ##### ==> test_set => te_s (example: 10)
# ##### ==> t = tr_s (80) + vs (10)
# ##### ==> train_set = x[:80]
# ##### ==> valid_set = x[80:t]
# ##### ==> test_set = x[t:] why t? because => vs = ts
# 
# ##### use this guideline if you are confused about how the train-validation-test split was done. Also, this is a future guide for me as well in case I forget.

# * train_set_size = 6,990,280 * 0.8 = 5,592,224
# * valid_set_size = 6,990,280 * 0.1 = 699,028
# * train_plus_valid = 5,592,224 + 699,028 = 6,291,252
# 
# ==> To Confirm: test_size = 6,990,280 - 6,291,252 = 699,028
# 
# * train_set = [0:5,592,224]
# * train_labels = [0:5,592,224]
# * validation_set = [5,592,224:6,291,252] ==> 699,028
# * validation_labels = [5,592,224:6,291,252] ==> 699,028
# * test_set = [6,291,252,6,990,280] ==> 699,028
# * test_labels = [6,291,252,6,990,280] ==> 699,028

# In[34]:


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


# In[35]:


if not os.path.exists(LARGER_DATASET_PATH):
    os.mkdir(LARGER_DATASET_PATH)


# In[36]:


if not os.path.exists(READY_DATASET_PATH):
    os.mkdir(READY_DATASET_PATH)


# In[37]:


if USE_READY_DATA:
    get_chunks([READY_DATA_URL],0,1,'ready_for_models',READY_DATASET_PATH,'.rar',False)
    Archive(os.path.join(READY_DATASET_PATH,"ready_for_models.rar")).extractall(READY_DATASET_PATH)


# In[38]:


if USE_READY_DATA:
    train_set_padded = pickle.load(open(READY_DATASET_PATH + "train_set_padded.pickle", "rb"))
    train_labels = pickle.load(open(READY_DATASET_PATH + "train_labels.pickle", "rb"))
    valid_set_padded = pickle.load(open(READY_DATASET_PATH + "valid_set_padded.pickle", "rb"))
    validation_labels = pickle.load(open(READY_DATASET_PATH + "validation_labels.pickle", "rb"))
    test_set_padded = pickle.load(open(READY_DATASET_PATH + "test_set_padded.pickle", "rb"))
    test_labels = pickle.load(open(READY_DATASET_PATH + "test_labels.pickle", "rb"))
    EMBEDDING_LAYER = pickle.load(open(READY_DATASET_PATH + "EMBEDDING_LAYER.pickle", "rb"))


# In[39]:


# NOTE: THIS CELL TAKES A WHILE TO RUN.
if not os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle"):
    print("Tokenizing the Dataset...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_set)
    words_to_index = tokenizer.word_index


# In[40]:


# NOTE: THIS CELL TAKES A WHILE TO RUN.
if not os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle"):
    print("Padding the Training Set.")
    train_sequences = tokenizer.texts_to_sequences(train_set)
    train_set_padded = pad_sequences(train_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
if not os.path.exists(READY_DATASET_PATH + "valid_set_padded.pickle"):
    print("Padding the Validation Set.")
    valid_sequences = tokenizer.texts_to_sequences(validation_set)
    valid_set_padded = pad_sequences(valid_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
if not os.path.exists(READY_DATASET_PATH + "test_set_padded.pickle"):
    print("Padding the Testing Set.")
    test_sequences = tokenizer.texts_to_sequences(test_set)
    test_set_padded = pad_sequences(test_sequences, maxlen=MAX_TEXT_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)


# In[41]:


if not os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle"):
    print("Re-formatting Train Set Shape.")
    train_set_padded = np.array(train_set_padded)
    
if not os.path.exists(READY_DATASET_PATH + "train_labels.pickle"):
    print("Re-formatting Train Labels Shape.")
    train_labels = np.array(train_labels)
    
if not os.path.exists(READY_DATASET_PATH + "valid_set_padded.pickle"): 
    print("Re-formatting Validation Set Shape.")
    valid_set_padded = np.array(valid_set_padded)

if not os.path.exists(READY_DATASET_PATH + "validation_labels.pickle"):
    print("Re-formatting Validation Labels Shape.")
    validation_labels = np.array(validation_labels)

if not os.path.exists(READY_DATASET_PATH + "test_set_padded.pickle"):
    print("Re-formatting Testing Set Shape.")
    test_set_padded = np.array(test_set_padded)

if not os.path.exists(READY_DATASET_PATH + "test_labels.pickle"):
    print("Re-formatting Testing Labels Shape.")
    test_labels = np.array(test_labels)


# In[42]:


if not USE_READY_DATA:
    word_to_vec_map = read_glove_vector(GLOVE_FILES_DIR + '/glove.6B.100d.txt')


# In[43]:


if not os.path.exists(READY_DATASET_PATH + "EMBEDDING_LAYER_GLOVE.pickle"):
    print("Generating Embedding From Scratch.")
    vocab_mapping = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_mapping, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector

    EMBEDDING_LAYER_GLOVE = Embedding(input_dim=vocab_mapping,\
                                output_dim=embed_vector_len, input_length=MAX_TEXT_LEN, weights = [emb_matrix], trainable=True)


# In[44]:


if os.path.exists(READY_DATASET_PATH + "train_set_padded.pickle"):
    print("Using Pickle File!")
    train_set_padded = pickle.load(open(READY_DATASET_PATH + "train_set_padded.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "train_set_padded.pickle",'wb')
    pickle.dump(train_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(READY_DATASET_PATH + "train_labels.pickle"):
    print("Using Pickle File!")
    train_labels = pickle.load(open(READY_DATASET_PATH + "train_labels.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "train_labels.pickle",'wb')
    pickle.dump(train_labels,pickle_out)
    pickle_out.close()


# In[45]:


if os.path.exists(READY_DATASET_PATH + "valid_set_padded.pickle"):
    print("Using Pickle File!")
    valid_set_padded = pickle.load(open(READY_DATASET_PATH + "valid_set_padded.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "valid_set_padded.pickle",'wb')
    pickle.dump(valid_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(READY_DATASET_PATH + "validation_labels.pickle"):
    print("Using Pickle File!")
    validation_labels = pickle.load(open(READY_DATASET_PATH + "validation_labels.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "validation_labels.pickle",'wb')
    pickle.dump(validation_labels,pickle_out)
    pickle_out.close()


# In[46]:


if os.path.exists(READY_DATASET_PATH + "test_set_padded.pickle"):
    print("Using Pickle File!")
    test_set_padded = pickle.load(open(READY_DATASET_PATH + "test_set_padded.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "test_set_padded.pickle",'wb')
    pickle.dump(test_set_padded,pickle_out)
    pickle_out.close()
    
if os.path.exists(READY_DATASET_PATH + "test_labels.pickle"):
    print("Using Pickle File!")
    test_labels = pickle.load(open(READY_DATASET_PATH + "test_labels.pickle", "rb"))
else:
    pickle_out = open(READY_DATASET_PATH + "test_labels.pickle",'wb')
    pickle.dump(test_labels,pickle_out)
    pickle_out.close()


# In[47]:


if os.path.exists(READY_DATASET_PATH + "EMBEDDING_LAYER_GLOVE.pickle"):
    print("Using Pickle File!")
    EMBEDDING_LAYER_GLOVE = pickle.load(open(READY_DATASET_PATH + "EMBEDDING_LAYER_GLOVE.pickle", "rb"))
else:
    print("Pickling Embedding Layer!")
    pickle_out = open(READY_DATASET_PATH + "EMBEDDING_LAYER_GLOVE.pickle",'wb')
    pickle.dump(EMBEDDING_LAYER_GLOVE,pickle_out)
    pickle_out.close()


# ## (3) RNN with Word2Vec

# In[48]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[49]:


tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
rnn_model_w2v = tf.keras.models.Sequential([
    EMBEDDING_LAYER_WORD2VEC,
    tf.keras.layers.Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same', activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(keepdims = True),
    tf.keras.layers.SimpleRNN(RNN_UNITS),
    tf.keras.layers.Dense(DENSE_UNITS, activation = "relu"),
    tf.keras.layers.Dense(LABELS_COUNT, activation = "softmax")
])


# In[50]:


rnn_model_w2v.summary()


# In[51]:


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


# In[52]:


metrics_callback = MetricsCallback(test_data = valid_set_padded, y_true = validation_labels)


# In[53]:


rnn_model_w2v.compile(optimizer=Adam(), loss = SparseCategoricalCrossentropy(), metrics=['accuracy'])
rnn_model_w2v.fit(train_set_padded, train_labels,validation_data = (valid_set_padded,validation_labels),\
              batch_size=BATCH_SIZE, epochs=EPOCHS,class_weight=class_weights,\
              callbacks=[tensorboard_callback,metrics_callback,EarlyStopping(patience=4),ReduceLROnPlateau(factor=0.1, patience=2)])


# In[54]:


if not os.path.exists(SAVED_MODELS_DIR):
    os.mkdir(SAVED_MODELS_DIR)


# In[56]:


rnn_model_w2v.save_weights(SAVED_MODELS_DIR + "/rnn_model_w2v_" + 'base' + ".h5")


# In[58]:


pickle_out = open(SAVED_MODELS_DIR + "/rnn_model_w2v_params_" + str(TRAINED_MODELS_COUNT) + ".pickle",'wb')
pickle.dump({'EMBEDDING_DIM':EMBEDDING_DIM,'MAX_TEXT_LEN':MAX_TEXT_LEN,'BATCH_SIZE':BATCH_SIZE,'EPOCHS':EPOCHS,\
             'train_set_size':len(train_set),'RNN_UNITS':RNN_UNITS,\
             'CONV_FILTERS':CONV_FILTERS,'CONV_KERNEL_SIZE':CONV_KERNEL_SIZE,
             'chunks_used':(len(train_set) + len(validation_set) + len(test_set)) // DATA_IN_CHUNK,\
                'DENSE_UNITS':DENSE_UNITS,'LABELS_COUNT':LABELS_COUNT},pickle_out)
pickle_out.close()


# ## (4) RNN with GloVe

# In[59]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[62]:


tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
rnn_model_glove = tf.keras.models.Sequential([
    EMBEDDING_LAYER_GLOVE,
    tf.keras.layers.Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same', activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(keepdims = True),
    tf.keras.layers.SimpleRNN(RNN_UNITS),
    tf.keras.layers.Dense(DENSE_UNITS, activation = "relu"),
    tf.keras.layers.Dense(LABELS_COUNT, activation = "softmax")
])


# In[63]:


rnn_model_glove.summary()


# In[64]:


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


# In[65]:


metrics_callback = MetricsCallback(test_data = valid_set_padded, y_true = validation_labels)


# In[66]:


rnn_model_glove.compile(optimizer=Adam(), loss = SparseCategoricalCrossentropy(), metrics=['accuracy'])
rnn_model_glove.fit(train_set_padded, train_labels,validation_data = (valid_set_padded,validation_labels),\
              batch_size=BATCH_SIZE, epochs=EPOCHS,class_weight=class_weights,\
              callbacks=[tensorboard_callback,metrics_callback,EarlyStopping(patience=4),ReduceLROnPlateau(factor=0.1, patience=2)])


# In[67]:


rnn_model_glove.save_weights(SAVED_MODELS_DIR + "/rnn_model_glove_" + 'base' + ".h5")


# In[69]:


pickle_out = open(SAVED_MODELS_DIR + "/rnn_model_glove_params_" + str(TRAINED_MODELS_COUNT) + ".pickle",'wb')
pickle.dump({'EMBEDDING_DIM':EMBEDDING_DIM,'MAX_TEXT_LEN':MAX_TEXT_LEN,'BATCH_SIZE':BATCH_SIZE,'EPOCHS':EPOCHS,\
             'train_set_size':len(train_set),'RNN_UNITS':RNN_UNITS,\
             'CONV_FILTERS':CONV_FILTERS,'CONV_KERNEL_SIZE':CONV_KERNEL_SIZE,
             'chunks_used':(len(train_set) + len(validation_set) + len(test_set)) // DATA_IN_CHUNK,\
                'DENSE_UNITS':DENSE_UNITS,'LABELS_COUNT':LABELS_COUNT},pickle_out)
pickle_out.close()


# ## (5) Conclusion

# In[70]:


if VAST:
    get_ipython().system('tar -czf rnn_logs.tar.gz logs')


# ### 5.1 Results

# ### 5.2 Tensorboard

# In[ ]:


# %load_ext tensorboard
# %tensorboard --logdir logs/fit

