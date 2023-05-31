from all_libs_dl import *

def create_train_test_split(X,y,vectorizer=False,test_size = 0.2):
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=test_size,stratify = y,random_state = 42)
    if vectorizer:
        vectorizer = vectorizer
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
    return x_train,x_test,y_train,y_test

def clean_small_dataset(df):
    df['full_review_text'] = [new_text[8:] for new_text in df['full_review_text']]
    df['full_review_text'] = [new_text.replace("check-in","") for new_text in df['full_review_text']]
    df['full_review_text'] = [new_text.lstrip('0123456789.- ') for new_text in df['full_review_text']]
    df['full_review_text'] = [new_text.lstrip('s') for new_text in df['full_review_text']]
    df['star_rating'] = df['star_rating'].str[:2]
    df['star_rating'] = [int(rating) for rating in df['star_rating']]
    
    return df

def remove_stop_words(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

def get_trained_model(classifier,x_train,y_train):
    model = classifier
    model.fit(x_train, y_train)
    return model

def get_model_accuracies(model,x_train,x_test,y_train,y_test):
    train_accuracy = model.score(x_train,y_train)
    val_accuracy = model.score(x_test, y_test)
    return ['{:.2f}%'.format(train_accuracy * 100),'{:.2f}%'.format(val_accuracy * 100)]

def print_accuracies(accuracy_scores):
    print("Training Accuracy:" + accuracy_scores[0])
    print("Testing Accuracy:" + accuracy_scores[1])

def adjust_xgb_labels(y_train,y_test):
    y_train_xgb = [label - 1 for label in y_train]
    y_valid_xgb = [label - 1 for label in y_test]
    
    return y_train_xgb,y_valid_xgb

def readjust_labels(y_train_xgb,y_test_xgb):
    y_train = [label + 1 for label in y_train_xgb]
    y_test = [label + 1 for label in y_test_xgb]
    
    return y_train,y_test

def show_classification_report(y_test,y_pred):
    print("Classification Report = \n ",metrics.classification_report(y_test,y_pred,zero_division=0))

def show_conf_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
    
    disp.plot()
    plt.show()

def get_roc_auc(y_pred, y_test):
    roc_auc_dict = {}
    unique_classes = set(y_test)
    for per_class in unique_classes:
        other_classes = [chosen_class for chosen_class in unique_classes if chosen_class != per_class]
        new_y_true = [0 if this_class in other_classes else 1 for this_class in y_test]
        new_y_pred = [0 if that_class in other_classes else 1 for that_class in y_pred]
    
        roc_auc = roc_auc_score(new_y_true, new_y_pred, average = 'micro')
        roc_auc_dict[per_class] = roc_auc
        
    print("ROC/AUC for each class: \n")
    print(roc_auc_dict)
    print("\n")

def show_learning_curve(model, X, y,cv = 5):
    train_sizes_model, train_scores_model, valid_scores_model, *_ = learning_curve(model, X, y,cv=cv,
                    scoring='accuracy',n_jobs=-1)
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(x=train_sizes_model,y= train_scores_model.mean(axis=1), color='b')
    ax.scatter(x=train_sizes_model,y=valid_scores_model.mean(axis=1), color='r')
    ax.plot(train_sizes_model,train_scores_model.mean(axis=1), color='b')
    ax.plot(train_sizes_model,valid_scores_model.mean(axis=1), color='r')
    ax.set_xlabel('Training in blue, Testing in red')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve')
    plt.show()

def show_metrics(clf_name,model,x_test,y_test,y_pred,X,y,cv = 5,show_cm=True,show_roc_auc=True,show_cr=True,show_lr=True):
    if not show_cm and not show_roc_auc and not show_cr and not show_lr:
        print("All Metrics are turned off. Skipping.")
    else:
        print("PRINTING METRIC(S) FOR " + str(clf_name))
        if show_cm:
            show_conf_matrix(model, x_test, y_test)  
        if show_roc_auc:
            get_roc_auc(y_pred, y_test)
        if show_cr:
            show_classification_report(y_test,y_pred)
        if show_lr:
            show_learning_curve(model, X, y,cv = cv)

def translate_labels(labels):

    """
    This function takes in all labels, and converts them to their corresponding sentiment.

    labels: x => list of integers, Required
    """

    translated_labels = []
    for label in labels:
        if label > 3:
            translated_labels.append('Positive Sentiment')
        elif label < 3:
            translated_labels.append('Negative Sentiment')
        else:
            translated_labels.append('Neutral Sentiment')
    return translated_labels

def encode_sent(sentiments,positive_label = 1,negative_label = 2,neutral_label = 3):

    """
    This function takes in all translated labels (labels transformed for star ratings to sentiments), and converts them to numerical values.

    sentiments: x => list of strings, Required

    positive_label: x => Positive Integer, Default = 1

    negative_label: x => Positive Integer, Default = 2

    neutral_label: x => Positive Integer, Default = 3

    """

    translated_labels = sentiments
    encoded_sent = []
    for label in translated_labels:
        if label == 'Positive Sentiment':
            encoded_sent.append(positive_label)
        elif label == 'Negative Sentiment':
            encoded_sent.append(negative_label)
        else:
            encoded_sent.append(neutral_label)
    return encoded_sent

def perform_cleaning(df):
    df.drop_duplicates(inplace = True)
    df['full_review_text'].replace('', np.nan, inplace=True)
    df.dropna(inplace = True)
    return df

def get_pos_tag(tag):
    if tag.startswith('N'):
        return 'n'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('J'):
        return 'a'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def process_corpus(text):
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    lower = [word.lower() for word in tokens]
    no_stopwords = [word for word in lower if word not in stopwords]
    no_alpha = [word for word in no_stopwords if word.isalpha()]
    tokens_tagged = nltk.pos_tag(no_alpha)
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word[0],pos=get_pos_tag(word[1])) for word in tokens_tagged]
    preprocessed_text = lemmatized_text
    return ' '.join(preprocessed_text)

def classify_sentiment(score):
    if score['neg'] > score['pos']:
        return "Negative Sentiment"
    elif score['neg'] < score['pos']:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"
    
def extract_sent_polarity(score):
    return score['compound']

def create_wordcloud(df):
    unique_words = set(' '.join(df['full_review_text']).split())
    unique_wordcloud = WordCloud(width=800, height=400, background_color='white',stopwords=set(nltk.corpus.stopwords.words('english'))).generate(' '.join(unique_words))
    plt.figure(figsize=(12, 6))
    plt.imshow(unique_wordcloud)
    plt.axis('off')
    plt.title('Word Cloud of Unique Words')

def create_bar_chart_most_common_words(df):
    unique_words = set(' '.join(df['full_review_text']).split())
    review_text_no_stop_words = pd.Series([remove_stop_words(review) for review in df['full_review_text']])

    unique_word_count = review_text_no_stop_words.str.split(expand=True).stack().value_counts()
    top_unique_words = unique_word_count.loc[unique_word_count.index.isin(unique_words)].head(20)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_unique_words.values, y=top_unique_words.index)

    plt.title('Top 20 Most Frequent Unique Words')
    plt.xlabel('Word Count')
    plt.ylabel('Word')

def visualize_ratings_bar(labels):
    sns.countplot(x=labels)
    plt.xlabel('Star Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Star Ratings')
    plt.show()

def create_vector_space_viz(df):
    unique_words = set(' '.join(df['full_review_text']).split())
    review_text_no_stop_words = pd.Series([remove_stop_words(review) for review in df['full_review_text']])
    vectorizer = CountVectorizer()
    word_embeddings = vectorizer.fit_transform(review_text_no_stop_words)
    viz_words = 30
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(word_embeddings[:viz_words, :])
    fig, ax = plt.subplots(figsize=(14, 10))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        int_to_vocab = {i: word for i, word in enumerate(set(unique_words))}
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.8, fontsize=13, color='black', horizontalalignment='right', verticalalignment='bottom')

def visualize_ratings_pie(labels,use_dict=False):
    if use_dict:
        _, _, autotexts = plt.pie(labels.value_counts(),colors = ['blue','green','red','black','orange'],labels = list(labels.unique()),textprops={'color':"w"},autopct= '%1.1f%%')
        for autotext in autotexts:
            autotext.set_color('white')
    else:
        labels_dict = dict(labels.value_counts())
        _, _, autotexts = plt.pie(labels_dict.values(),colors = ['blue','green','red','black','orange'],labels=labels_dict.keys(),textprops={'color':"w"},autopct= '%1.1f%%')
        for autotext in autotexts:
            autotext.set_color('white')

def get_chunks(urls,limit=0,verbose = 1,base_name = "temp",file_path="",file_format='.csv',loading_chunks = True):
    #downloads all data from their url(s)
    for i,url in enumerate(urls):
        if limit:
            if i == limit:
                return
        if loading_chunks:
            file_name = base_name + str(i + 1)
        else:
            file_name = base_name
        #checks if file already exists
        if os.path.exists(file_path + file_name + file_format):
            print(f"{file_name} already exists.")
            continue
        if i % verbose == 0:
            print(f"Downloading {file_name}...")
        r = requests.get(url)
        with open(file_path + file_name + file_format, 'wb') as fd:
            for chunk in r.iter_content():
                #save file in the current directory of the notebook
                fd.write(chunk)
        if i % verbose == 0:
            print(f"{file_name} was downloaded successfully.")

def get_all_file_names(base_name,limit_num):
    return [base_name + str(num) for num in range(1,limit_num + 1)]

def read_chunks(files,file_path = "",file_format = ".csv"):
    #reads chunks csvs and converts them to a dataframe format
    final_df = []
    for file in files:
        df = pd.read_csv(file_path + file + file_format)
        final_df.append(df)
    #function returns a list of dfs
    return final_df

def group_up_chunks(dfs):
    #adds up all dataframes together
    return pd.concat(dfs)

def check_no_missing_data(shape_loaded,shape_actual):
    actual_shape_loaded = (shape_loaded / shape_actual) * 100
    return actual_shape_loaded

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
            
    return word_to_vec_map

class MetricsCallback(Callback):
    def __init__(self, test_data, y_true):
        # Should be the label encoding of your classes
        self.y_true = y_true
        self.test_data = test_data
        
    def on_epoch_end(self, epoch, logs=None):
        # Here we get the probabilities
        y_pred = self.model.predict(self.test_data)
        # Here we get the actual classes
        y_pred = tf.argmax(y_pred,axis=1)
        # Actual dictionary
        report_dictionary = classification_report(self.y_true, y_pred, output_dict = True)
        # Only printing the report
        print(classification_report(self.y_true,y_pred,output_dict=False))
        macro_f1_pred = f1_score(self.y_true, y_pred, average='weighted',zero_division=0)
        print(f"Macro Weighted F1-Score: {macro_f1_pred}")

def get_classes_count(y,start_label = 0):
    """

    takes in a list of labels, and returns a list of dictionaries containing the label as the key, and the number of samples for the label as a value.

    start_label: n | 0
            A positive integer that indicates the first numerical label in the list. If not given, the first label will be given the default value of 0.

    """
    samples = []
    y_unique_labels = list(y.unique())
    for i in range(start_label,len(y_unique_labels)):
        samples.append({i:len([label for label in y if label == y_unique_labels[i]])})
    return samples

def get_class_weights(labels_dict,mu=0.15):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    weights = dict()
    for i in keys:
        score = np.log((mu*total)/float(labels_dict[i]))
        weights[i] = score if score > 1 else 1
    return weights

def get_dependencies(url,file_name,file_extension):
    if os.path.exists(file_name + file_extension):
        return print(file_name + " already exists.")
    else:
        print(f"downloading {file_name}...")
        r = requests.get(url)
        with open(file_name + file_extension, 'wb') as fd:
            for chunk in r.iter_content():
                fd.write(chunk)