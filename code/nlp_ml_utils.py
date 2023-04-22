import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt

def create_train_test_split(X,y,vectorizer=False,test_size = 0.2):
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=test_size,stratify = y,random_state = 42)
    if vectorizer:
        vectorizer = vectorizer
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
    return x_train,x_test,y_train,y_test

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

def show_metrics(model,x_test,y_test,y_pred,X,y):
    show_conf_matrix(model, x_test, y_test)
    get_roc_auc(y_pred, y_test)
    show_classification_report(y_test,y_pred)
    show_learning_curve(model, X, y)

def translate_labels(labels):
    translated_labels = []
    for label in labels:
        if label > 3:
            translated_labels.append('Positive Sentiment')
        elif label < 3:
            translated_labels.append('Negative Sentiment')
        else:
            translated_labels.append('Neutral Sentiment')
    return translated_labels

def encode_sent(sentiments):
    translated_labels = sentiments
    encoded_sent = []
    for label in translated_labels:
        if label == 'Positive Sentiment':
            encoded_sent.append(1)
        elif label == 'Negative Sentiment':
            encoded_sent.append(2)
        else:
            encoded_sent.append(3)
    return encoded_sent

def perform_cleaning(df):
    df.drop_duplicates(inplace = True)
    df.isnull().value_counts()
    df.dropna(inplace = True)
    return df

def process_corpus(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token[0]) for token in tokens]
    return ' '.join(tokens)

def classify_sentiment(score):
    if score['neg'] > score['pos']:
        return "Negative Sentiment"
    elif score['neg'] < score['pos']:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"
    
def extract_sent_polarity(score):
    return score['compound']
