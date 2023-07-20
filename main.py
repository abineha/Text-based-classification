# This is a sample Python script.
import os
import random
import string
import nltk
from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import nltk
from nltk.corpus import wordnet


def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if '_' not in lemma.name() and '-' not in lemma.name():  # Filter out multi-word synonyms
                synonyms.add(lemma.name())
    return list(synonyms)


BASE_DIR='/Users/abine/PycharmProjects/nlp'
LABELS=['Arrange','Act','Assert']

# [(label,text),(label,text)]
stop_words=set(stopwords.words('english'))
stop_words.add('said')
stop_words.add('mr')


def create_data_set():
    with open('data.txt','w',encoding='utf8') as outfile:
        for label in LABELS:
            dir='%s/%s' % (BASE_DIR, label)
            for filename in os.listdir(dir):
                fullfilename='%s/%s'% (dir, filename)
                print(fullfilename)
                with open(fullfilename,'rb') as file:
                    text= file.read().decode(errors='replace').replace('\n','')
                    outfile.write('%s\t%s\t%s\n'%(label,filename,text))


def setup_docs():
    docs=[]  #  (label,text)
    with open('data.txt','r',encoding='utf8') as datafile:
        for row in datafile:
            if row!="\n":
                parts=row.strip().split('\t')
                for i in (parts[2].split(" ")):
                    doc=(parts[0],i)
                    docs.append(doc)
    print("original doc:")
    print(docs)
    expanded_dataset = []
    for label, text in docs:
        expanded_dataset.append((label, text))
        synonyms = get_synonyms(text)
        for synonym in synonyms:
            expanded_dataset.append((label, synonym))
    print("expanded doc:")
    print(expanded_dataset)
    return expanded_dataset


def get_splits(docs):
    #scramble docs
    random.shuffle(docs)
    X_train=[]  #training docs
    y_train=[]  #corresponding training labels

    X_test=[]   #test docs
    y_test=[]   #corresponding test label

    pivot=int(0.70*len(docs))

    for i in range(0,pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])

    for i in range(pivot,len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0])
    print("X_train")
    print(X_train)
    print("y_train")
    print(y_train)
    print('x_test')
    print(X_test)
    print("y_test")
    print(y_test)

    return X_train,X_test,y_train,y_test


def evaluate_classifier(title,classifier,vectorizer,X_test,y_test):
    X_test_tfidf=vectorizer.transform(X_test)
    y_pred=classifier.predict(X_test_tfidf)

    precision=metrics.precision_score(y_test,y_pred,average='micro')
    recall=metrics.recall_score(y_test,y_pred,average='micro')
    f1=metrics.f1_score(y_test,y_pred,average='micro')

    print("%s\t%f\t%f\t%f\n" % (title,precision,recall,f1))


def train_classifier(docs):
    X_train,X_test,y_train,y_test=get_splits(docs)

    #the object that turns text into vectors
    vectorizer=CountVectorizer(stop_words='english',ngram_range=(1,1),min_df=1,analyzer='word')

    #create doc-term matrix
    dtm=vectorizer.fit_transform(X_train)

    #train naive baiyes classifier
    naives_bayes_classifier=MultinomialNB().fit(dtm,y_train)

    evaluate_classifier("Naive Bayes\tTRAIN\t",naives_bayes_classifier,vectorizer,X_train,y_train)
    evaluate_classifier("Naive Bayes\tTEST\t",naives_bayes_classifier,vectorizer,X_test,y_test)

    #store the classifier
    clf_filename='naive_bayes_classifier.pkl'
    pickle.dump(naives_bayes_classifier,open(clf_filename,'wb'))

    #also store the vectorizer so we can transform new data
    vec_filename='count_vectorizer.pkl'
    pickle.dump(vectorizer,open(vec_filename,'wb'))


def classify(text):
    #load classifier
    clf_filename='naive_bayes_classifier.pkl'
    nb_clf=pickle.load(open(clf_filename,'rb'))

    #vectorize the new text
    vec_filename='count_vectorizer.pkl'
    vectorizer=pickle.load(open(vec_filename,'rb'))

    pred=nb_clf.predict(vectorizer.transform([text]))
    print(pred[0])


def print_hi(name):
    # Use a brSeakpoint in the code line below to debug your script.
    create_data_set()
    docs=setup_docs()
    print(docs)
    train_classifier(docs)
    l=['literacy', 'banking', 'online', 'budgeting', 'savings', 'parents', 'parent', 'financial', 'teach', 'educational']
    for i in l:
        new_doc=i
        print(new_doc)
        classify(new_doc)
        print()
    new_doc="Run"
    print(new_doc)
    classify(new_doc)
    print("Done")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
