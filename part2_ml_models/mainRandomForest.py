import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from statistics import mode
import math
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc


#fortwnw ta dedomena apo thn bash dedomenwn 
#einai lista me arithmous pou antiprosopeyoun thn thesh ths lexis sto le3iko
(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

#metarepw thn panw lista sthn kanonikh me le3eis gia to ka8e review

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])

#metatrepw thn lista se binary 1 h 0 analoga an uparxei h le3h sto review
binary_vectorizer = CountVectorizer(binary=True, max_df=2000,min_df=100,max_features=100)
x_train_imdb_binary = binary_vectorizer.fit_transform(x_train_imdb)
x_test_imdb_binary = binary_vectorizer.transform(x_test_imdb)

print(
    'Vocabulary size:', len(binary_vectorizer.vocabulary_)
)

featuresData=[]
for i in binary_vectorizer.vocabulary_:
    featuresData.append(i)

x_train_imdb_binary = x_train_imdb_binary.toarray()
x_test_imdb_binary = x_test_imdb_binary.toarray()

#autos o kwdikas einai epeidh h bash mas exei sunolika 25000 paradeigmata ekpaideushs emeis 
#theloume na paroume ligotera genika gia na kathorisoume posa theloume na paroume pio polu gia dokimes
#gia na dokimasoume kai me liga kai me polla
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i in range(500):
    X_train.append(np.array(x_train_imdb_binary[i]))
    y_train.append(y_train_imdb[i])
    X_test.append(np.array(x_test_imdb_binary[i]))
    y_test.append(y_test_imdb[i])


for i in range(500,1000):
    X_test.append(np.array(x_test_imdb_binary[i]))
    y_test.append(y_test_imdb[i])
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

# kalw ton Random Forest ths bibliothhkhs
random_forest = RandomForestClassifier(n_estimators=20, oob_score=True, random_state=42)
# ekpaideuw ton algorithmo
random_forest.fit(X_train, y_train)
# metatropi tου X_train sε np.float32
X_train_float32 = X_train.astype(np.float32)
# apokthsh twn out-of-bag deigmatwn
oob_samples_indices = random_forest.estimators_[0].tree_.apply(X_train_float32)
oob_samples_mask = ~np.in1d(np.arange(len(X_train)), np.unique(oob_samples_indices))

X_dev = X_train[oob_samples_mask]
y_dev = y_train[oob_samples_mask]

y_pred_rf=random_forest.predict(X_test)


# upologizw thn akribeia
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')

#///////////////////////////////////////////////////////////////////////////////
#emfanizontai oi kampules mathhshs
def plot_learning_curve(estimator, title,
                        X_for_val, y_for_val,
                        X_for_test=None, y_for_test=None,
                        ylim=None,
                        val_cv=None,
                        test_cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    t=train_sizes

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    print(train_sizes)



    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_for_val, y_for_val,
        cv=val_cv, n_jobs=-1, scoring='accuracy',
        train_sizes=train_sizes)

    print(train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1,
                     color="green")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="green",
             label="Validation score")
    
    _, _, test_scores = learning_curve(estimator,
                                        X_for_test, y_for_test,
                                        cv=test_cv, n_jobs=-1,
                                        scoring='accuracy',
                                        train_sizes=t)


    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()


    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="red")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="red",
              label="Test score")


    plt.legend(loc="lower right")
    plt.show()
    return plt
# kalw gia na emfanistoun oi kampules mathhshs
plot_learning_curve(estimator=random_forest, title='Learning Curve for library Random forest',
                    X_for_val=np.concatenate((X_train, X_dev), axis=0),
                    y_for_val=np.concatenate((y_train, y_dev), axis=0),
                    X_for_test=X_test, y_for_test=y_test,
                    val_cv=None)

#//////////////////////////////////////////////////////////////////////////////////////
#pinakas akribeias, anaklhshs kai F1
dt = DecisionTreeClassifier(criterion='entropy')
rf = random_forest
rf.fit(X_train, y_train)
print(classification_report(y_train, rf.predict(X_train),
                            zero_division=1))
print(classification_report(y_test, rf.predict(X_test),
                            zero_division=1))
#/////////////////////////////////////////////////////////////////////////////////////
#kampulh akribeias anaklhshs
# upologismos pithanothtvn gia ta dedomena elegxou
y_scores = rf.predict(X_test)
# ypologismos akribeias kampulhs kai anaklhshs
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auc_score = auc(recall, precision)
# emfanizetai kampulh akribeias-anaklhshs
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='b', label='Precision-Recall Curve (AUC-PR = {:.2f})'.format(auc_score))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for library Random forest')
plt.legend()
plt.show()

#///////////////////////////////////////////////////////////////////////////////////////
# Orizoume mia lista me ta megethoi tou X_train
training_sizes = [100, 250, 300, 350, 500]
# dhmiourgoume keno pinaka gia na apothhkeusoume ta pososta orthothtas
results = []
for size in training_sizes:
    # epilegoume uposunolo diaforetikvn dedomenwn ekpaideushs pou theloume na emfanistei to pososto tous
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    for i in range(size):
        X_train.append(np.array(x_train_imdb_binary[i]))
        y_train.append(y_train_imdb[i])
        X_test.append(np.array(x_test_imdb_binary[i]))
        y_test.append(y_test_imdb[i])
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test)

    # problepseis gia ta dedomena ekpaideushs
    y_train_pred = random_forest.predict(X_train)

    # problepseis gia ta dedomena elegxou
    y_test_pred = random_forest.predict(X_test)

    # upologizoume thn akribeia gia ta dedomena ekpaideushs kai elegxou
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # apothhkeuoume ta dedomena sthn lista result
    results.append({'Training Size': size, 'Accuracy (Training)': accuracy_train, 'Accuracy (Test)': accuracy_test})

# dhmiourgoume data frame kai apothkeuoume ta apatelesmata me ta pososta
results_df = pd.DataFrame(results)
# ekthpvnoume ta pososta se pinaka
print(results_df)






