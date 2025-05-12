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
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding, Flatten
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from queue import Queue
from joblib import Parallel, delayed



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


#/////////////////////////////////////////////////////////////////////////////////////////////////
class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
class ID3:
    def __init__(self, features, max_depth=None):
        self.tree = None
        self.features = features
        self.max_depth = max_depth
    
    def fitID3(self, x, y):
        '''
        creates the tree
        '''
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common, depth=0)
        return self.tree
    
       
       
    def create_tree(self, x_train, y_train, features, category, depth):
        
        # check empty data
        if len(x_train) == 0 or depth == self.max_depth:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
        
        # check all examples belonging in one category
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
        
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        
        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))
        
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, 
                                           category=m,depth=depth + 1)  # go left for X = 1
        
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m,depth=depth + 1)  # go right for X = 0
        
        return root
 
    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)
 
        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x
 
            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    
 
        
 
    def predictID3(self, x):
        predicted_classes = list()
 
        for unlabeled in x:  # for every example 
            tmp = self.tree  # begin at root
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)

        
#//////////////////////////////////////////////////////////////////////////////////////////////////
def fit_tree(tree_index, X, y, featuresData):
    X_boot, y_boot = resample(X, y, random_state=np.random.randint(0, 100))
    tree = ID3(featuresData, max_depth=4)
    tree.fitID3(X_boot, y_boot)
    return tree    

class RandomForest:
    def get_params(self, deep=True):
        return {'n_estimators': self.n_estimators}
    
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.trees = []
 


    def fit(self, X, y):
        devs = []

        def fit_tree_parallel(tree_index):
            return fit_tree(tree_index, X, y, featuresData)

        self.trees = Parallel(n_jobs=-1)(delayed(fit_tree_parallel)(i) for i in range(self.n_estimators))

        unused_samples = []
        for n in range(self.n_estimators):
            X_boot, y_boot = resample(X, y, random_state=np.random.randint(0, 100))
            unused = np.array([i for i in range(X.shape[0]) if i not in np.unique(X_boot, axis=0)])
            unused_samples.append(unused)

        all_unused_samples = np.unique(np.concatenate(unused_samples))
        X_dev = X[all_unused_samples]
        y_dev = y[all_unused_samples]

        devs.append(X_dev)
        devs.append(y_dev)
        return devs
 
    #def predict(self, X):
        #problepseiis apo kathe dentro gia ola ta paradeigmata ekpaideushs
        #predictions = np.array([tree.predictID3(X) for tree in self.trees])
        # emeis omws theloume gia kathe paradeigma ekpaideushs tis problepseis apo ta dentra
        #votingOfX=[]
        #allpre=[]
        #for j in range(predictions.shape[1]):
            #for i in range(predictions.shape[0]):
                #votingOfX.append(predictions[i][j])
            #votingOfX=np.array(votingOfX)
            #final_predictions = mode(votingOfX.flatten())
            #allpre.append(final_predictions)
            #votingOfX=[]
        #allpre=np.array(allpre)
        #o pinakas allpre exei tis apofaseis gia kathe paradeigma ekpaideushs gia ola ta dentra
 
        #return allpre
    def predict(self, X):
        predictions = np.array([tree.predictID3(X) for tree in self.trees])
        all_predictions = []

        for j in range(predictions.shape[1]):
            voting_of_X = predictions[:, j]
            final_prediction = np.argmax(np.bincount(voting_of_X))
            all_predictions.append(final_prediction)

        return np.array(all_predictions)
   



X_train=[]
y_train=[]
X_test=[]
y_test=[]
#autos o kwdikas einai epeidh h bash mas exei sunolika 25000 paradeigmata ekpaideushs emeis 
#theloume na paroume ligotera genika gia na kathorisoume posa theloume na paroume pio polu gia dokimes
#gia na dokimasoume kai me liga kai me polla
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



# Xrhsh tou RandomForest pou eftiaksa egw
random_forest = RandomForest(n_estimators=20)
devs=[]
devs=random_forest.fit(X_train, y_train)
y_pred_rf=random_forest.predict(X_test)


# upologismos ths akribeias
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

plot_learning_curve(estimator=random_forest, title='Learning Curve of our Random Forest',
                    X_for_val=np.concatenate((X_train, devs[0]), axis=0),
                    y_for_val=np.concatenate((y_train, devs[1]), axis=0),
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
plt.title('Precision-Recall Curve of our Random Forest')
plt.legend()
plt.show()

#///////////////////////////////////////////////////////////////////////////////////////
# Orizoume mia lista me ta megethoi tou X_train
training_sizes = [100, 150, 200, 350, 500]
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

#//////////////////////////////////////////////////////////////////////////////////////////
