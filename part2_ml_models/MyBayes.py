import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

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

binary_vectorizer = CountVectorizer(binary=True, max_df=2000,min_df=100,max_features=8500)
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
#////////////////////////////////////////////////////////////////////////////////////////
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
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
#//////////////////////////////////////////////////////////////////////////////

#lave iposin to information gain 
class NaiveBayes:
    def __init__(self):
        self.class_probs = {}  # P(Y)
        self.feature_probs = {}  # P(X_i|Y)
        #print("init")

    def fit(self, X, y):
        # Calculate class probabilities P(Y)
       # print("fit")
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for c, count in zip(classes, counts):
            self.class_probs[c] = count / total_samples

            # Calculate P(X_i|Y) for each feature
            features_present = X[y == c, :]
            feature_probs = (np.sum(features_present, axis=0) + 1) / (count + 2)  # Laplace smoothing
            self.feature_probs[c] = feature_probs

    def predict(self, X):
      #  print("predict")
        predictions = []
        for x in X:
            # Calculate class probabilities using Bayes' theorem
            class_scores = {c: np.log(self.class_probs[c]) + np.sum(np.log(self.feature_probs[c][x == 1]))
                            for c in self.class_probs.keys()}

             # Choose the class with the highest log probability
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions
    

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self,deep=True):
        return {}


 
#emfanizei ton pinaka me tis akribeies gia diaforous arithmous paradeigmatvn ekpaideushs
    def plot_accuracy_table(self, X_train, y_train, X_test, y_test):
        training_sizes = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
        results = []

        for size in training_sizes:
            X_train_subset = X_train[:size]
            y_train_subset = y_train[:size]

            self.fit(X_train_subset, y_train_subset)

            y_train_pred = self.predict(X_train_subset)
            y_test_pred = self.predict(X_test)

            accuracy_train = accuracy_score(y_train_subset, y_train_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)

            results.append({'Training Size': size, 'Accuracy (Training)': accuracy_train, 'Accuracy (Test)': accuracy_test})

        results_df = pd.DataFrame(results)
        print(results_df)
    #emfanizei thn kampulh akribeias, anaklhshs kai F1
    def plot_precision_recall_curve(self, X, y):
        y_scores = self.predict(X)
        # upologizei thn akribeia kai anaklhsh
        precision, recall, thresholds = precision_recall_curve(y, y_scores)
        auc_score = auc(recall, precision)
        # ftiaxnei tia kampules
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='b', label=f'Precision-Recall Curve (AUC-PR = {auc_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for our Bayes')
        plt.legend()
        plt.show()
# Use Naive Bayes classifier
naive_bayes_classifier = NaiveBayes()

#edw ginetai akpaideush tou montelou
naive_bayes_classifier.fit(X_train, y_train)
y_pred_nb = naive_bayes_classifier.predict(X_test)
#emfanizetai o pinakas anaklhshs, akribeias kai F1
def plot_precision_recall_table(X_train, y_train, X_test, y_test):

    nb = naive_bayes_classifier
    nb.fit(X_train, y_train)
    print(classification_report(y_train, nb.predict(X_train),
                            zero_division=1))
    print(classification_report(y_test, nb.predict(X_test),
                            zero_division=1))


#emfanizontai oi kampules mathhshs
def plot_learning_curve(estimator, title,
                            X_for_val, y_for_val,
                            X_for_test=None, y_for_test=None,
                            ylim=None,
                            val_cv=None,
                            test_cv=None,
                            train_sizes=np.linspace(.1, 1.0, 5)):

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        #print(train_sizes)
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X_for_val, y_for_val,
            cv=val_cv, n_jobs=-1, scoring='accuracy',
            train_sizes=train_sizes)

        #print(train_sizes)

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
                                            train_sizes=train_sizes)


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
#kaloume gia na emfanistoun oi kampules mathhshs
plot_learning_curve(estimator=naive_bayes_classifier, title='Learning Curve for our Bayes',
                    X_for_val=X_train,
                    y_for_val=y_train,
                    X_for_test=X_test,y_for_test=y_test,
                    val_cv=None)
#kaloume gia na emfanistei h kampulh anaklhshs, akribeias kai F1
naive_bayes_classifier.plot_precision_recall_curve(X_test, y_test)
#kaloume gia na emfanistei o pinakas me thn akribeia, anaklhsh kai F1
plot_precision_recall_table(X_train, y_train, X_test, y_test)
#naive_bayes_classifier.plot_learning_curve(X_train, y_train)
naive_bayes_classifier.plot_accuracy_table(X_train, y_train, X_test, y_test)
# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("")
print(f'Naive Bayes Accuracy: {accuracy_nb}')


