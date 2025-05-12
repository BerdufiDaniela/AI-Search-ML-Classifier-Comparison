import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Embedding, SimpleRNN
import tensorflow.compat.v1 as tf_compat
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd


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
binary_vectorizer = CountVectorizer(binary=True, max_df=2000,min_df=200,max_features=5000)
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
#///////////////////////////////////////////////////////////////////////////////////////
#autos o kwdikas einai epeidh h bash mas exei sunolika 25000 paradeigmata ekpaideushs emeis 
#theloume na paroume ligotera genika gia na kathorisoume posa theloume na paroume pio polu gia dokimes
#gia na dokimasoume kai me liga kai me polla
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i in range(25000):
    # Προσθήκη του x_train_imdb_binary[i] στον X_train
    X_train.append(np.array(x_train_imdb_binary[i]))
    
    # Προσθήκη του y_train_imdb[i] στον y_train
    y_train.append(y_train_imdb[i])

    # Προσθήκη του x_test_imdb_binary[i] στον X_test
    X_test.append(np.array(x_test_imdb_binary[i]))
    
    # Προσθήκη του y_test_imdb[i] στον y_test
    y_test.append(y_test_imdb[i])
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

#//////////////////////////////////////////////////////////////////////////////////////////

#Μερος Γ
#------------------------------------MLP------------------------------------------------
#edw einai pou ftiaxnoume kai ekpaideuoume to mlp montelo
#dhmiourgoume to montelo 
mlp = tf.keras.models.Sequential()
#prosthetoume ta xarakthristika gia to embending / edv dhmiourghtai to epipedo embending
mlp.add(Embedding(input_dim=len(binary_vectorizer.vocabulary_), output_dim=100, input_length=len(binary_vectorizer.vocabulary_),trainable=True))
#prosthetoume flatten gia na metetrapoun ta dedomena se ena epipedo
mlp.add(Flatten())
#prosthetoume ta epipeda dence gia mlp
mlp.add(Dense(units=128, activation='relu'))
#kanoume BatchNormalization einai antistoixo tou drop out
#kanoume drop out gia kalutera apotelesmata
mlp.add(tf.keras.layers.Dropout(rate=0.5))
mlp.add(tf.keras.layers.BatchNormalization())
mlp.add(Dense(units=64, activation='relu'))
mlp.add(tf.keras.layers.BatchNormalization())
mlp.add(tf.keras.layers.Dropout(rate=0.5))

mlp.add(Dense(units=1, activation='sigmoid'))
#ginetai h sunthesh tou montelou
mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#auta den exoun sxesh me to mlp ta ebala epeidh xtupouse sto cmd
tf_compat.ragged.RaggedTensorValue
tf_compat.executing_eagerly_outside_functions
#ginetai h ekpaideush tou montelou
history =mlp.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1, validation_split=0.2)
print("Akribeia sta test")
print(mlp.evaluate(X_test, y_test))
#------------------------------Telos MLP---------------------------------------------------



# Εκπαίδευση του μοντέλου
#history = mlp.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

# Πάρτε τα δεδομένα από την εκπαίδευση
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Σχεδίαση της καμπύλης εκπαίδευσης
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, 'o-',color="b",label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'o-',color="green", label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#/////////////////////////////////////////////////////////////////////////////////
# gia na emfanistei h kampulh sfalmatos gia ta training kai validation paradeigmata
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Καμπύλες Σφάλματος κατά τη Διάρκεια των Εποχών')
plt.show()

#////////////////////////////precision,recall,f1/////////////////////////////////////
# kanoume problepsh gia ta dedomena ekpaideushs kai test
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)
# kanoume metatroph twn problepsewn se kathgories 0 h 1
y_train_pred_classes = (y_train_pred > 0.5).astype(int)
y_test_pred_classes = (y_test_pred > 0.5).astype(int)
#ektupwsh twn apotelesmatwn
print("Classification Report for Training Set:")
print(classification_report(y_train, y_train_pred_classes, zero_division=1))
print("\nClassification Report for Test Set:")
print(classification_report(y_test, y_test_pred_classes, zero_division=1))
#////////////////////////////////////////////////////////////////////////
#kwdikas gia na ektupwsoume pinaka 
# Ορίζουμε μια λίστα για τα μεγέθη του training set
training_sizes = [5000, 10000, 15000, 20000, 25000 ]
#dhmiourgoume enan keno pinaka gia na apothhkeusoume ta pososta orthothtas
results = []

for size in training_sizes:
    # pairnoume posa X_train kai y_train tha thelame na doume to pososto akribeias ston pinaka mas
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
    #kanoume problepsh gia ta dedomena ekpaideushs
    y_train_pred = mlp.predict(X_train)
    # kanoume problepsh gia ta dedomena elegxou
    y_test_pred = mlp.predict(X_test)
    # kanoume metatroph twn problepsewn se kathgories 0 h 1
    y_train_pred_classes = (y_train_pred > 0.5).astype(int)
    y_test_pred_classes = (y_test_pred > 0.5).astype(int)
    # kanoume upologismo akribeias sta dedomena ekpaideushs kai elegxou
    accuracy_train = accuracy_score(y_train, y_train_pred_classes)
    accuracy_test = accuracy_score(y_test, y_test_pred_classes)
    #kanoume apothhkeush twn dedomenwn sthn lista result
    results.append({'Training Size': size, 'Accuracy (Training)': accuracy_train, 'Accuracy (Test)': accuracy_test})
# dhmiourgoume data frame me ta apotelesmta
results_df = pd.DataFrame(results)
# ta emfanizoume se pinaka
print(results_df)