# AI_Search_And_ML_Classifier_Comparison
This repository contains the implementation of a university project in Artificial Intelligence, divided into two distinct parts:

# Part A – Bridge Crossing Problem
Implementation of the A* heuristic search algorithm to solve the well-known Bridge Crossing problem. The goal is to find an optimal strategy for a family to cross a bridge under specific time constraints.

# Problem Description
A group of family members must cross a bridge at night. The following constraints apply:

- The bridge can support a maximum of 2 people at a time.

- A torch is required to cross and must be brought back if anyone remains on the starting side.

- Each member has a different crossing time.

- When two people cross together, the time taken is equal to the slower person's time.

 #  Method
- State representation using the State class.

- Heuristic computation based on crossing times.

- Cost-based search using the formula f = g + h.

- The goal state is when all members have crossed to the other side.
  
# Code Structure
- FamMember.java – Class representing a family member.

- State.java – Class representing a state in the search space.

- mainClass.java – Executable program with user input support.

#  Technologies
- Language: Java

- Input via terminal

- Uses data structures like ArrayList and HashMap

 # Execution

- javac mainClass.java
- java mainClass

 # Part B – Comparison of Machine Learning Algorithms
The second part off this project compares custom implementations of Naive Bayes, Random Forest, and MLP with their counterparts from Python libraries (scikit-learn, tensorflow) on the IMDB movie reviews dataset. The goal is to evaluate performance metrics (accuracy, precision, recall, F1-score) and analyze learning curves.

# Functionality:
- Preprocessing of IMDB reviews from TensorFlow datasets.

- Feature extraction using bag-of-words and TF-IDF.

- Training and evaluation of all models.

- Metrics: Accuracy, Precision, Recall, F1 Score.

- Visualizations: Learning curves, Precision-Recall curves.

  # Code Structure
- MyBayes.py – Our custom implementation of Bernoulli Naive Bayes with Laplace smoothing.
- Key Features:
 - Handles text classification using binary word features.

 - Includes learning curves, precision-recall analysis, and accuracy tables.

- mainMYRandomForest.py – Our custom Random Forest implementation using ID3 decision trees.
- Key Features:

 - 20 decision trees with max depth=4.

 - Uses Information Gain for splits.

 - Parallel training with joblib.

 - Includes learning curves, precision-recall curves, and accuracy tables.

- mainMLP.py – Our custom Multi-Layer Perceptron (MLP) for text classification.
- Architecture:

 - Embedding layer (100-dim) + Flatten.

 - Dense layers (128, 64 units) with Dropout (0.5) and BatchNorm.

 - Trained for 15 epochs on 25,000 samples.

   2. Library Implementations (For Comparison)
- Bayes.py: Uses scikit-learn's BernoulliNB for Naive Bayes (Benchmark against our custom Naive Bayes implementation).

- mainRandomForest.py: Uses scikit-learn's RandomForestClassifier (Benchmark against our custom Random Fores)


# Technologies Used:
- Python 3

- TensorFlow / Keras

- NumPy, pandas, matplotlib, scikit-learn
