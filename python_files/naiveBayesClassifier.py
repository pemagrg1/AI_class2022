import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix

le = LabelEncoder()
sc = StandardScaler()

# LOAD THE CSV TO PANDAS
dataset = pd.read_csv(
    '/Volumes/MacPema/githubsProj/AI_class2022/datasets/Social_Network_Ads.csv')

# SEPARATE THE DATASET LABEL COLUMN FROM THE DATASET
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

# LABEL ENCODE THE TEXT DATA TO INTEGERS AS COMPUTERS DONT UNDERSTAND TEXT DATA
X[:, 0] = le.fit_transform(X[:, 0])

# AFTER ALL THE CONVERSION IS DONE, DIVIDE THE DATASET TO TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# NORMALIZE THE DATA USING STANDARD SCALER
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LOADING THE MACHINE LEARNING MODEL
classifier = GaussianNB()

# TRAINING THE MODEL WITH THE DATA AND LABELS
classifier.fit(X_train, y_train)

# TESTING THE TEST DATA ON THE TRAINED MODEL
y_pred = classifier.predict(X_test)

# TESTING THE ACCURACY
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print("ACCURACY:", ac)
print()
print("CONFUSION MATRIX:\n", cm)

# VISUALIZING THE CONFUSION MATRIX
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()
