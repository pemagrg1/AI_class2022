import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups  # using the dataset from sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import SGDClassifier

# TAKING ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med'] categories out of 20 categories from fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

# TAKING THE TRAIN AND TEST DATA
twenty_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories)

# SEPARATING THE DATA AND LABELS
train, train_labels = twenty_train.data, twenty_train.target
test, test_labels = twenty_test.data, twenty_test.target

# CONVERT TEXT TO NUMBERS
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train)

# CONVERT THE DATA TO VECTORS using TFIDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# LOADING THE MACHINE LEARNING MODEL
MultinomialNB_model = MultinomialNB()
sgd_model = SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42, max_iter=5, tol=None)

# TRAINING THE MODEL WITH THE DATA AND LABELS
clf = MultinomialNB_model.fit(X_train_tfidf, train_labels)
clf2 = sgd_model.fit(X_train_tfidf, twenty_train.target)


# TESTING THE TEST DATA ON THE TRAINED MODEL
# convert the test data to vectors
X_new_counts = count_vect.transform(test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# Test the data using the trained model
y_pred = clf.predict(X_new_tfidf)
y_pred2 = clf2.predict(X_new_tfidf)

# TESTING THE ACCURACY of MultinomialNB_model
cm = confusion_matrix(test_labels, y_pred)
ac = accuracy_score(test_labels, y_pred)
print("===TESTING THE ACCURACY of MultinomialNB model===")
print("ACCURACY:", ac)
print("CONFUSION MATRIX:\n", cm)
print()


# TESTING THE ACCURACY of SGD model
cm = confusion_matrix(test_labels, y_pred2)
ac = accuracy_score(test_labels, y_pred2)
print("===TESTING THE ACCURACY of SGD model===")
print("ACCURACY:", ac)
print("CONFUSION MATRIX:\n", cm)
