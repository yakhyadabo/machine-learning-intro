import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Metadata
print "Features'names : ",  iris.feature_names
print "Labels'names : ",  iris.target_names

#Data of features[0] and Labels[0]
print "Features[0]",  iris.data[0]
print "Labels[0]",  iris.target[0]


# for i in range(len(iris.target)): 
#     print "Exemple %d: label %s, feature %s" %(i, iris.target[i], iris.data[i])


# Testing data
test_idx = [0,50,100]
test_target = iris.target[test_idx] # Add iris.target[0], iris.target[50],iris.target[100],
test_data = iris.data[test_idx] # Add iris.data[0], iris.data[50],iris.data[100],

# Train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# Predict 
print "Features : %s => Label %s" %(test_data, clf.predict(test_data))

# Results : 
# Features : 
#     [ 5.1  3.5  1.4  0.2] => 0
#     [ 7.   3.2  4.7  1.4] => 1
#     [ 6.3  3.3  6.   2.5] => 2

