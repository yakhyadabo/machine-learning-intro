from sklearn import tree

# features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170,"bumpy"]] 
features = [[140, 1], [130, 1], [150, 0], [170,0]] 
labels = [0,0,1,1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)
# fit : Find Pattern in Data

print classifier.predict([[150,0]])

