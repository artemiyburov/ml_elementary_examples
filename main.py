from classification import SGClassifier
import numpy as np
from sklearn import datasets

X, y = datasets.make_classification(n_samples=10000)
y = y*2-1
clf = SGClassifier(.1,.1)
clf.fit(X[:-10],y[:-10],15)
#clf.fit(X[:-1],y[:-1],1000)
test_classes = clf.predict(X[-10:])
'''
if test_classes == y[-1:]:
	print('lesson taught')
else:
	print('bad teacher')
'''
print(test_classes)
print(y[-10:])
