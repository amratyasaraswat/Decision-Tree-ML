import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC as SVC
from math import log, sqrt


data = np.loadtxt("data.csv")
np.random.seed(100)
np.random.shuffle(data)

features = []
digits = []

for row in data:
    if(row[0]==1 or row[0]==5):
        features.append(row[1:])
        digits.append(str(row[0]))

numTrain = int(len(features)*.2)

trainFeatures = features[:numTrain]
testFeatures = features[numTrain:]
trainDigits = digits[:numTrain]
testDigits = digits[numTrain:]

X = []
Y = []
simpleTrain = []

colors = []
for index in range(len(trainFeatures)):
    xNew = 2*np.average(trainFeatures[index])+.75 
    yNew = 3*np.var(trainFeatures[index])-1.5
    X.append(xNew)  
    Y.append(yNew)
    simpleTrain.append([xNew,yNew])
    if(trainDigits[index]=="1.0"):
        colors.append("b")
    else:
        colors.append("r")

# Markov function calculates the markov bound given confidence interval and epsilon
def markov(eTest, confidence):
    confidence = 2 * (1 - confidence)
    return 1 - (eTest / confidence)

def chebyshev(eTest, confidence):
   length = len(testDigits)
   delta = 1 - confidence
   return sqrt(1/(4 * length * delta))

def hoeffding(eTest, confidence):
    length = len(testDigits)
    delta = 1 - confidence
    return sqrt ((1/(2 * length) * log(2/delta)))


svc =  SVC = SVC(C = 10,  kernel= 'poly', degree = 5)
svc.fit (trainFeatures, trainDigits)
nn = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100), activation = "relu", epsilon = 0.001, max_iter= 10000, alpha = 0, solver = "adam")
nn.fit(trainFeatures, trainDigits)
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', max_leaf_nodes = 1000)
rf = rf.fit(trainFeatures, trainDigits) 
adab = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, criterion='entropy'), n_estimators = 1000)
adab = adab.fit(trainFeatures, trainDigits) 

# ! Finding the error with testing set 
svcError = 1 - svc.score(testFeatures, testDigits)
nnError = 1 - nn.score(testFeatures, testDigits)
rfError = 1 - rf.score(testFeatures, testDigits)
adabError = 1 - adab.score(testFeatures, testDigits)


print ("***SVM model*** ")
print ("Bounds for Markov with different confidence interval ")
print ("99 % - ", markov(svcError, 0.99))
print ("95 % - ", markov(svcError, 0.95))
print ("75 % - ", markov(svcError, 0.75))

print ("Bounds for Chebyshev with different confidence interval ")
print ("99 % - ", 1 - chebyshev(svcError, 0.99))
print ("95 % - ", 1 - chebyshev(svcError, 0.95))
print ("75 % - ", 1 - chebyshev(svcError, 0.75))

print ("Bounds for Hoeffding with different confidence interval ")
print ("99 % - ", 1 - hoeffding(svcError, 0.99))
print ("95 % - ", 1 - hoeffding(svcError, 0.95))
print ("75 % - ", 1 - hoeffding(svcError, 0.75))


print ("***Neural Network model*** ")
print ("Bounds for Markov with different confidence interval ")
print ("99 % - ", markov(nnError, 0.99))
print ("95 % - ", markov(nnError, 0.95))
print ("75 % - ", markov(nnError, 0.75))

print ("Bounds for Chebyshev with different confidence interval ")
print ("99 % - ", 1 - chebyshev(nnError, 0.99))
print ("95 % - ", 1 - chebyshev(nnError, 0.95))
print ("75 % - ", 1 - chebyshev(nnError, 0.75))

print ("Bounds for Hoeffding with different confidence interval ")
print ("99 % - ", 1 - hoeffding(nnError, 0.99))
print ("95 % - ", 1 - hoeffding(nnError, 0.95))
print ("75 % - ", 1 - hoeffding(nnError, 0.75))


print ("***Random Forest model*** ")
print ("Bounds for Markov with different confidence interval ")
print ("99 % - ", markov(rfError, 0.99))
print ("95 % - ", markov(rfError, 0.95))
print ("75 % - ", markov(rfError, 0.75))

print ("Bounds for Chebyshev with different confidence interval ")
print ("99 % - ", 1 - chebyshev(rfError, 0.99))
print ("95 % - ", 1 - chebyshev(rfError, 0.95))
print ("75 % - ", 1 - chebyshev(rfError, 0.75))

print ("Bounds for Hoeffding with different confidence interval ")
print ("99 % - ", 1 - hoeffding(rfError, 0.99))
print ("95 % - ", 1 - hoeffding(rfError, 0.95))
print ("75 % - ", 1 - hoeffding(rfError, 0.75))


print ("***Adaboost model*** ")
print ("Bounds for Markov with different confidence interval ")
print ("99 % - ", markov(adabError, 0.99))
print ("95 % - ", markov(adabError, 0.95))
print ("75 % - ", markov(adabError, 0.75))

print ("Bounds for Chebyshev with different confidence interval ")
print ("99 % - ", 1 - chebyshev(adabError, 0.99))
print ("95 % - ", 1 - chebyshev(adabError, 0.95))
print ("75 % - ", 1 - chebyshev(adabError, 0.75))

print ("Bounds for Hoeffding with different confidence interval ")
print ("99 % - ", 1 - hoeffding(adabError, 0.99))
print ("95 % - ", 1 - hoeffding(adabError, 0.95))
print ("75 % - ", 1 - hoeffding(adabError, 0.75))





