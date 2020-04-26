# Written by Amratya Saraswat
# Email - amratyasaraswat@gmail.com

#From the console, run the following
#pip install numpy
#pip install scipy
#pip install scikit-learn
#pip install matplotlib

# Import required packages here (after they are installed)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load data. csv file should be in the same folder as the notebook for this to work, otherwise
# give data path.
data = np.loadtxt("data.csv")

#shuffle the data and select training and test data
np.random.seed(100)
np.random.shuffle(data)

features = []
digits = []

for row in data:
    #import the data and select only the 1's and 5's
    if(row[0]==1 or row[0]==5):
        features.append(row[1:])
        digits.append(str(row[0]))

        
#Select the proportion of data to use for training. 
#Notice that we have set aside 80% of the data for testing
numTrain = int(len(features)*.2)

trainFeatures = features[:numTrain]
testFeatures = features[numTrain:]
trainDigits = digits[:numTrain]
testDigits = digits[numTrain:]


#Convert the 256D data (trainFeatures) to 2D data
#We need X and Y for plotting and simpleTrain for building the model.
#They contain the same points in a different arrangement

X = []
Y = []
simpleTrain = []

#Colors will be passed to the graphing library to color the points.
#1's are blue: "b" and 5's are red: "r"
colors = []
for index in range(len(trainFeatures)):
    #produce the 2D dataset for graphing/training and scale the data so it is in the [-1,1] square
    xNew = 2*np.average(trainFeatures[index])+.75 
    yNew = 3*np.var(trainFeatures[index])-1.5
    X.append(xNew)  
    Y.append(yNew)
    simpleTrain.append([xNew,yNew])
    #trainDigits will still be the value we try to classify. Here it is the string "1.0" or "5.0"
    if(trainDigits[index]=="1.0"):
        colors.append("b")
    else:
        colors.append("r")

####################################################################################

#! 1(a) Decision Tree classifier
cv_score = []
max_leaf_nodes = [5,10,15,20,30,40,50,75,100,200,500,1000,10000]

for x in (max_leaf_nodes):
    model = DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes = x)
    predict_score = cross_val_score(model, simpleTrain, trainDigits, cv= 10)
    cv_score.append(1 - predict_score.mean())

mp.title("Figure 4.1 CV Error for Decision Tree with different max_leaf_node")
mp.plot(max_leaf_nodes, cv_score)
mp.xscale("log")
mp.show()


#! 1(b) Decision Tree classifier

# ? Looking at the graph, it looks like cv error below max_leaf_nodes = 15 is underfit since 
# ? the CV error is too high for this trained model.


#! 1(c) Decision Tree classifier

# ? The optimal model with minimum cross validation error is with max_leaf_nodes = 40, 500 amd 10000 
# ? but with max_leaf_nodes = 500, 10000 the model gets overfit. Hence when max_leaf_nodes = 40 the models has lowest cross validation error

xPred = []
yPred = []
cPred = []

model = DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes = 40)
model = model.fit(simpleTrain, trainDigits) 

# Use input points to get predictions here
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(model.predict([[xP,yP]])=="1.0"):
            cPred.append("b")
        else:
            cPred.append("r")

#plot the points
mp.scatter(X,Y,s=3,c=colors)

#plot the regions
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.1)

#setup the axes
mp.xlim(-1,1)
mp.ylim(-1,1)
mp.title("Figure 4.2 Decision Region with minimum CV Error for DT")
show()

#! 1(e) Random Forest Classifier

rf_max_leaf_nodes = [10,100,1000]
rf_n_estimator = [5,10,15,20,30,40,50,75,100,200,500,1000,10000]

cvs_rf_first =[] 
cvs_rf_second = []
cvs_rf_third = []

#? Cross validation Error for Random Forest with max_leaf_node = 10 and different n_estimators
for x in (rf_n_estimator):
    rf_model = RandomForestClassifier(n_estimators = x, criterion = 'entropy', max_leaf_nodes = 10)
    predict_rf_score = cross_val_score(rf_model, simpleTrain, trainDigits, cv=10)
    cvs_rf_first.append(1 - predict_rf_score.mean())

mp.title("Figure 4.3 CV Error for Random Forest with max_leaf_node = 10")
mp.plot(rf_n_estimator, cvs_rf_first)
mp.xscale("log")
mp.show() 

#? Cross validation Error for Random Forest with max_leaf_node = 100 and different n_estimators
for x in (rf_n_estimator):
    rf_model = RandomForestClassifier(n_estimators = x, criterion = 'entropy', max_leaf_nodes = 100)
    predict_rf_score = cross_val_score(rf_model, simpleTrain, trainDigits, cv=10)
    cvs_rf_second.append(1 - predict_rf_score.mean())

mp.title("Figure 4.4 CVError for Random Forest with max_leaf_node = 100")
mp.plot(rf_n_estimator, cvs_rf_second)
mp.xscale("log")
mp.show() 

#? Cross validation Error for Random Forest with max_leaf_node = 1000 and different n_estimators
for x in (rf_n_estimator):
    rf_model = RandomForestClassifier(n_estimators = x, criterion = 'entropy', max_leaf_nodes = 1000)
    predict_rf_score = cross_val_score(rf_model, simpleTrain, trainDigits, cv=10)
    cvs_rf_third.append(1 - predict_rf_score.mean())

mp.title("Figure 4.5 CV Error for Random Forest with max_leaf_node = 1000")
mp.plot(rf_n_estimator, cvs_rf_third)
mp.xscale("log")
mp.show() 

# #! 1(f) Random Forest Classifier
# 

# #! 1(g) Random Forest Classifier
# On increasing the n_estimator it basically increases the tree in the forest which basically
# helps in making decision better but after a while increasing the n_estimator increases the 
# chances for the model to be overfit. So indentifying that sweet spot is really important 
# for the model to be accurate. 
# So in both case of max_leaf_node = 10 and max_leaf_nodes = 100, the point at which 
# increasing the estimator did not make any impact on the mode was n_estimator = 1000
#  While in the case of max_leaf_node = 1000, there was no sign of such point (till 10000)
#  according to the the figure 4.5 

#! 1(h)

xPred = []
yPred = []
cPred = []

rf_model_dr = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', max_leaf_nodes = 1000)
rf_model_dr = rf_model_dr.fit(simpleTrain, trainDigits) 

# Use input points to get predictions here
for xP in range(-100,100):
    xP = xP/50.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(rf_model_dr.predict([[xP,yP]])=="1.0"):
            cPred.append("b")
        else:
            cPred.append("r")

#plot the points
mp.scatter(X,Y,s=3,c=colors)

#plot the regions
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.1)

#setup the axes
mp.xlim(-1,1)
mp.ylim(-1,1)
mp.title("Figure 4.6 Decision Region with minimum CV Error for RF")
show()


#! 1(j) AdaBoost Classifier

adab_n_estimator = [1,5,10,100,1000,10000]
cvs_adab = []

for x in adab_n_estimator:
    adab_model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, criterion='entropy'), n_estimators = x)
    predict_adab_score = cross_val_score(adab_model, simpleTrain, trainDigits, cv=10)
    cvs_adab.append(1 - predict_adab_score.mean())

mp.title("Figure 4.7 CV Error for AdaBoost max depth =1")
mp.plot(adab_n_estimator, cvs_adab)
mp.xscale("log")
mp.show() 

#! 1(k) AdaBoost Classifier

cvs_adab2 = []
for x in adab_n_estimator:
    adab_model2 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10, criterion='entropy'), n_estimators = x)
    predict_adab_score2 = cross_val_score(adab_model2, simpleTrain, trainDigits, cv=10)
    cvs_adab2.append(1 - predict_adab_score2.mean())

mp.title("Figure 4.8 CV Error for AdaBoost with max depth =10")
mp.plot(adab_n_estimator, cvs_adab2)
mp.xscale("log")
mp.show() 

#! 1(l) AdaBoost Classifier

cvs_adab3 = []
for x in adab_n_estimator:
    adab_model3 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1000, criterion='entropy'), n_estimators = x)
    predict_adab_score3 = cross_val_score(adab_model3, simpleTrain, trainDigits, cv=10)
    cvs_adab3.append(1 - predict_adab_score3.mean())

mp.title("Figure 4.9 CV Error for AdaBoost with max depth =1000")
mp.plot(adab_n_estimator, cvs_adab3)
mp.xscale("log")
mp.show() 

#! 1(m) Decision region for AdaBoost Classifier

xPred = []
yPred = []
cPred = []

decision_model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, criterion='entropy'), n_estimators = 1000)
decision_model = decision_model.fit(simpleTrain, trainDigits) 

# Use input points to get predictions here
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(decision_model.predict([[xP,yP]])=="1.0"):
            cPred.append("b")
        else:
            cPred.append("r")

#plot the points
mp.scatter(X,Y,s=3,c=colors)

#plot the regions
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.1)

#setup the axes
mp.xlim(-1,1)
mp.ylim(-1,1)
mp.title("Figure 4.10 Decision Region for AdaBoost with min CV Error")
show()


