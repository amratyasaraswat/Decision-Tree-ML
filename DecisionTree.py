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

# 1(a) Decision Tree classifier
cv_score = []
max_leaf_nodes = [5,10,15,20,30,40,50,75,100,200,500,1000,10000]

for x in (max_leaf_nodes):
    model = DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes = x)
    predict_score = cross_val_score(model, simpleTrain, trainDigits, cv= 10)
    cv_score.append(1 - predict_score.mean())

mp.title("Figure 4.1 Cross validation Error for Decision Tree with different max_leaf_node")
mp.plot(max_leaf_nodes, cv_score)
mp.xscale("log")
mp.show()


# 1(b) Decision Tree classifier

# ?  Looking at the graph we can say that as the value of the max_leaf_node


# 1(c) Decision Tree classifier

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

## Visualize Results
#plot the points
mp.scatter(X,Y,s=3,c=colors)

#plot the regions
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.1)

#setup the axes
mp.xlim(-1,1)
mp.ylim(-1,1)
mp.title("Figure 4.2 Decision Tree with minimum Cross Validation Error")
show()

# 1(e) Random Forest Classifier

rf_max_leaf_nodes = [10,100,1000]
rf_n_estimator = [5,10,15,20,30,40,50,75,100,200,500,1000,10000]

cvs_rf_first =[] 
cvs_rf_second = []
cvs_rf_third = []

# ! Cross validation Error for Random Forest with max_leaf_node = 10 and different n_estimators
for x in (rf_n_estimator):
    rf_model = RandomForestClassifier(n_estimators = x, criterion = 'entropy', max_leaf_nodes = 10)
    predict_rf_score = cross_val_score(rf_model, simpleTrain, trainDigits, cv=10)
    cvs_rf_first.append(1 - predict_rf_score.mean())

mp.title("Figure 4.3 Cross validation Error for Random Forest with max_leaf_node = 10")
mp.plot(rf_n_estimator, cvs_rf_first)
mp.xscale("log")
mp.show() 

# ! Cross validation Error for Random Forest with max_leaf_node = 100 and different n_estimators
for x in (rf_n_estimator):
    rf_model = RandomForestClassifier(n_estimators = x, criterion = 'entropy', max_leaf_nodes = 100)
    predict_rf_score = cross_val_score(rf_model, simpleTrain, trainDigits, cv=10)
    cvs_rf_second.append(1 - predict_rf_score.mean())

mp.title("Figure 4.4 Cross validation Error for Random Forest with max_leaf_node = 100")
mp.plot(rf_n_estimator, cvs_rf_second)
mp.xscale("log")
mp.show() 

# ! Cross validation Error for Random Forest with max_leaf_node = 1000 and different n_estimators
for x in (rf_n_estimator):
    rf_model = RandomForestClassifier(n_estimators = x, criterion = 'entropy', max_leaf_nodes = 1000)
    predict_rf_score = cross_val_score(rf_model, simpleTrain, trainDigits)
    cvs_rf_third.append(1 - predict_rf_score.mean())

mp.title("Figure 4.5 Cross validation Error for Random Forest with max_leaf_node = 100")
mp.plot(rf_n_estimator, cvs_rf_third)
mp.xscale("log")
mp.show() 







#####################################################################################
# xPred = []
# yPred = []
# cPred = []
# # Use input points to get predictions here
# for xP in range(-100,100):
#     xP = xP/100.0
#     for yP in range(-100,100):
#         yP = yP/100.0
#         xPred.append(xP)
#         yPred.append(yP)
#         if(model.predict([[xP,yP]])=="1.0"):
#             cPred.append("b")
#         else:
#             cPred.append("r")

# ## Visualize Results
# #plot the points
# mp.scatter(X,Y,s=3,c=colors)

# #plot the regions
# mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.1)

# #setup the axes
# mp.xlim(-1,1)
# mp.xlabel("Average Intensity")
# mp.ylim(-1,1)
# mp.ylabel("Intensity Variance")
# mp.title("Figure 1.4 Decision Region for 256D")
# show()




