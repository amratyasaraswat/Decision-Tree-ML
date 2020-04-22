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

#plot the data points
## https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
# mp.scatter(X,Y,s=3,c=colors)

#specify the axes
# mp.xlim(-1,1)
# mp.xlabel("Average Intensity")
# mp.ylim(-1,1)
# mp.ylabel("Intensity Variance")
# mp.title("Figure 1.1")
# #display the current graph
# show()
#create the model
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# Lists to hold inpoints, predictions and assigned colors
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

## Visualize Results
#plot the points

# mp.scatter(X,Y,s=3,c=colors)

# #plot the regions
# mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.1)

# #setup the axes
# mp.xlim(-1,1)
# mp.xlabel("Average Intensity")
# mp.ylim(-1,1)
# mp.ylabel("Intensity Variance")
# mp.title("Figure 1.2 Decision Region for 2D")
# show()



# oddKValaues = list(range(1,50,2))

# cvScores256d = []
# colorList = []

# cvstd = []

# for k in oddKValaues:
#     model2 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#     model2.fit(trainFeatures, trainDigits)
#     predicted256d = cross_val_score(model2, trainFeatures, trainDigits, cv=10)
#     cvScores256d.append (predicted256d.mean()*100)
#     cvstd.append(196*np.std (predicted256d))
#     colorList.append("r")

# for k in oddKValaues:
#     model0 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#     model0.fit(simpleTrain, trainDigits)
#     predicted2d = cross_val_score(model0, simpleTrain, trainDigits, cv=10)
#     cvScores256d.append (predicted2d.mean()*100)
#     cvstd.append(196*np.std (predicted2d))
#     colorList.append("b")

# zeroes = [0]*25

# csError256d = [100 - x for x in cvScores256d]

# mp.errorbar(oddKValaues, csError256d[25:], yerr = [tuple(zeroes), tuple(cvstd[25:])], ls='', )
# mp.errorbar(oddKValaues, csError256d[:25], yerr = [tuple(zeroes), tuple(cvstd[:25])], ls='', )
# mp.scatter(oddKValaues*2 , csError256d, s=3, c = colorList)
# mp.xlabel("K values")
# mp.ylabel("Cross Validation Error")
# mp.title("Odd K value Cross Validation Graph")
# mp.title("Figure 1.3")
# mp.show()

####################################################################################

print ("The length of simple ----", len(simpleTrain))

# 1(a) Decision Tree classifier
cv_score = []
max_leaf_nodes = [5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000, 10000]
ax = list(range(1, len(max_leaf_nodes)+1, 1))

for x in (max_leaf_nodes):
    model = DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes = x)
    #model = model.fit(simpleTrain, trainDigits) 
    predict_score = cross_val_score(model, simpleTrain, trainDigits)
    cv_score.append(1 - predict_score.mean())

print (cv_score)

mp.title("The CV Score with different max_leaf_node")
mp.plot(ax, cv_score)
mp.xscale("log")
mp.show()


# 1(b) Decision Tree classifier








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




