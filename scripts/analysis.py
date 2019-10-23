import time
import os
import shutil
from distutils.dir_util import copy_tree
import numpy as np
from itertools import cycle
from sklearn.utils.fixes import signature
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

x = np.genfromtxt('../results/predictions/all_truth_pred.csv', delimiter=',',usecols=(0,)).astype(int)
y = np.genfromtxt('../results/predictions/all_truth_pred.csv',delimiter=',',usecols=(1,)).astype(int)
timing = np.genfromtxt('../results/timing.csv',delimiter=',',usecols=(1,)).astype(float)

assert(max(x) == max(y))

localtime = time.asctime( time.localtime(time.time()) )
print localtime
if not os.path.exists(localtime):
    os.makedirs(localtime)
    
f = open(localtime + "/analysis.csv","w")
report = open(localtime + "/classification_report.txt","w")

shutil.copy('../results/timing.csv', localtime + "/timing.csv")

#Timings
print('\training_fex: {0}'.format(timing[0]))
print('\ntraining_time: {0}'.format(timing[1]))
print('\nprediction_fex: {0}'.format(timing[2]))
print('\nprediction_time: {0}'.format(timing[3]))

f.write("training_fex,%f\n" % timing[0])
f.write("training_time,%f\n" % timing[1])
f.write("prediction_fex,%f\n" % timing[2])
f.write("prediction_time,%f\n" % timing[3])

# Class recall, precision and f1-scores
classRecall = recall_score(x, y,average=None)
classPrecision = precision_score(x, y,average=None)
classF1Score = f1_score(x,y,average=None)

print('class recalls:\n{0}'.format(classRecall))
print('class precisions:\n{0}'.format(classPrecision))
print('class f1-scores:\n{0}'.format(classF1Score))

f.write("class recalls,")
for item in classRecall:
	f.write("%f," % item)
        	
f.write("\nclass precisions,")
for item in classPrecision:
	f.write("%f," % item)
	
f.write("\nf1-scores,")
for item in classF1Score:
	f.write("%f," % item)

# Mean recall, precision, f1-scores and accuracy
meanRecall = recall_score(x, y, average='macro')
meanPrecision = precision_score(x, y, average='macro')
meanF1Score= f1_score(x, y, average='macro')
meanAccuracy = accuracy_score(x, y, normalize = True)

print('\nmean recall: {0}'.format(meanRecall))
print('\nmean precision: {0}'.format(meanPrecision))
print('\nmean f1-score: {0}'.format(meanF1Score))
print('\noverall accuracy: {0}'.format(meanAccuracy))

f.write("\nmean recall,%f" % meanRecall)
f.write("\nmean precision,%f" % meanPrecision)
f.write("\nmean f1-score,%f" % meanF1Score)
f.write("\nmean accuracy,%f" % meanAccuracy)

#Confusion matrix
confusionMatrix = confusion_matrix(x, y)
print('\nconfusion matrix:\n{0}'.format(confusionMatrix))

f.write("\nconfusion matrix\n")
for row in confusionMatrix:
	for item in row:
		f.write("%f," % item)
	f.write("\n")
	
	
# Normalized confusion matrix
normalizedCM = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]

print('\nnormalized confusion matrix:\n{0}'.format(normalizedCM))

f.write("\nnormalized confusion matrix\n")
for row in normalizedCM:
	for item in row:
		f.write("%.4f," % item)
	f.write("\n")

#Classification report
classif_report = classification_report(x, y)
report.write(classif_report)


copy_tree("../results/predictions", localtime + "/predictions")


# Plot confusion matrices

def plot_confusion_matrix(cm, classes,title,cmap=plt.cm.Blues):

    """
    from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True class',
           xlabel='Predicted class')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    savefig(localtime + '/confusion.png', bbox_inches='tight')
    return ax


np.set_printoptions(precision=2)


# Class names, change depending on dataset
#class_names = ['keep', 'discard'] # montelupo / songomnara
#class_names = ['ground', 'building', 'vegetation', 'pole', 'wire'] # oakland
class_names = ['man-made terrain', 'natural terrain', 'high vegetation', 'low vegetation', 'buildings', 'hard scape', 'scanning artefacts', 'cars'] # semantic3d

# Plot normalized confusion matrix
plot_confusion_matrix(normalizedCM, classes=class_names,
                      title='Normalized confusion matrix')

plt.show()

