import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import gini_index, t_score, CFS
from scipy.stats import pearsonr
from gini import *

# Number of features to select
num_fea = 25

# Store all feature data from CSV
X = np.genfromtxt('../results/features.csv', delimiter=',',usecols=range(0,72)).astype(float)
y = np.genfromtxt('../results/features.csv',delimiter=',',usecols=(72,)).astype(int)


# ------------------------ Gini Co-efficient ------------------------
print "Gini Co-efficient:"

scores1 = np.array([])
for x in range(X.shape[1]):
	gini_index = gini(X[:, [x]].flatten())
	scores1 = np.append(scores1,gini_index)

g1 = lambda e: e[1]
g10 = lambda e: e[1][0]
# don't change scores sign for sorting here (lower gini index = better)
R1, _ = zip(*sorted(enumerate(sorted(enumerate(scores1), key=g1)), key=g10)) 

#print scores1
formatted_scores1 = [ '%.2f' % elem for elem in scores1]
print formatted_scores1
print R1

# ------------------------ ANOVA F-value------------------------
print "ANOVA F-value:"
method = f_classif
test2 = SelectKBest(score_func=method, k=num_fea)
fit = test2.fit(X, y)
scores2 = test2.scores_

g1 = lambda e: e[1]
g10 = lambda e: e[1][0]
R2, _ = zip(*sorted(enumerate(sorted(enumerate(-scores2), key=g1)), key=g10))

#print scores2
formatted_scores2 = [ '%.2f' % elem for elem in scores2]
print formatted_scores2
print R2

# ------------------------ Mutual Information ------------------------
print "Mutual Information:"
method = mutual_info_classif
test3 = SelectKBest(score_func=method, k=num_fea)
fit = test3.fit(X, y)
scores3 = test3.scores_

g1 = lambda e: e[1]
g10 = lambda e: e[1][0]
R3, _ = zip(*sorted(enumerate(sorted(enumerate(-scores3), key=g1)), key=g10))

#print scores3
formatted_scores3 = [ '%.2f' % elem for elem in scores3]
print formatted_scores3
print R3


# ------------------------ Pearson Correlation ------------------------
print "Pearson Correlation:"
scores4 = np.array([])
for x in range(X.shape[1]):
	pearsonscore = pearsonr(X[:, [x]].flatten(), y.flatten())
	scores4 = np.append(scores4,abs(pearsonscore[0])) #absolute value because -1 or +1 represent perfect correlation

g1 = lambda e: e[1]
g10 = lambda e: e[1][0]
R4, _ = zip(*sorted(enumerate(sorted(enumerate(-scores4), key=g1)), key=g10))

#print scores4
formatted_scores4 = [ '%.2f' % elem for elem in scores4]
print formatted_scores4
print R4

# ------------------------ Fisher Score ------------------------
print "Fisher Score:"
scores5 = fisher_score.fisher_score(X, y)

g1 = lambda e: e[1]
g10 = lambda e: e[1][0]
R5, _ = zip(*sorted(enumerate(sorted(enumerate(-scores5), key=g1)), key=g10))

#print scores5
formatted_scores5 = [ '%.2f' % elem for elem in scores5]
print formatted_scores5
print R5


# ------------------------ Relief-F ------------------------
print "Relief-F:"
scores6 = reliefF.reliefF(X, y)

g1 = lambda e: e[1]
g10 = lambda e: e[1][0]
R6, _ = zip(*sorted(enumerate(sorted(enumerate(-scores6), key=g1)), key=g10))

#print scores6
formatted_scores6 = [ '%.2f' % elem for elem in scores6]
print formatted_scores6
print R6

# ------------------------ Final Ranking Calculation  ------------------------
finalRanks = R1
finalRanks = np.add(finalRanks, R2)
finalRanks = np.add(finalRanks, R3)
finalRanks = np.add(finalRanks, R4)
finalRanks = np.add(finalRanks, R5)
finalRanks = np.add(finalRanks, R6)

print finalRanks

selectedFeat = np.zeros(X.shape[1])
bestIdx = np.argsort(finalRanks)

for x in range(num_fea):
	selectedFeat[bestIdx[x]] = 1
	
print selectedFeat

# Save To Disk
with open("../results/filteredFeat.txt", 'w+') as f:
    for item in selectedFeat:
        f.write("%s\n" % item.astype(int))
        
# Save To Disk
with open("../results/filteredRanks.txt", 'w+') as f:
    for item in finalRanks:
        f.write("%s\n" % item.astype(int))
