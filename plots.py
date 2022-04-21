#%%
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from random import sample, seed


trainLabels = joblib.load('./data/train_labels.joblib')
testLabels = joblib.load('./data/test_labels.joblib')
procTrainImages = joblib.load('./data/procTrainImages.joblib')
TrainImages = joblib.load('./data/train_images.joblib')
#%%
sns.set_context('paper')
sns.barplot(x=np.unique(trainLabels), y=np.bincount(trainLabels), color='gray').set(xticklabels=[])
plt.tight_layout()
plt.savefig('class distributions.png')
plt.show()


#%%

seed(32)
   
concat = zip(TrainImages, procTrainImages)
selected = sample(list(concat),6) 

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
axs = axes.ravel()

for i,ax in enumerate(axs):
    if i > 5:
        ax.imshow(np.squeeze(selected[i-6][1]), cmap=plt.cm.gray)
        ax.axis('off')
    else:      
        ax.imshow(np.squeeze(selected[i][0]), cmap=plt.cm.gray) 
        ax.axis('off')

plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('before-after.png')

plt.show()


#%%
signs = []
with open('names.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()
# %%
