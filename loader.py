#%%
import utils
from preprocessing import preprocessor
import joblib
from matplotlib import pyplot as plt

#%%
#loading and saving original training images
X_train, y_train = utils.readTrafficSigns('./GTSRB','training', (32, 32))
print(len(y_train), len(X_train))
plt.imshow(X_train[42])
plt.show()

joblib.dump(X_train, './data/train_images.joblib')
joblib.dump(y_train, './data/train_labels.joblib')

#%%
#loading and saving original test images
X_test, y_test = utils.readTrafficSigns('./GTSRB','test', (32, 32))
print(len(y_test), len(X_test))
plt.imshow(X_test[42])
plt.show()

joblib.dump(X_test, './data/test_images.joblib')
joblib.dump(y_test, './data/test_labels.joblib')

#%%
#processing and saving train and test images
processedImages = preprocessor(X_train)
joblib.dump(processedImages, './data/procTrainImages.joblib')

processedTestImages = preprocessor(X_test)
joblib.dump(processedTestImages, './data/procTestImages.joblib')
# %%
