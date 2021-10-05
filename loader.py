#%%
import utils
from preprocessing import preprocessor
import joblib
from matplotlib import pyplot as plt

#%%
X_train, y_train = utils.readTrafficSigns('./GTSRB','training', (32, 32))
print(len(y_train), len(X_train))
plt.imshow(X_train[42])
plt.show()

#%%
joblib.dump(X_train, 'train_images.joblib')
joblib.dump(y_train, 'train_labels.joblib')

#%%
X_test, y_test = utils.readTrafficSigns('./GTSRB','test', (32, 32))
print(len(y_test), len(X_test))
plt.imshow(X_test[42])
plt.show()

#%%
joblib.dump(X_test, 'test_images.joblib')
joblib.dump(y_test, 'test_labels.joblib')

#%%
processedImages = preprocessor(X_train)
joblib.dump(processedImages, 'procTrainImages.joblib')

# %%
processedTestImages = preprocessor(X_test)
joblib.dump(processedTestImages, 'procTestImages.joblib')
# %%
