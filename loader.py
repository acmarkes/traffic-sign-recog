#%%
import utils
from preprocessing import preprocessor
import pathlib
import joblib
from matplotlib import pyplot as plt

#%%
data_dir = './GTSRB/Training/Images' 
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.ppm'))) #number of images in dataset
print(image_count)

trainImages, trainLabels = utils.readTrafficSigns('./GTSRB/Training/Images', (32, 32))
print(len(trainLabels), len(trainImages))
plt.imshow(trainImages[42])
plt.show()

joblib.dump(trainImages, 'images.joblib')
joblib.dump(trainLabels, 'labels.joblib')

#%%
processedImages = preprocessor(trainImages)
joblib.dump(processedImages, 'procImages.joblib')

