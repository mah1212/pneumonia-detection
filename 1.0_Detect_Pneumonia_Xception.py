#Import some useful libraries
import numpy as np #array handling and linear algebra
import matplotlib.pyplot as plt #plotting lib


#Data loading
from PIL import Image
import cv2 as cv
import os
from pathlib import Path
import glob

import pandas as pd

#Machine learning framework
# =============================================================================
# import keras
# from sklearn.model_selection import train_test_split
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, AveragePooling2D, Dropout, Input
# from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import concatenate
# from keras import Model, Sequential
# 
# =============================================================================
import keras
from keras.models import Sequential 

# Convolution is sequential
# Since we are going to work on images which are 2 dimentional unlike videos (time)
# API update use Conv2D instead of Convolution2D
from keras.layers import Conv2D 

# Step 2 Pooling step
from keras.layers import MaxPooling2D

# Avoid overfitting, import Dropout
from keras.layers import Dropout

# Step 3 Flatten
from keras.layers import Flatten

# Add fully connected layers in an ANN
from keras.layers import Dense

print("Importing finished!")


''' We are inside cell_images folder'''
# Load Data
print(os.getcwd())

print(os.listdir('..\\chest_xray'))
# print(os.listdir("../cell_images")) for linux

train_path = os.listdir("..\\chest_xray\\train")
print(os.listdir("..\\chest_xray\\train"))
train_normal = os.listdir("..\\chest_xray\\train\\NORMAL")
print(len(train_normal)) # no. of images = 1342

train_path = os.listdir("..\\chest_xray\\train")
print(os.listdir("..\\chest_xray\\train"))
train_pneumonia = os.listdir("..\\chest_xray\\train\\PNEUMONIA")
print(len(train_pneumonia)) # no. of images = 3876



test_path = os.listdir("..\\chest_xray\\test")
print(os.listdir("..\\chest_xray\\test"))
test_normal = os.listdir("..\\chest_xray\\test\\NORMAL")
print(len(test_normal)) # no. of images = 1342

test_path = os.listdir("..\\chest_xray\\test")
print(os.listdir("..\\chest_xray\\test"))
test_pneumonia = os.listdir("..\\chest_xray\\test\\PNEUMONIA")
print(len(test_pneumonia)) # no. of images = 3876

print(test_normal[1]) # show 1st image label
print(test_pneumonia[1])



''' File path reading technique NO MORE OS.Path '''
data_dir = Path('..\\chest_xray\\')
print(data_dir)

train_dir = data_dir / 'train'
print(train_dir)

test_dir = data_dir / 'test'
print(test_dir)

val_dir = data_dir / 'val'
print(val_dir)


''' Work on Train Dataset '''
normal_dir = train_dir / 'NORMAL'
print(normal_dir)

pneumonia_dir = train_dir / 'PNEUMONIA'
print(pneumonia_dir)


# Get the list of all images
normal_img_list_obj = normal_dir.glob('*.jpeg')
print(normal_img_list_obj)
#print(len(normal_img_list)) len does not work on object like this

pneumonia_img_list_obj = pneumonia_dir.glob('*.jpeg')
print(pneumonia_img_list_obj)

# take an empty list. We will fill this list with (image_path, label)
train_data = []


# Go through all the normal images. The label for normal is 0
for img in normal_img_list_obj:
    train_data.append((img, 0))

print(train_data[1])
print(len(train_data))

# Go through all the pneumonia images. The label for pneumonia is 1
for img in pneumonia_img_list_obj:
    train_data.append((img, 1))

print(train_data[1342])
print(len(train_data))


# create a pandas dataframe
train_df = pd.DataFrame(train_data, columns = ['image', 'label'], index = None)

# View some normal 
train_df.head()

# View some pneumonia
train_df.tail()

# Shuffle the data

''' How to shuffle pandas dataframe?
The more idiomatic way to do this with pandas is to use the .sample method of 
your dataframe, i.e.

df.sample(frac=1)

The frac keyword argument specifies the fraction of rows to return in the random sample, 
so frac=1 means return all rows (in random order).

Note: If you wish to shuffle your dataframe in-place and reset the index, 
you could do e.g.

df = df.sample(frac=1).reset_index(drop=True)

Here, specifying drop=True prevents .reset_index from creating a column containing 
the old index entries.
'''
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.head()
train_df.tail()


''' How many samples for each class are there? '''
class_count = train_df['label'].value_counts()
print(class_count)


import seaborn as sns

plt.figure(figsize=(10,8))
sns.barplot(x = class_count.index, y = class_count.values)
plt.title('Normal vs Pneumonia')
plt.xlabel('Class type')
plt.ylabel('Count')
plt.xticks(range(len(class_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()


''' Is Data balanced? Imbalanced? '''
# We can see from the plot that data is highly imbalanced

# plot images 
# Get samples from both classes, take 5 samples from each, total 10 samples
normal_samples = train_df[train_df['label']==0]['image'].iloc[:5].tolist()
pneumonia_samples = train_df[train_df['label']==1]['image'].iloc[:5].tolist()

# Concatenate the samples in a single list and delete the above two lists
samples = normal_samples + pneumonia_samples 
del normal_samples, pneumonia_samples

# plot the samples
from skimage.io import imread

print(1//5)
print(1%5)

f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Normal")
    else:
        ax[i//5, i%5].set_title("Pneumonia")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()



''' Prepare validation data '''
#. read directory
normal_dir = val_dir / 'NORMAL'
print(normal_dir)

pneumonia_dir = val_dir / 'PNEUMONIA'
print(pneumonia_dir)

# get the list
normal_img_list = normal_dir.glob('*.jpeg')
pneumonia_img_list = pneumonia_dir.glob('*.jpeg')

valid_data = []
valid_labels = []

# Some images are in grayscale while majority of them contains 3 channels. 
# So, if the image is grayscale, we will convert into a image with 3 channels.
# We will normalize the pixel values and resizing all the images to 128x128
from keras.utils import to_categorical

for img in normal_img_list:
    img = cv.imread(str(img))
    img = cv.resize(img, (128, 128))
    
    if img.shape[2] == 1:
        img = np.dstack[img, img, img]
        
    # Convert to rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Scale
    img = img.astype(np.float32)/255.0
    
    # label
    label = to_categorical(0, num_classes = 2)
    
    # append
    valid_data.append(img)
    valid_labels.append(label)
    
    
    
for img in pneumonia_img_list:    
    img = cv.imread(str(img))
    img = cv.resize(img, (128, 128))
    
    if img.shape[2] == 1:
        img = np.dstack[img, img, img]
        
    # Convert to rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Scale
    img = img.astype(np.float32)/255.0
    
    # label
    label = to_categorical(1, num_classes = 2)
    
    # append
    valid_data.append(img)
    valid_labels.append(label)

    
# Convert the image list into numpy array
valid_data = np.array(valid_data)    
valid_labels = np.array(valid_labels)

print('Valid Data Shape: ', valid_data.shape)
print('Valid Labels Shape: ', valid_labels.shape)
    
''' How to convert grayscale image to colored image? 
2. Why use np.dstack? 
3. Why convert to numpy array later?
4. How to check all the image channels at once? 
meaning if they are gray or colored?


'''


''' Create a data generator '''

def data_generator(data, batch_size):
    
    # Get total number of samples in the data
    n = len(data)
    steps = n/batch_size
    
    # Define 2 numpy arrays for containing batch data and batch labels
    batch_data = np.zeros((batch_size, 128, 128, 3), dtype = np.float32)
    batch_lables = np.zeros((batch_size, 2), dtype = np.float32)
    
    # Get a numpy array for all the indices of the input data
    indices = np.arrange(n)
    
    
    i = 0
    while True:
        
        # shuffle indices
        np.random.shuffle(indices)
        
        # Get the next batch
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        
        count = 0
        for j, idx in enumerate(next_batch):
            img_path = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes = 2)
            
            
            # read the image and resize
            img = cv.imread(img_path)
            img = cv.resize(img, (128, 128))
            
            
            # check if it's a gray scale
            if img.shape[2] == 1:
                img = np.dstack(img, img, img)
                
            
            # convert BGR to RGB
            img_original = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            
            # normalize / scale the image
            img_original = img.astype(np.float32)/255.0
            
            
            # Store in the batch
            batch_data[count] = img_original
            batch_labels[count] = encoded_label
            
            
        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0
            
            
            
            
# Create depthwise xception convolution neural network
# following this paper https://arxiv.org/abs/1610.02357
# keras link: https://keras.io/applications/#xception            

from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.models import Model

def build_model():
    input_img = Input(shape=(128, 128, 3), name='InputImage')
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='Pool1')(x)
            

    x = SeparableConv2D(128, (3, 3,), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3, 3,), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='Pool2')(x)    

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)


    x = Dropout(0.2, name='dropout1')(x)
    x = Flatten(name='flatten')(x)
    
    x = Dense(units = 128, activation = "relu", name='Out1')(x)
    x = Dense(units = 2, activation = "softmax", name='Out2')(x)
    
    model = Model(inputs = input_img, outputs = x)
    return model



model = build_model()
model.summary()


'''
Class SeparableConv2D
Aliases:

    Class tf.keras.layers.SeparableConv2D
    Class tf.keras.layers.SeparableConvolution2D

Defined in tensorflow/python/keras/layers/convolutional.py.

Depthwise separable 2D convolution.

Separable convolutions consist in first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.

Intuitively, separable convolutions can be understood as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
Arguments:

    filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding: one of "valid" or "same" (case-insensitive).
    data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
    depth_multiplier: The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix.
    pointwise_initializer: Initializer for the pointwise kernel matrix.
    bias_initializer: Initializer for the bias vector.
    depthwise_regularizer: Regularizer function applied to the depthwise kernel matrix.
    pointwise_regularizer: Regularizer function applied to the pointwise kernel matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
    depthwise_constraint: Constraint function applied to the depthwise kernel matrix.
    pointwise_constraint: Constraint function applied to the pointwise kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.

Input shape: 4D tensor with shape: (batch, channels, rows, cols) if data_format='channels_first' or 4D tensor with shape: (batch, rows, cols, channels) if data_format='channels_last'.

Output shape: 4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format='channels_first' or 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.
__init__

__init__(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    pointwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    **kwargs
)

'''
'''
https://arxiv.org/abs/1610.02357
Convolutional neural networks have emerged as the master
algorithm in computer vision in recent years, and developing
recipes for designing them has been a subject of
considerable attention. The history of convolutional neural
network design started with LeNet-style models [10], which
were simple stacks of convolutions for feature extraction
and max-pooling operations for spatial sub-sampling. In
2012, these ideas were refined into the AlexNet architecture
[9], where convolution operations were being repeated
multiple times in-between max-pooling operations, allowing
the network to learn richer features at every spatial scale.
What followed was a trend to make this style of network
increasingly deeper, mostly driven by the yearly ILSVRC
competition; first with Zeiler and Fergus in 2013 [25] and
then with the VGG architecture in 2014 [18].
At this point a new style of network emerged, the Inception
architecture, introduced by Szegedy et al. in 2014 [20]
as GoogLeNet (Inception V1), later refined as Inception V2
[7], Inception V3 [21], and most recently Inception-ResNet
[19]. Inception itself was inspired by the earlier Network-
In-Network architecture [11]. Since its first introduction,
Inception has been one of the best performing family of
models on the ImageNet dataset [14], as well as internal
datasets in use at Google, in particular JFT [5].
The fundamental building block of Inception-style models
is the Inception module, of which several different versions
exist. In figure 1 we show the canonical form of an
Inception module, as found in the Inception V3 architecture.
An Inception model can be understood as a stack of
such modules. This is a departure from earlier VGG-style
networks which were stacks of simple convolution layers.
While Inception modules are conceptually similar to convolutions
(they are convolutional feature extractors), they
empirically appear to be capable of learning richer representations
with less parameters. How do they work, and
how do they differ from regular convolutions? What design
strategies come after Inception?


A complete description of the specifications of the network
is given in figure 5. The Xception architecture has
36 convolutional layers forming the feature extraction base
of the network. In our experimental evaluation we will exclusively
investigate image classification and therefore our
convolutional base will be followed by a logistic regression
layer. Optionally one may insert fully-connected layers before
the logistic regression layer, which is explored in the
experimental evaluation section (in particular, see figures
7 and 8). The 36 convolutional layers are structured into
14 modules, all of which have linear residual connections
around them, except for the first and last modules.

4.1. The JFT dataset
JFT is an internal Google dataset for large-scale image
classification dataset, first introduced by Hinton et al. in [5],
which comprises over 350 million high-resolution images
annotated with labels from a set of 17,000 classes. To evaluate
the performance of a model trained on JFT, we use an
auxiliary dataset, FastEval14k.
FastEval14k is a dataset of 14,000 images with dense
annotations from about 6,000 classes (36.5 labels per image
on average). On this dataset we evaluate performance
using Mean Average Precision for top 100 predictions
(MAP@100), and we weight the contribution of each class
to MAP@100 with a score estimating how common (and
therefore important) the class is among social media images.
This evaluation procedure is meant to capture performance
on frequently occurring labels from social media, which is
crucial for production models at Google.

4.2. Optimization configuration
A different optimization configuration was used for ImageNet
and JFT:
 On ImageNet:
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.045
– Learning rate decay: decay of rate 0.94 every 2
epochs
 On JFT:
– Optimizer: RMSprop [22]
– Momentum: 0.9
– Initial learning rate: 0.001
– Learning rate decay: decay of rate 0.9 every
- 3,000,000 samples


4.3. Regularization configuration
 Weight decay: The Inception V3 model uses a weight
decay (L2 regularization) rate of 4e 􀀀 5, which has
been carefully tuned for performance on ImageNet. We
found this rate to be quite suboptimal for Xception
and instead settled for 1e 􀀀 5. We did not perform
an extensive search for the optimal weight decay rate.
The same weight decay rates were used both for the
ImageNet experiments and the JFT experiments.
 Dropout: For the ImageNet experiments, both models
include a dropout layer of rate 0.5 before the logistic
regression layer. For the JFT experiments, no dropout
was included due to the large size of the dataset which
made overfitting unlikely in any reasonable amount of
time.
 Auxiliary loss tower: The Inception V3 architecture
may optionally include an auxiliary tower which backpropagates
the classification loss earlier in the network,
serving as an additional regularization mechanism. For
simplicity, we choose not to include this auxiliary tower
in any of our models.


4.4. Training infrastructure
All networks were implemented using the TensorFlow
framework [1] and trained on 60 NVIDIA K80 GPUs each.
For the ImageNet experiments, we used data parallelism
with synchronous gradient descent to achieve the best classification
performance, while for JFT we used asynchronous
gradient descent so as to speed up training. The ImageNet
experiments took approximately 3 days each, while the JFT
experiments took over one month each. The JFT models
were not trained to full convergence, which would have
taken over three month per experiment.

Top-1 accuracy Top-5 accuracy
VGG-16 0.715 0.901
ResNet-152 0.770 0.933
Inception V3 0.782 0.941
Xception 0.790 0.945



Documentation for individual models
Model 	Size 	Top-1 Accuracy 	Top-5 Accuracy 	Parameters 	Depth
Xception 	88 MB 	0.790 	0.945 	22,910,480 	126
VGG16 	528 MB 	0.713 	0.901 	138,357,544 	23
VGG19 	549 MB 	0.713 	0.900 	143,667,240 	26
ResNet50 	98 MB 	0.749 	0.921 	25,636,712 	-
ResNet101 	171 MB 	0.764 	0.928 	44,707,176 	-
ResNet152 	232 MB 	0.766 	0.931 	60,419,944 	-
ResNet50V2 	98 MB 	0.760 	0.930 	25,613,800 	-
ResNet101V2 	171 MB 	0.772 	0.938 	44,675,560 	-
ResNet152V2 	232 MB 	0.780 	0.942 	60,380,648 	-
ResNeXt50 	96 MB 	0.777 	0.938 	25,097,128 	-
ResNeXt101 	170 MB 	0.787 	0.943 	44,315,560 	-
InceptionV3 	92 MB 	0.779 	0.937 	23,851,784 	159
InceptionResNetV2 	215 MB 	0.803 	0.953 	55,873,736 	572
MobileNet 	16 MB 	0.704 	0.895 	4,253,864 	88
MobileNetV2 	14 MB 	0.713 	0.901 	3,538,984 	88
DenseNet121 	33 MB 	0.750 	0.923 	8,062,504 	121
DenseNet169 	57 MB 	0.762 	0.932 	14,307,880 	169
DenseNet201 	80 MB 	0.773 	0.936 	20,242,984 	201
NASNetMobile 	23 MB 	0.744 	0.919 	5,326,716 	-
NASNetLarge 	343 MB 	0.825 	0.960 	88,949,818 	-

'''            
            
'''    



It's easier to understand what np.vstack, np.hstack and np.dstack* do by looking at the .shape attribute of the output array.

Using your two example arrays:

print(a.shape, b.shape)
# (3, 2) (3, 2)

    np.vstack concatenates along the first dimension...

    print(np.vstack((a, b)).shape)
    # (6, 2)

    np.hstack concatenates along the second dimension...

    print(np.hstack((a, b)).shape)
    # (3, 4)

    and np.dstack concatenates along the third dimension.

    print(np.dstack((a, b)).shape)
    # (3, 2, 2)

Since a and b are both two dimensional, np.dstack expands them by inserting a third dimension of size 1. This is equivalent to indexing them in the third dimension with np.newaxis (or alternatively, None) like this:

print(a[:, :, np.newaxis].shape)
# (3, 2, 1)

If c = np.dstack((a, b)), then c[:, :, 0] == a and c[:, :, 1] == b.

You could do the same operation more explicitly using np.concatenate like this:

print(np.concatenate((a[..., None], b[..., None]), axis=2).shape)
# (3, 2, 2)

* Importing the entire contents of a module into your global namespace using import * is considered bad practice for several reasons. The idiomatic way is to import numpy as np.

'''    
data = []
labels = []

# Read all images inside Parasitized Path
for uninfected in uninfected_path:
    try:
        image = cv.imread('..\\cell_images\\Uninfected\\' + uninfected)        
        
        # Convert to PIL array 
        img_pil_array = Image.fromarray(image, 'RGB')
        
        # Resize image
        img_resized_np = img_pil_array.resize((64, 64))
        
        # append to data
        data.append(np.array(img_resized_np))
        
        # Label the image as 1 = uninfected
        labels.append(0)
        
    except AttributeError:
        print('Error exception uninfected_path')

for parasitized in parasitized_path:
    try:
        image = cv.imread('..\\cell_images\\Parasitized\\'+ parasitized)
        # Convert to numpy array
        img_pil_array = Image.fromarray(image, 'RGB')
        
        # Resize all images to get same size
        img_resized_np = img_pil_array.resize((64, 64))
        
        # alternative approach: using sklearn to resize   
        #from skimage.transform import resize
        #img_resized_sklearn = resize(image, (64, 64), anti_aliasing=True)
    
        # or you can resize image using openCV
        # you need to convert it to an array then, you can append to data array
        #img_resized_opncv = cv.resize(image, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
        
        # append all images into single array    
        data.append(np.array(img_resized_np))
        
        '''
        How can we track parasitized and normal?
        We are using all parasitized as label 1
        and all uninfected as label 1
        So, if the label is 1, it is parasitized
        '''
        labels.append(1) 
    except AttributeError:
        print('Error exeption parasitized_path')
    
#print(data[:2])
'''
To do    
1. Use openCV to resize image
2. Check PIL vs OpenCV for resize and array conversion

'''
print(data[1]) # numpy array
print(labels[1]) # 0

print(len(data)) # 27558
print(len(labels)) # 27558


#Shape of the data
# data = np.array(data)
# labels = np.array(labels)

print("Shape of the data array: ", np.shape(data))
print("Shape of the label array: ", np.shape(labels))


# Save image array to use later. Made it easy
cells = np.array(data)
labels = np.array(labels)

np.save('Cells' , cells)
np.save('Labels' , labels)


print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))

print(cells.shape) # (27558, 64, 64, 3)
print(cells.shape[0]) # 27558

plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(49):
    n += 1 
    
    # Take random image
    r = np.random.randint(0 , cells.shape[0] , 1)
    
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    
    plt.imshow(cells[r[0]])
    
    plt.title('{} : {}'
              .format('Infected' if labels[r[0]] == 1 
                                  else 'Unifected' , labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()

plt.figure(1, figsize = (15, 9))
n = 0
for i in range(16):
    
    n += 1
    
    # Take a random number in each iteration
    # np.random.randint(low, high, dtType)
    r = np.random.randint(0, cells.shape[0], 1)
    
    # Create subplot
    plt.subplot(4, 4, n)
    
    # Adjust subplots
    plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
    
    # Show single image using random selection
    # For each iteration, random number will be selected
    # image will be shown according to random numbered
    plt.imshow(cells[ r[0] ])
    
    
    # Show title
    plt.title('{} : {}'.format(
            'Infected' if labels[r[0]] == 1
            else 'Uninfected', labels[r[0]]
            ))
    
    plt.xticks([]), plt.yticks([])
plt.show()


print(cells.shape) # 27558, 64, 64, 3
plt.figure(1, figsize = (10 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(cells[0])
plt.title('{} : {}'.format(
            'Infected' if labels[0] == 1
            else 'Uninfected', labels[0]
            ))
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cells[26356])
plt.title('{} : {}'.format(
            'Infected' if labels[26356] == 1
            else 'Uninfected', labels[26356]
            ))
plt.xticks([]) , plt.yticks([])
plt.show()



# Load from the saved numpy array cells and labels
cells_loaded=np.load("Cells.npy")
labels_loaded=np.load("Labels.npy")

print(cells_loaded.shape[0]) # 27558

# Arrange 
shuffled_cells = np.arange(cells_loaded.shape[0])
print(shuffled_cells)

# Random shuffle
np.random.shuffle(shuffled_cells)
print(shuffled_cells)


print(cells_loaded[shuffled_cells])
print(labels_loaded[shuffled_cells])

cells_randomly_shuffled = cells_loaded[shuffled_cells]
labels_randomly_shuffled = labels_loaded[shuffled_cells]

print(np.unique(labels_randomly_shuffled)) # [0 1]
print(len(cells_randomly_shuffled)) # 27558

num_classes = len(np.unique(labels_randomly_shuffled))
print(num_classes)

len_data = len(cells_randomly_shuffled)
print(len_data)

print(0.1*len_data) # 2755.8

print(len(cells_randomly_shuffled[(int)(0.1*len_data):])) # 24803
print(len(cells_randomly_shuffled[:(int)(0.1*len_data)])) # 2755


''' Train Test Split Technique 1'''
(x_train, x_test) = cells_randomly_shuffled[(int)(0.1*len_data):], cells_randomly_shuffled[:(int)(0.1*len_data)]
print(len(x_train))
print(len(x_test))

# As we are working on image data we are normalizing data by divinding 255.
x_train = x_train.astype('float32')/255 
print(x_train)

x_test = x_test.astype('float32')/255

x_train_len = len(x_train)
x_test_len = len(x_test)
print(x_train_len)
print(x_test_len)

(y_train, y_test) = labels_randomly_shuffled[(int)(0.1*len_data):], labels_randomly_shuffled[:(int)(0.1*len_data)]

''' sklearn train test split version
from sklearn.model_selection import train_test_split

train_x , x , train_y , y = train_test_split(cells , labels , 
                                            test_size = 0.2 ,
                                            random_state = 111)

eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 111)

How to split .2, .2, 6?
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test 
= train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val 
= train_test_split(X_train, y_train, test_size=0.25, random_state=1)

'''

#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

print(y_train)
print(len(y_train))
print(y_test)
print(len(y_test))



# Create model
model = Sequential()

'''
api reference r1.13
tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
    
    
)

What does conv2D do in tensorflow?
https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow

'''
# Add hidden layer
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,3)))

# Step 2 - Max Pooling - Taking the maximum
# Why? Reduce the number of nodes for next Flattening step
model.add(MaxPooling2D(pool_size = (2, 2)))

# Add another hidden layer
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Droput to avoid overfitting
model.add(Dropout(0.2))

# Step 3 - Flatten - huge single 1 dimensional vector
model.add(Flatten())

# Step 4 - Full Connection
# output_dim/units: don't take too small, don't take too big
# common practice take a power of 2, such as 128, 256, etc.
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax")) 

model.summary()


# Step 5
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Step 6 Fitting model
model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1)

# Check accuracy
accuracy = model.evaluate(x_test, y_test, verbose=1)

# Save model weights
from keras.models import load_model
model.save('malaria_tfcnnsoftmax_category.h5')

# Use model using tkinter
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2 as cv

def convert_to_array(img):
    img = cv.imread(img)
    img = Image.fromarray(img)
    img = img.resize(64, 64)
    return np.array(img)

def get_label(label):
    if label == 0:
        return 'Uninfected'
    if label == 1:
        return 'Parasitized'

def predict_malaria(img_file):
    
    model = load_model('malaria_tfcnnsoftmax_category.h5')
    
    print('Predciting Malaria....')
    
    img_array = convert_to_array(img_file)
    img_array = img_array/255
    
    img_data = []
    img_data.append(img_array)
    img_data = np.array(img_data)
    
    score = model.predict(img_data, verbose=1)
    print('Score', score)
    
    label_index = np.argmax(score)
    
    result = get_label(label_index)
    return result, 'Predicted image is : ' + result + 'with accuracy = ' + str(accuracy)


"""from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter import filedialog 
from tkinter import messagebox as mbox

class Example(Frame):

    def __init__(self):
        super().__init__()   

        self.initUI()


    def initUI(self):

        self.master.title("File dialog")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)        

        

    def onOpen(self):

        ftypes = [('Image', '*.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()
        c,s=predict_cell(fl)
        root = Tk()
        T = Text(root, height=4, width=70)
        T.pack()
        T.insert(END, s)
        

def main():

    root = Tk()
    ex = Example()
    root.geometry("100x50+100+100")
    root.mainloop()  


if __name__ == '__main__':
    main()"""
    
'''
Note:
Kernel size vs filter?

https://stackoverflow.com/questions/51180234/keras-conv2d-filters-vs-kernel-size


Each convolution layer consists of several convolution channels (aka. depth or filters). 
In practice, they are in number of 64,128,256, 512 etc. This is equal to 
number of channels in output of convolution layer. kernel_size on the other 
hand is size of these convolution filters. In practice they are 3x3, 1x1 or 5x5. 
As abbreviation, they could be written as 1 or 3 or 5 as they are mostly square 
in practice.

Edit

Following quote should make it more clear.

Discussion on vlfeat

Suppose X is an input with size W x H x D x N (where N is the size of the batch) 
to a convolutional layer containing filters F (with size FW x FH x FD x K) in a network.

The number of feature channels D is the third dimension of the input X here 
(for example, this is typically 3 at the first input to the network if the 
input consists of colour images). The number of filters K is the fourth dimension 
of F. The two concepts are closely linked because if the number of filters in a 
layer is K, it produces an output with K feature channels. So the input to the 
next layer will have K feature channels.

The FW x FH above is filter size you are looking for.

Added

You should be familiar with filters. You can consider each filter to be responsible
ible to extract some type of feature from raw image. The CNNs try to learn such 
filters i.e. the filters are parametrized in CNNs are learned during training of 
CNNs. These are filters in CNN. You apply each filter in a Conv2D to each input 
channel and combine these to get output channels. So, number of filter and output 
channels are same.
    
'''
    