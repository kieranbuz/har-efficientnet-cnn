# -*- coding: utf-8 -*-
'''
Created on Sun Apr 22 17:16:14 2022

@author: Kieran Burrell
'''
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
sns.set_style('darkgrid')
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.layers.experimental import preprocessing # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from keras import regularizers
from keras.layers import concatenate # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

train_csv_path=r'../Artefact/Dataset 1/Human Action Recognition/Training_set.csv'
test_csv_path=r'../Artefact/Dataset 1/Human Action Recognition/Testing_set.csv'
train_img_path=r'../Artefact/Dataset 1/Human Action Recognition/train'
test_img_path=r'../Artefact/Dataset 1/Human Action Recognition/Testing_set.csv'
df=pd.read_csv(train_csv_path)
# modify df to have column names filepaths and labels
df.columns=['filepaths', 'labels']
# modify df so entries in filepaths column are the full path to the image file
df['filepaths']=df['filepaths'].apply(lambda x: os.path.join(train_img_path, x))
# split df into a train_df , a valid_df and a test_df
train_df, dummy_df=train_test_split(df, train_size=.9, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df= train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])     
print()
print('Dataframe Sizes')
print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
print()
# get the number of classes and the images count for each class in train_df
classes=sorted(list(train_df['labels'].unique()))
class_count = len(classes)
print('The number of classes in the dataset is: ', class_count)
print()
groups=train_df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
countlist=[]
classlist=[]
for label in sorted(list(train_df['labels'].unique())):
    group=groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

# get the classes with the minimum and maximum number of train images
max_value=np.max(countlist)
max_index=countlist.index(max_value)
max_class=classlist[max_index]
min_value=np.min(countlist)
min_index=countlist.index(min_value)
min_class=classlist[min_index]
print()
print(max_class, 'has the most images =', max_value)
print(min_class, 'has the least images =', min_value)
print()
def trim(df, max_samples, min_samples, column):
    df=df.copy()
    groups=df.groupby(column)    
    trimmed_df = pd.DataFrame(columns = df.columns)
    groups=df.groupby(column)
    for label in df[column].unique(): 
        group=groups.get_group(label)
        count=len(group)    
        if count > max_samples:
            sampled_group=group.sample(n=max_samples, random_state=123,axis=0)
            trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            if count>=min_samples:
                sampled_group=group        
                trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
    print('after trimming, the maximum samples in any class is now',max_samples)
    print('and the minimum samples in any class is', min_samples)
    return trimmed_df
max_samples=400
min_samples=400
column='labels'
train_df= trim(train_df, max_samples, min_samples, column)
print()
working_dir=r'./'
img_size=(200,200)
batch_size=32 
trgen=ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images)
t_and_v_gen=ImageDataGenerator()
print('Training Set...')
train_gen=trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
print('Validation Set...')
valid_gen=t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
length=len(test_df)
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
test_steps=int(length/test_batch_size)
print('Testing Set...')
test_gen=t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
classes=list(train_gen.class_indices.keys())
class_indices=list(train_gen.class_indices.values())
class_count=len(classes)
labels=test_gen.labels
print()
print('Test batch size: ' ,test_batch_size, '  Test steps: ', test_steps, ' Number of classes : ', class_count)
print()
def show_image_samples(gen ):
    t_dict=gen.class_indices
    classes=list(t_dict.keys())    
    images,labels=next(gen) # get a sample batch from the generator 
    plt.figure(figsize=(20, 20))
    length=len(labels)
    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25
    for i in range(r):        
        plt.subplot(5, 5, i + 1)
        image=images[i] /255       
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()
    
show_image_samples(train_gen)

img_shape=(img_size[0], img_size[1], 3)

#Base Model
def conv_block(filters, inputs):
    x = layers.SeparableConv2D(filters, 3, activation='relu', padding='same')(inputs)
    x = layers.SeparableConv2D(filters, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.MaxPool2D()(x)

    return outputs

def dense_block(units, dropout_rate, inputs):
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(0.001) , activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dropout(dropout_rate)(x)

    return outputs

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    x = preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(32, x)
    x = conv_block(64, x)

    x = conv_block(128, x)
    x = layers.Dropout(0.2)(x)

    x = conv_block(256, x)
    x = layers.Dropout(0.2)(x)
    
    # x = conv_block(512, x)
    # x = layers.Dropout(0.2)(x)
    
    # x = conv_block(1024, x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    #Adding the Hidden layers
    # x = dense_block(1024, 0.1, x)
    x = dense_block(512, 0.1, x)
    x = dense_block(256, 0.1, x)
    x = dense_block(128, 0.1, x)
    x = dense_block(64, 0.1, x)
    #Adding the Output Layer
    x = layers.Dense(15, activation='softmax')(x)
    
    booster_model = EfficientNetB0(include_top=False, input_shape=input_shape, classes=15, input_tensor=inputs)
    booster_model.trainable = False
    booster_model.layers.pop() # Get rid of the classification layer
    booster_model.outputs = [booster_model.layers[-1].output]
    y = layers.Flatten()(booster_model.output)
    y = layers.Dense(15, activation='softmax')(y)
    
    x_y = concatenate([x,y], name='concatenated_layer')
    
    outputs = layers.Dense(15, activation='softmax')(x_y)
    
    return keras.Model(inputs, outputs)

model = make_model(input_shape=img_shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

epochs = 500

history=model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, validation_steps=None, shuffle=False, initial_epoch=0)

def tr_plot(tr_data, start_epoch):
    #Plot the training and validation data
    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout    
    plt.show()
    
tr_plot(history,0)

def predictor(test_gen, test_steps):
    y_pred= []
    y_true=test_gen.labels
    classes=list(train_gen.class_indices.keys())
    class_count=len(classes)
    errors=0
    preds=model.predict(test_gen, steps=test_steps, verbose=1) # predict on the test set
    tests=len(preds)
    for i, p in enumerate(preds):
            pred_index=np.argmax(p)         
            true_index=test_gen.labels[i]  # labels are integer values
            if pred_index != true_index: # a misclassification has occurred                                           
                errors=errors + 1
            y_pred.append(pred_index)
    acc=( 1-errors/tests) * 100
    print(f'there were {errors} in {tests} tests for an accuracy of {acc:6.2f}')
    ypred=np.array(y_pred)
    ytrue=np.array(y_true)
    if class_count <=30:
        cm = confusion_matrix(ytrue, ypred )
        # plot the confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
        plt.xticks(np.arange(class_count)+.5, classes, rotation=90)
        plt.yticks(np.arange(class_count)+.5, classes, rotation=0)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes, digits= 4) # create classification report
    print('Classification Report:\n----------------------\n', clr)
    return errors, tests
errors, tests=predictor(test_gen, test_steps)

subject='activities' 
acc=str(( 1-errors/tests) * 100)
index=acc.rfind('.')
acc=acc[:index + 3]
save_id= subject + '_' + str(acc) + '.h5' 
model_save_loc=os.path.join(working_dir, save_id)
model.save(model_save_loc)
print ('model was saved as ' , model_save_loc ) 
