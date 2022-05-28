#Importing utitlies
import uuid
from flask import Flask,request,render_template,jsonify
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
app=Flask(__name__)
#In the start of application it returns the content of index3.html
@app.route('/')
def my_form():
    return render_template('index3.html')

#When we add "/project2" to the url it renders the index.html
@app.route('/project2')
def my_form_ex1():
    return render_template('index.html')

#After reaching the index.html we get the values written in the textfield by mention the name
@app.route('/project2', methods=['POST'])
def my_form_post():
    req1=request.form.get("u1")#beardyesornno
    req2=request.form.get("u2")#eyecolor
    req3=request.form.get("u3")#glasses
    req4=request.form.get("u4")#nosetype
    req5=request.form.get("u5")#skintype
    req6=request.form.get("u8")
    val=[]
    #For video capture-> If the user wants to capture their photo for updatign database
    if(req6=='y'):
        cam = cv2.VideoCapture(0)
    
        cv2.namedWindow("test")
        
        img_counter = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)
        
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name=os.path.join('G:\\ML_facereg\\cloned\\Tensorflow1\\workspace\\images\\all\\all'+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(img_name, frame)
                
                img_counter += 1
        
        cam.release()
        cv2.destroyAllWindows()
    #As we have 10 fiels to be taken care of
    '''
    Case1 : With beard
    Case 2 : Without beard
    Case 3 : With brown eyes
    Case 4 : Without brown eyes
    Case 5 : With glass
    Case 6 : Without glasses
    Case 7 : With sharp nose
    Case 8 : With blunt nose
    Case 9 : With black skin
    Case 10 : With white skin 
    '''
    if(req1=='y'):
        val.append('T')
        val.append('F')
    if(req1=='y'):
        val.append('F')
        val.append('T')
    if(req2=='brown'):
        val.append('T')
        val.append('F')
    if(req2=='other'):
        val.append('F')
        val.append('T')
    if(req3=='y'):
        val.append('T')
        val.append('F')
    if(req3=='n'):
        val.append('F')
        val.append('T')
    if(req4=='blunt'):
        val.append('T')
        val.append('F')
    if(req4=='sharp'):
        val.append('F')
        val.append('T')
    if(req5=='black'):
        val.append('T')
        val.append('F')
    if(req5=='white'):
        val.append('F')
        val.append('T')
    
    #This is the image training and classification code (BEtter explanation in myproj.ipynb)
    import pathlib
    data_dir1='G:\\ML_facereg\\cloned\\Tensorflow1\\workspace\\images\\collectedimages'
    data_dir1 = pathlib.Path(data_dir1)
    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir1,validation_split=0.2,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir1,validation_split=0.2,subset="validation",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    num_classes = len(class_names)
    model = keras.Sequential([layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),layers.Conv2D(16, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(32, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(num_classes)])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    epochs=10
    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),layers.RandomRotation(0.1),layers.RandomZoom(0.1),])
    model = keras.Sequential([data_augmentation,layers.Rescaling(1./255),layers.Conv2D(16, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(32, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Dropout(0.2),layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(num_classes)])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    epochs = 15
    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    #Creating list to sort out the images from database and updating each feature with the image
    beardyes = []
    beardno=[]
    eyebrown = []
    eyeother = []
    glassyes=[]
    noglasseyes=[]
    noseblunt=[]
    nosesharp=[]
    skinblack=[]
    skinwhite=[]
    def argpass(s):
        pathss = "G:\\ML_facereg\\cloned\\Tensorflow1\\workspace\\images\\all\\{}".format(s)

        img = tf.keras.utils.load_img(
            pathss, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        score=score*100
        #After training we get the score of each feature and based on  the quality of the score we update the feature(score is the percentage of confidentialty that an image belongs to a certain feature)
        c=0
        while(c!=8):
            if(c==0):
                if(score[c]>20):
                    beardyes.append(s)
                else:
                    beardno.append(s)
                c=c+1
            elif(c==1):
                if(score[c]>score[c+1]):
                    eyebrown.append(s)
                else:
                    eyeother.append(s)
                c=c+2
            elif(c==3):
                if(score[c]>20):
                    glassyes.append(s)
                else:
                    noglasseyes.append(s)
                c=c+1
            elif(c==4):
                if(score[c]>score[c+1]):
                    noseblunt.append(s)
                else:
                    nosesharp.append(s)
                c=c+2
            elif(c==6):
                if(score[c]>score[c+1]):
                    skinblack.append(s)
                else:
                    skinwhite.append(s)
                c=c+2
    
    arrn=os.listdir('G:\\ML_facereg\\cloned\\Tensorflow1\\workspace\\images\\all')
    for i in arrn:
        argpass(i)
    ret=[]
    #Now we create a list of images to be returned
    #As the user enters the feature we classfy as T/F for each feature and based on that we classify them 
    if(val[8]=='T'):
        for i in skinblack:
            ret.append(i)
    else:
        for i in skinwhite:
            ret.append(i)
    if(val[4]=='T'):
        for i in ret:
            if(i not in glassyes):
                ret.remove(i)
    else:
        for i in ret:
            if(i not in noglasseyes):
                ret.remove(i)
    if(val[0]=='T'):
        for i in ret:
            if(i not in beardyes):
                ret.remove(i)
    else:
        for i in ret:
            if(i not in beardno):
                ret.remove(i)
    if(len(ret)>15):
        if(val[2]=='T'):
            for i in ret:
                if(i not in eyebrown):
                    ret.remove(i)
        else:
            for i in ret:
                if(i not in eyeother):
                    ret.remove(i)
        if(len(ret)>15):
            if(val[6]=='T'):
                for i in ret:
                    if(i not in noseblunt):
                        ret.remove(i)
            else:
                for i in ret:
                    if(i not in nosesharp):
                        ret.remove(i)
    
    ret=list(set(ret))
    
    s="The relative images possible are [G:\ML_facereg\\cloned\\Tensorflow1\\workspace\\images\\all\\] :"
    for i in ret:
        s+=i
        s+='\n'
    return s

#If "project1" written along with URl-> 
@app.route('/project1')
def my_form_ex2():
    return render_template('index1.html')

#We entered the project 1
@app.route('/project1', methods=['POST'])
def my_form_post_2():
    #Same explaantion in myproj.ipynb
    import pathlib
    pathss=request.form.get("u7")
    data_dir1='G:\\ML_facereg\\cloned\\Tensorflow\\workspace\\images\\collectedimages1'
    data_dir1 = pathlib.Path(data_dir1)
    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir1,validation_split=0.2,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir1,validation_split=0.2,subset="validation",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    num_classes = len(class_names)
    model = keras.Sequential([layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),layers.Conv2D(16, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(32, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(num_classes)])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    epochs=10
    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),layers.RandomRotation(0.1),layers.RandomZoom(0.1),])
    model = keras.Sequential([data_augmentation,layers.Rescaling(1./255),layers.Conv2D(16, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(32, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64, 3, padding='same', activation='relu'),layers.MaxPooling2D(),layers.Dropout(0.2),layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(num_classes)])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    epochs = 15
    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    img = tf.keras.utils.load_img(
        pathss, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

if __name__ == "__main__":
    app.debug=True
    app.run()


