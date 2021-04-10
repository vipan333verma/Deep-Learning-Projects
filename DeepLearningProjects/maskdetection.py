import os
import cv2

from keras_preprocessing import image

categories = ['with_Mask','without_Mask']

data =[]
withMaskPath ="D:\pyProjects\DeepLearningProjects\maskDetection\\training"
for cat in categories:
    path = os.path.join(withMaskPath,cat)
    label = categories.index(cat)


    for file in os.listdir(path):
        img_path = os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        #print(img.shape)

        data.append([img,label])
print(len(data))

import random
random.shuffle(data)

x =[]
y =[]

for features,label in data:
    x.append(features)
    y.append(label)

print(len(x))
print(len(y))

import numpy as np

x= np.array(x)
y = np.array(y)

print(x.shape)
x = x/255


from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

print(x_train.shape)



from keras.applications.vgg16 import VGG16
vgg = VGG16()
print(vgg.summary())


from keras import Sequential
model = Sequential()
#model.summary()

for layer in vgg.layers[:-1]:
    model.add(layer)

print(model.summary())

for layer in model.layers:
    layer.trainable=False

print(model.summary())

from keras.layers import Dense
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,validation_data=(x_test,y_test))


#webcam
cap = cv2.VideoCapture(1)

#draw
def draw_labl(img,text,pos,bg_color):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x = pos[0]+text_size[0][0]+2
    end_y = pos[1]+text_size[0][1]-2

    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

#detect face
def detect_mask(img):
     y_predict=  model.predict(img.reshape(1,224,224,3))
     return  y_predict[0][0]


while True:
    rep,frame =cap.read()

    #detect
    imgg =cv2.resize(frame,(224,224))
    y_predict = detect_mask(imgg)

    result= round(y_predict,1)
    print(result)

    if result ==0.0:
        draw_labl(frame,"MASK",(30,30),(0,255,0))
    else:
        draw_labl(frame," NO MASK",(30,30),(0,255,0))
   # draw_labl(frame,"FACE Mask Detetctor",(10,10),(0,0,255))
   # print(y_predict)

    cv2.imshow("window",frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
#cv2.destroyWindow()



sample ="D:\pyProjects\DeepLearningProjects\maskDetection\\training\with_Mask\with_Mask.d8d77f14-99d5-11eb-b6d2-002b67c616aa.jpg"
testt = cv2.imread(sample)
testt =cv2.resize(testt,(224,224))

print(detect_mask(testt))










