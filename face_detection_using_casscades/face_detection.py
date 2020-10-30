
import cv2
import matplotlib.pyplot as plt


# ...image reading and showing....

# image of jinks rahane (meme)
img_path1 = 'Screenshot from 2020-10-05 20-29-28.png'

#image of kl_rahul :)
img_path2 = 'kl_rahul.jpg'

img = cv2.imread(img_path2,1)
grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# gives coordinates of the faces detected in the image
faces = face_cascade.detectMultiScale(grey_img,scaleFactor=1.05,
                                      minNeighbors = 5)
# print(type(img))
print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)

#printing the type and shape of faces
print(type(faces))
print(faces)

# drawing rectangles on the image
for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


# showing the image
cv2.imshow('meme',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

