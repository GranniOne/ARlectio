import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.image import Image
import kivy
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import lectio
from datetime import datetime


# laver og eller opdatere billederne så dataen på dem passer!!!
font = ImageFont.truetype('venv/Roboto-Light.ttf', size=15)
augImages = []
loadAugImage = []
lokaleDict = {1620295453: 1, 1620295457: 2, 1620295440: 3,}
dt = datetime.now()
for lokale in lokaleDict:
    # Create a new image with a white background
    lectioDataImage = Image.new('RGB', (300, 300), ImageColor.getrgb("#0bbbef"))

    # Initialize the drawing context
    draw = ImageDraw.Draw(lectioDataImage)
    # henter data fra lectio og skriver det ind på billedet
    text = "\n\n".join(lectio.Lokale.get_one_day_short(dt.weekday()+2,lokale))
    text_pos = (2, 5)
    # Draw the text on the image
    draw.text(text_pos, text, fill='white', font=font)

    # Save the image to a file
    lectioDataImage.save('venv/data/{}.png'.format(lokaleDict[lokale]))
    augImages.append(str(lokaleDict[lokale])+".png")
# Billeder indlæses
for augImg in augImages:
    print(os.path.exists("venv/data/{}".format(augImg)))
    loadAugImage.append(cv2.imread("venv/data/{}".format(augImg)))

print(augImages)

cap = cv2.VideoCapture(0)

# funktion til at finde markers fra kamera
def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    if draw == True:
        # indbygget funktion til at tegne de funde markeringer
        cv2.aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs, ids]


# Tegner billeder over marker
def augmentAruco(bbox, id, img, imgAug, drawID=False):
    # starter med at vrænge billedet så det passer til det som kammeraet ser
    tl = bbox[0][0][0],bbox[0][0][1]
    tr = bbox[0][1][0],bbox[0][1][1]
    br = bbox[0][2][0],bbox[0][2][1]
    bl = bbox[0][3][0],bbox[0][3][1]

    h,w,c= imgAug.shape


    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    # markere stedet markeren er på med sort firkant
    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))
    # fylder det sorte område ud med det vrængede billede
    imgOut = img + imgOut

    # Tegn ID på img out
    if drawID:
        cv2.putText(imgOut,str(id),(int(tl[0]),int(tl[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (255,0,255),2)
    return imgOut

# Loop for at koden hele tiden tjekker efter markers


class MainApp(App):
    # bygger Kivy app
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.image = kivy.uix.image.Image()
        layout.add_widget(self.image)
        self.updateimg()
        # sørger for at opdatere billedet hele tiden
        Clock.schedule_interval(self.updateimg,1.0/30.0)
        return layout
    # Opdatere billedet, kalder på augmenter funktion
    def updateimg(self, *args):
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)
        # gå alle markers igennem og kald på augmenter funktionen
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                try:
                    img = augmentAruco(bbox,id,img,loadAugImage[int(id)-1])
                except:
                    print("id findes ikke i database: "+str(id))


        # Omformer CV2 billede til billede som kivy kan forstå
        buffer = cv2.flip(img,0).tostring()
        texture = Texture.create(size=(img.shape[1],img.shape[0]),colorfmt='bgr')
        texture.blit_buffer(buffer,colorfmt='bgr',bufferfmt='ubyte')
        # tegner billedet på skærmen.
        self.image.texture = texture

if __name__ == '__main__':
    MainApp().run()
