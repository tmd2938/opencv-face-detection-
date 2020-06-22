import cv2
import numpy as np
import os
from datetime import datetime
from time import sleep
import pymysql
from opcua import Client, ua








recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
cascadePath = "../opencv/opencv-3.4.3/data/haarcascades/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

con = pymysql.connect(host = '172.21.50.110', user='root', password='fbtmdals12', db='enterance', charset='utf8')
cur = con.cursor()
sql = "insert into aaa(name,employeenum,state) values(%s,%s,%s)"


url = "opc.tcp://172.21.42.150:4840"
client = Client(url)
client.connect()
var = client.get_node("ns=2;s=client")
print("client connect at {}".format(url))


id = 0
names = ['unknown','seungmin']


cam = cv2.VideoCapture(-1)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
dbInsertFlag = False
kncount = 0
uncount = 0



while True:
    
    
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    TIME = datetime.now()
    cv2.putText(img,str(TIME),(10,30),font,1,(255,255,255),2,cv2.LINE_AA)
    
    
    

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if (confidence <100):
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
            if (id == "seungmin"):
                print('kn confidence = ', confidence)
                kncount+=1
                uncount = 0
                if (dbInsertFlag == True and kncount == 5):
                    cur.execute(sql, ("seungmin",36177,"restart"))
                    con.commit()
                    var.set_value(3, ua.VariantType.Int32)
                    kncount = 0
                    print('restart')
                    dbInsertFlag = False
        
        elif(confidence >100):
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))
            print('un confidence = ', confidence)
            uncount+=1
            kncount = 0
            if (dbInsertFlag == False and uncount == 5):
                cur.execute(sql, ("unknown",00000,"stop"))
                con.commit()
                var.set_value(1, ua.VariantType.Int32)
                uncount = 0
                print('emergency stop')
                dbInsertFlag = True
            
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0),1)
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    
    #cv2.namedWindow('camera', cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty('camera',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        con.close()
        break

print("\n [INFO] Exiting Program and cleanup stuff")

cam.release()
cv2.destroyAllWindows()



