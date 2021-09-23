import cv2


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

classNomes = []
classFilept = 'coco.pt'
with open(classFilept, 'rt') as f:
    classNomes = f.read().rstrip('\n').split('\n')

configPath = 'obj-ssd-coco.pbtxt'
weightsPath = 'obj-frozen.pb'
thres = 0.5

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


captura = cv2.VideoCapture(0)
captura.set(3, 800)
captura.set(4, 600)

while True:
    success, meuframe = captura.read()
    classIds, confs, bbox = net.detect(meuframe, confThreshold=thres)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if round(confidence*100) > 60:
            if classId <= (len(classNames)):
                cv2.rectangle(meuframe, box, color=(0, 255, 0), thickness=2)
                print(classId, classNames[classId - 1].upper(), round(confidence * 100, 2), box)
                cv2.putText(meuframe, classNomes[classId-1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(meuframe, str(round(confidence*100, 1)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Preview', meuframe)
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break
