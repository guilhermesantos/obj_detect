import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
 

directory = 'models/deep-learning-opencv/'
prototxt = directory+'bvlc_googlenet.prototxt'
model = directory+'bvlc_googlenet.caffemodel'
labels = directory+'synset_words.txt'

rows = open(labels).read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]
print('Classes', classes[0:5])

net = cv2.dnn.readNetFromCaffe(prototxt, model)

capture = cv2.VideoCapture(0)
while(True):
	ret, frame = capture.read()
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	cv2.imshow('frame', frame)

	blob = cv2.dnn.blobFromImage(frame, 1, (224, 244), (104, 117, 123))
	net.setInput(blob)

	start = time.time()
	preds = net.forward()
	end = time.time()

	print('Tempo de predicao: {:.5}'.format(end-start))

	idxs = np.argsort(preds[0])[::-1][:5]
	print('Classe', classes[idxs[0]], 'Probabilidade', preds[0][idxs[0]]*100)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

capture.release()
cv2.destroyAllWindows()



#fig, axis = plt.subplots()
#axis.set_title(classes[idxs[0]])
#axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#print('Probabilidade {:.5}', preds[0][idxs[0]]*100)
