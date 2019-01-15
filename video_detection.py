import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import glob

def load_net():
	directory = 'models/object-detection-deep-learning/'
	prototxt = directory+'MobileNetSSD_deploy.prototxt.txt'
	model = directory+'MobileNetSSD_deploy.caffemodel'

	classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	net = cv2.dnn.readNetFromCaffe(prototxt, model)	

	return classes, colors, net

#directory = 'models/object-detection-deep-learning/'
#prototxt = directory+'MobileNetSSD_deploy.prototxt.txt'
#model = directory+'MobileNetSSD_deploy.caffemodel'

#classes = ["background", "aeroplane", "bicycle", "bird", "boat",
#    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#    "sofa", "train", "tvmonitor"]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))
#net = cv2.dnn.readNetFromCaffe(prototxt, model)

def detect(image, classes, colors, net):
	conf_thresh = 0.2

	h, w = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1, (224, 244), (104, 117, 123))
	
	net.setInput(blob)
	start = time.time()
	detections = net.forward()
	end = time.time()
	#print('Tempo de predicao: {:.5}'.format(end-start))
	
	bounding_boxes = []
	labels = []
	class_indexes = []
	confidences = []
	for i in np.arange(0, detections.shape[2]):

        #Confianca da predicao
		confidence = detections[0, 0, i, 2]
		confidences.append(confidence)

		if(confidence > conf_thresh):

			#Indice da classe (p/ descoberta do label e da cor)
			class_index = int(detections[0, 0, i, 1])
			class_indexes.append(class_index)

			#Numeros que representan a bounding box
			bnd_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			#Converte numeros da bounding box em coordenadas da imagem
			bounding_boxes.append(bnd_box.astype('int'))

	return bounding_boxes, class_indexes, confidences

def main():
	classes, colors, net = load_net()

	capture = cv2.VideoCapture(0)
	while(True):
		ret, image = capture.read()
		#rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#cv2.imshow('frame', frame)

		boxes, class_indexes, confidences = detect(image, classes, colors, net)

		for box, class_index, confidence in zip(boxes, class_indexes, confidences):
			start_x, start_y, end_x, end_y = box

			#Desenha um retangulo na regiao da bounding box
			cv2.rectangle(image, (start_x, start_y), (end_x, end_y), colors[class_index], 2)

			#Usa o indice para pegar o label da classe e monta o texto que vai aparecer no quadro 
			class_label = classes[class_index]
			output_label = '{} {:.2f}%'.format(class_label, confidence*100)
			print('Detection: '+output_label)

			#Texto que sera colocado na bounding box
			text_y = start_y - 15 if start_y-15 > 15 else start_y + 15

			cv2.putText(image, output_label, (start_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            	colors[class_index], 2)

		cv2.imshow('Detection', image)

		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break

	capture.release()
	cv2.destroyAllWindows()

main()