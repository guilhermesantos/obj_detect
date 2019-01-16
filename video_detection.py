import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import argparse

def load_net(model_id):
	directory = 'models/object-detection-deep-learning/'

	net = None
	model = dict()

	if(model_id == 1):
		directory += 'mobnetssd1/'
		model_file = directory+'MobileNetSSD_deploy.caffemodel'
		prototxt = directory+'MobileNetSSD_deploy.prototxt.txt'
		net = cv2.dnn.readNetFromCaffe(prototxt, model_file)	

		classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]

	elif(model_id == 2):
		directory += 'mobnetssd2/'
		frozen_graph = directory+'frozen_inference_graph.pb'
		graph_config = directory+'graph.pbtxt'
		net = cv2.dnn.readNetFromTensorflow(frozen_graph, graph_config)
	
		classes = []
		with open('labels.txt') as f:
			data = f.read()
			rows = data.split('\n')
			split_rows = [row.split() for row in rows]
			classes = [line[2] for line in split_rows]

	else:
		exit()
	
	model['net'] = net	
	model['id'] = model_id


	colors = np.random.uniform(0, 255, size=(len(classes), 3))


	return classes, colors, model

def detect(image, classes, colors, model):
	conf_thresh = 0.2

	h, w = image.shape[:2]
	
	image = cv2.resize(image, (300,300))


	model_id = model['id']
	net = model['net']
	
	if(model_id == 1):	
		blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
	else:	
		blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
	
	net.setInput(blob)
	start = time.time()
	detections = net.forward()
	end = time.time()
	print('Tempo de inferencia: {:.5}%'.format(end-start))
	
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
			print('class index', class_index)
			class_indexes.append(class_index)

			#Numeros que representan a bounding box
			bnd_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			#Converte numeros da bounding box em coordenadas da imagem
			bounding_boxes.append(bnd_box.astype('int'))

	return bounding_boxes, class_indexes, confidences

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', dest='model', required=True, help='Especificar o modelo v1, v2')
	args = vars(parser.parse_args())
	print(args['model'])

	classes, colors, model = load_net(int(args['model']))

	capture = cv2.VideoCapture(0)
	while(True):
		ret, image = capture.read()
		#rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#cv2.imshow('frame', frame)

		boxes, class_indexes, confidences = detect(image, classes, colors, model)

		for box, class_index, confidence in zip(boxes, class_indexes, confidences):

			start_x, start_y, end_x, end_y = box

			#Desenha um retangulo na regiao da bounding box
			cv2.rectangle(image, (start_x, start_y), (end_x, end_y), colors[class_index-1], 2)

			#Usa o indice para pegar o label da classe e monta o texto que vai aparecer no quadro 

			#print('Detection: '+output_label)

			#Texto que sera colocado na bounding box

			if(model['id'] == 1):
				class_label = classes[class_index]
				output_label = '{} {} {:.2f}%'.format(class_index, class_label, confidence*100)

			else:
				class_label = classes[class_index-1]
				output_label = '{} {}'.format(class_index, class_label)

			text_y = start_y - 15 if start_y-15 > 15 else start_y + 15
			cv2.putText(image, output_label, (start_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            	colors[class_index-1], 2)

		cv2.imshow('Detection', image)

		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break

	capture.release()
	cv2.destroyAllWindows()
	print('forcing git to upload this file as')

main()
