import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import argparse
import os

def load_coco_classes():
	classes = []
	with open('labels.txt') as f:
		data = f.read()
		rows = data.split('\n')
		split_rows = [row.split() for row in rows]
		classes = [line[2] for line in split_rows]
	return classes

def load_net(model_id):
	directory = 'models/object-detection-deep-learning/'

	net = None
	model = dict()
	classes = []

	if(model_id == 1):
		model['name'] = 'MobileNet-SSD v1 Tutorial'
		directory += 'mobnetssd1/'
		model_file = directory+'MobileNetSSD_deploy.caffemodel'
		prototxt = directory+'MobileNetSSD_deploy.prototxt.txt'
		net = cv2.dnn.readNetFromCaffe(prototxt, model_file)	

		model['classes'] = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]

	elif(model_id == 2):
		model['name'] = 'MobiletNet-SSD v1 Coco'
		directory += 'mobnetssd1_alt/'
		frozen_graph = directory+'frozen_inference_graph.pb'
		config = directory+'graph.pbtxt'
		net = cv2.dnn.readNetFromTensorflow(frozen_graph, config)
		model['classes'] = load_coco_classes()

	elif(model_id == 3):
		model['name'] = 'MobiletNet-SSD v2 Coco'
		directory += 'mobnetssd2/'
		frozen_graph = directory+'frozen_inference_graph.pb'
		graph_config = directory+'graph.pbtxt'
		net = cv2.dnn.readNetFromTensorflow(frozen_graph, graph_config)
		model['classes'] = load_coco_classes()
	
	model['net'] = net	
	model['id'] = model_id
	model['colors'] = np.random.uniform(0, 255, size=(len(model['classes']), 3))

	return model

def detect(model, image):
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
	#print('Tempo de inferencia: {:.5}'.format(end-start))
	elapsed = end-start
	
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

	return bounding_boxes, class_indexes, confidences, elapsed

def draw_boxes(model, image, box, class_index, confidence):
	start_x, start_y, end_x, end_y = box

	#Desenha um retangulo na regiao da bounding box
	cv2.rectangle(image, (start_x, start_y), (end_x, end_y), model['colors'][class_index-1], 2)

	#Texto que sera colocado na bounding box
	if(model['id'] == 1):
		class_label = model['classes'][class_index]
		output_label = '{} {} {:.2f}%'.format(class_index, class_label, confidence*100)

	else:
		class_label = model['classes'][class_index-1]
		output_label = '{} {} {:.2f}%'.format(class_index, class_label, confidence*100)

	text_y = start_y - 15 if start_y-15 > 15 else start_y + 15


	cv2.putText(image, output_label, (start_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            	model['colors'][class_index-1], 2)

	return image

def parse_commandline():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', dest='model', type=int, required=False, help='Especificar o modelo v1, v2')
	
	parser.add_argument('-f', dest='file', required=False)
	args = vars(parser.parse_args())
	return args

def record_measurements(model, measurements):
	measure_file = '{}.dat'.format(model['name'])
	print('{} measurements collected'.format(len(measurements)))
	print('writing to {}'.format(measure_file))

	if(os.path.exists(measure_file)):
		os.remove(measure_file)
	
	content = ''
	for measurement in measurements:
		content += str(measurement)+'\n'			

	with open(measure_file, 'w') as f:
		f.write(content)

	
def detect_and_measure(model, video_file=None):
	if(video_file == None):
		capture = cv2.VideoCapture(0)
	else:
		capture = cv2.VideoCapture(video_file)
	
	measurements = []

	while(capture.isOpened()):
		ret, image = capture.read()
		boxes, class_indexes, confidences, elapsed = detect(model, image)
		measurements.append(elapsed)

		for box, class_index, confidence in zip(boxes, class_indexes, confidences):
			draw_boxes(model, image, box, class_index, confidence)
		
		image = cv2.resize(image, (1000,500))
		cv2.imshow('Detection', image)

		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break

		if(len(measurements) % 100 == 0):
			record_measurements(model, measurements)
			break

	capture.release()
	cv2.destroyAllWindows()

def main():
	args = parse_commandline()
	print(args['model'])	
	print(args['file'])

	model = load_net(args['model'])

	detect_and_measure(model, args['file'])

#main()
