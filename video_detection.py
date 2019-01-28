import cv2
import numpy as np
import time
import glob
import argparse
import os
import json
import matplotlib.pyplot as plt
import operator
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
		model['name'] = 'MobileNet-SSD v1 Coco'
		directory += 'mobnetssd1_alt/'
		frozen_graph = directory+'frozen_inference_graph.pb'
		config = directory+'graph.pbtxt'
		net = cv2.dnn.readNetFromTensorflow(frozen_graph, config)
		model['classes'] = load_coco_classes()

	elif(model_id == 3):
		model['name'] = 'MobileNet-SSD v2 Coco'
		directory += 'mobnetssd2/'
		frozen_graph = directory+'frozen_inference_graph.pb'
		graph_config = directory+'graph.pbtxt'
		net = cv2.dnn.readNetFromTensorflow(frozen_graph, graph_config)
		model['classes'] = load_coco_classes()
	
	model['net'] = net	
	model['id'] = model_id
	model['colors'] = np.random.uniform(0, 255, size=(len(model['classes']), 3))

	return model

def detections_to_dictionary(image, predictions, image_id=None):
	objects = []
	conf_thresh = 0.2
	h, w = image.shape[:2]

	for i in np.arange(0, predictions.shape[2]):

    	#Confianca da predicao
		confidence = predictions[0, 0, i, 2]

		if(confidence > conf_thresh):

			#Indice da classe (p/ descoberta do label e da cor)
			class_index = int(predictions[0, 0, i, 1])

			#Numeros que representan a bounding box			
			bnd_box = predictions[0, 0, i, 3:7] 
			bnd_box = bnd_box * np.array([w, h, w, h])
			start_x = float(bnd_box[0])
			start_y = float(bnd_box[1])
			end_x = float(bnd_box[2])
			end_y = float(bnd_box[3])

			width = float(end_x-start_x)
			height = float(end_y-start_y)

			detection = {
				'category_id': int(class_index),
				'bbox': [start_x, start_y, width, height],
				'score': float(confidence)
			}

			if(image_id != None):
				detection['image_id'] = int(image_id)

			objects.append(detection)

	return objects

def detect(model, image, image_id=None):
	model_id = model['id']
	net = model['net']
	
	if(model_id == 1):	
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 0.007843, (300, 300), 127.5)
	else:	
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), size=(300, 300), swapRB=True, crop=False)
	
	net.setInput(blob)
	start = time.time()
	detections = net.forward()
	end = time.time()

	time_elapsed = end-start
	
	objects = detections_to_dictionary(image, detections, image_id)

	return objects, time_elapsed

def draw_boxes(model, image, objects):
	#Desenha um retangulo na regiao da bounding box
	for detected in objects:
		class_index = detected['category_id']
		confidence = detected['score']
		start_x, start_y, width, height = detected['bbox']
		end_x = start_x + width
		end_y = start_y + height

		cv2.rectangle(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), model['colors'][class_index-1], 2)

		#Texto que sera colocado na bounding box
		if(model['id'] == 1):
			class_label = model['classes'][class_index]
			output_label = '{} {} {:.2f}%'.format(class_index, class_label, confidence*100)

		else:
			class_label = model['classes'][class_index-1]
			output_label = '{} {} {:.2f}%'.format(class_index, class_label, confidence*100)

		text_y = start_y - 15 if start_y-15 > 15 else start_y + 15


		cv2.putText(image, output_label, (int(start_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
	            	model['colors'][class_index-1], 2)

	return image

def record_time(model, time_measurements, test_name):
	measure_file = 'benchmark/time_{}_{}.dat'.format(model['name'], test_name)
	print('{} measurements collected'.format(len(time_measurements)))
	print('writing time measurement results to {}'.format(measure_file))

	if(os.path.exists(measure_file)):
		os.remove(measure_file)
	
	content = ''
	for measurement in time_measurements:
		content += str(measurement)+'\n'

	with open(measure_file, 'w') as f:
		f.write(content)

def record_detections(model, detections, test_name):
	detection_file = './benchmark/detections_{}_{}.json'.format(model['name'], test_name)
	if(os.path.exists(detection_file)):
		os.remove(detection_file)

	with open(detection_file, 'wt') as f:
		print('Dumping detection results to', detection_file)
		json.dump(detections, f)


def record_coco_output(model, test_name):
	detection_file = 'benchmark/detections_{}_{}.json'.format(model['name'], test_name)
	coco_out_file = 'benchmark/coco_{}_{}.dat'.format(model['name'], test_name)

	annotations = './dataset/val2017/annotations/instances_val2017.json'
	annotation_types = ['bbox']

	coco_gt = COCO(annotations)
	dataset = coco_gt.loadRes(detection_file)
	coco_eval = COCOeval(coco_gt, dataset, annotation_types[0])
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	out = ''
	for result in coco_eval.stats:
		out += str(result)+'\n'
	print('resulting file content', out)

	if(os.path.exists(coco_out_file)):
		os.remove(coco_out_file)

	with open(coco_out_file, 'wt') as f:
		f.write(out)

def record_test_output(model, time_measurements, detections, test_name):
	record_time(model, time_measurements, test_name)
	record_detections(model, detections, test_name)
	if(test_name == 'dataset'):
		record_coco_output(model, test_name)

def count_objects_in_detection(model, objects):
	obj_count = dict()
	for detection in objects:
		cat = model['classes'][detection['category_id']-1]
		if(cat in obj_count):
			obj_count[cat] += 1
		else:
			obj_count[cat] = 1

	return obj_count

def get_total_counts_per_object(obj_count_per_time, prev_caculated_total=None):
	if(prev_caculated_total == None):
		total_count = dict()
		for time, objs in obj_count_per_time.items():
			for obj in objs.items():
				cat = obj[0]
				if(cat in list(total_count.keys())):
					total_count[cat] += obj[1]
				else:
					total_count[cat] = obj[1]
	else:
		total_count = prev_caculated_total
		last_count_time = list(obj_count_per_time.keys())[-1]#checar
		for cat in obj_count_per_time[last_count_time]:
			if(cat in list(total_count.keys())):
				total_count[cat] += obj_count_per_time[last_count_time][cat]
			else:
				total_count[cat] = obj_count_per_time[last_count_time][cat]

	return total_count

def get_most_frequent_object_names(total_counts_per_object, num_objects):
	if(num_objects > len(list(total_counts_per_object.keys()))):
		num_objects = len(list(total_counts_per_object.keys()))
	sorted_total_counts = sorted(total_counts_per_object.items(), key=operator.itemgetter(1))
	most_frequent = sorted_total_counts[-1:-num_objects-1:-1]
	most_frequent_names = [name_count_tuple[0] for name_count_tuple in most_frequent]
	return most_frequent_names

def get_detection_time_series(obj_count_per_time, obj_names):
	num_intervals = len(obj_count_per_time.keys())
	num_objects = len(obj_names)
	time_series = np.zeros(shape=(num_objects, num_intervals), dtype=np.int32)

	for obj_index, obj in enumerate(obj_names):
		for i, time_index in enumerate(obj_count_per_time.keys()):
			if(obj in obj_count_per_time[time_index]):
				time_series[obj_index, i] = obj_count_per_time[time_index][obj]
			else:
				time_series[obj_index, i] = 0
		
	return time_series

def plot_detection_histogram(total_counts_per_object, fig, axis):
	if(fig == None):
		plt.ion()
		fig, axis = plt.subplots(1, 2, figsize=(15, 10))

	axis[0].bar(list(total_counts_per_object.keys()), list(total_counts_per_object.values()), color='g')
	axis[0].set_title('Total de objetos detectados por categoria')	
	return fig, axis

def plot_detection_series(object_names, time_series, detection_times, fig=None, axis=None, lines=[]):
	num_objects = len(object_names)
	x = detection_times
	if(fig == None):
		plt.ion()
		fig, axis = plt.subplots(1, 2, figsize=(15, 10))

	if(len(lines) == 0):
		for i in range(0, time_series.shape[0]):
			lines.append(axis[1].plot(x, time_series[i,:], label=object_names[i])[0])
			axis[1].set_ylim(bottom=time_series.min(), top=time_series.max())
			axis[1].set_xlim(left=x[0], right=x[-1])
			axis[1].set_title('Detecção dos {} objetos mais frequentes: evolução temporal'.format(num_objects))
			axis[1].legend()

	else:
		axis[1].lines.clear()
		for i in range(0, time_series.shape[0]):
			#if(i < len(lines)):
			#	lines[i].set_xdata(x)
			#	lines[i].set_ydata(time_series[i,:])
			#else:
			#	lines.append(axis[1].plot(x, time_series[i, :],label=object_names[i])[0])

			
			lines.append(axis[1].plot(x, time_series[i, :],label=object_names[i])[0])
			axis[1].set_title('Detecção dos {} objetos mais frequentes: evolução temporal'.format(num_objects))
			axis[1].set_ylim(bottom=time_series.min(), top=time_series.max())
			axis[1].set_xlim(left=x[0], right=x[-1])
			axis[1].legend()

	plt.draw()
	plt.pause(0.001)
	return fig, axis, lines

def get_obj_count_for_current_time(model, detection, starting_time, last_rec_time, obj_count_per_time):
	cur_time = time.time()
	if(cur_time-last_rec_time >= 1):
		video_time = int(cur_time-starting_time)
		obj_count_per_time[video_time] = count_objects_in_detection(model, detection)
		last_rec_time = time.time()

	return starting_time, last_rec_time, obj_count_per_time

def detect_objects_in_region(image, detections, starting_point, ending_point):
	reg_x0, reg_y0 = starting_point
	reg_w = ending_point[0]-reg_x0
	reg_h = ending_point[1]-reg_y0
	num_collisions = 0

	categories_detected = []
	for detection in detections:
		obj_x0, obj_y0, obj_w, obj_h = detection['bbox']
		if(obj_x0 < reg_x0 +reg_w and 
			obj_x0 + obj_w > reg_x0 and
			obj_y0 < reg_y0 + reg_h and
			obj_y0 + obj_h > reg_y0):
			
			rect_coords = (int(obj_x0), int(obj_y0)),(int(obj_x0+obj_w), int(obj_y0+obj_h))
			cv2.rectangle(image, *rect_coords, (255,255,255), thickness=5)
			num_collisions += 1
			categories_detected.append(detection['category_id'])


	print('number of detected collisions', num_collisions)
	return image, categories_detected
		

def detect_from_video(model, video_file=None):
	fig = None
	axis = None
	lines = []
	total_counts_per_object = None

	starting_time = time.time()
	cur_time = time.time()
	last_count_time = time.time()

	time_measurements = []
	detections = []
	obj_count_per_time = dict()

	if(video_file == None):
		capture = cv2.VideoCapture(0)
		test_name = 'webcam'
	else:
		capture = cv2.VideoCapture(video_file)
		test_name = 'video_file'

	while(capture.isOpened()):
		ret, image = capture.read()
		objects, time_elapsed = detect(model, image)
		time_measurements.append(time_elapsed)

		for detection in objects:
			detections.append(detection)

		starting_time, last_count_time, obj_count_per_time = get_obj_count_for_current_time(model, objects, starting_time, 
			last_count_time, obj_count_per_time)

		draw_boxes(model, image, objects)

		region = ((400, 200), (600, 300))
		cv2.rectangle(image, *region, (255,255,0), 5)
		detect_objects_in_region(image, objects, *region)
		
		image = cv2.resize(image, (1000,500))
		cv2.imshow('Detection', image)

		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break
		if(len(time_measurements) % 20 == 0):
			detection_times = list(obj_count_per_time.keys())
			total_counts_per_object = get_total_counts_per_object(obj_count_per_time, total_counts_per_object)
			fig, axis = plot_detection_histogram(total_counts_per_object, fig, axis)
			
			most_frequent_object_names = get_most_frequent_object_names(total_counts_per_object, 4)
			most_frequent_time_series = get_detection_time_series(obj_count_per_time, most_frequent_object_names)
			fig, axis, lines = plot_detection_series(most_frequent_object_names, most_frequent_time_series, detection_times, fig, axis, lines)

		if(len(time_measurements)  > 2000):
			record_test_output(model, time_measurements, detections, test_name)
			break

	capture.release()
	cv2.destroyAllWindows()	

def detect_from_dataset(model, folder):
	if(folder == None):
		print('Please specify the dataset folder')
		exit()
	else:
		files = os.listdir(folder)
		file_num = len(files)
		
		detections = []
		time_measurements = []
		print('Model {} - Dataset accuracy evaluation'.format(model['name']))

		for i, image_file in enumerate(files):
			image = cv2.imread(folder+image_file)
			image_id = int(image_file.strip('0')[:image_file.rfind('.')].strip('.jpg'))
			objects, elapsed = detect(model=model, image=image, image_id=image_id)

			for detection in objects:
				detections.append(detection)
			time_measurements.append(elapsed)

			if(i % 500 == 0):
				print('Progress: {}%'.format((i/file_num)*100))
		
		record_test_output(model, time_measurements, detections, 'dataset')


def parse_commandline():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', dest='model', type=int, required=False, help='Specify model')
	parser.add_argument('-s', dest='source', required=False, help='Specify source type')
	parser.add_argument('-f', dest='file', required=False, help='Specify input file')
	args = vars(parser.parse_args())
	return args

def main():
	args = parse_commandline()
	print(args['model'])	
	print(args['source'])
	print(args['file'])

	if(args['model'] < 4):
		model = load_net(args['model'])

	if(args['source'] == 'webcam'):
		detect_from_video(model, None)
	if(args['source'] == 'video'):
		if(args['file'] == None):
			print('Specify video file with -f')
		else:
			detect_from_video(model, args['file'])	
	elif(args['source'] == 'dataset'):
		detect_from_dataset(model, args['file'])
	else:
		print('Unknown source type. Valid options: webcam, video, dataset')

if __name__ == '__main__':
	main()
