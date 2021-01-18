from detect_people import detect
from detect_people import MIN_CONF,NMS_THRESH, MIN_DISTANCE, MODEL_PATH
from scipy.spatial import distance as dist
import numpy as np
import argparse
import cv2
import os
import csv
import datetime


USE_GPU = False



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([MODEL_PATH,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join(["yolo","yolov3.weights"])
configPath = os.path.sep.join(["yolo","yolov3.cfg"])

print("initializing  yolo...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if USE_GPU:
	print("targetting to cuda for gpu use.......")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

print("accessing video stream.....")
vs = cv2.VideoCapture("vtest.mp4")
writer = None

with open('trial2.csv', 'w', newline='') as file:
	fieldnames = ['TIME','Violations','X','Y','H','W']
	writercsv = csv.DictWriter(file, fieldnames=fieldnames)

	while True:
		t = datetime.datetime.now()
		(grabbed, frame) = vs.read()

		if not grabbed:
			break

		width = 700
		height = 700
		dim = (width, height)
		frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
		results = detect(frame, net, ln,
								personIdx=LABELS.index("person"))


		violate = set()

		if len(results) >=2:
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):

					if D[i, j] < MIN_DISTANCE:

						violate.add(i)
						violate.add(j)
		for (i, (prob, bbox, centroid)) in enumerate(results):

			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			if i in violate:
				color = (0, 0, 255)

				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				cv2.circle(frame, (cX, cY), 5, color, 1)

		(wx,wy,wh,ww)=bbox
		wh = wh - wx
		ww = ww - wy








		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
					cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

		writercsv.writerow({'TIME': t,'Violations':len(violate),'X': wx, 'Y': wy, 'H': wh, 'W': ww})
		print('stats recorded')

		if args["display"] > 0:

			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF


			if key == ord("q"):
				break



		if args["output"] != "" and writer is None:

			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
				(frame.shape[1], frame.shape[0]), True)



		if writer is not None:
			writer.write(frame)
