import os
import sys

from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, BASE_DIR + '/deepsort/')

import cv2
import time
import numpy as np
import tensorflow as tf
import core.utils as utils
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from tensorflow.python.saved_model import tag_constants
from tools import generate_detections as gdet
from collections import defaultdict
from typing import NamedTuple

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from .models import Image
from django.core.files.base import ContentFile


INPUT_SHAPE = (416 ,416)
IMAGE_SIZE_LIMIT = 50
THREASHOLD = 1.15


class Frame(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    person_number: str
    array: np.array


class VideoSaving:
    def __init__(self, vid, path, FORMAT='XVID'):
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FORMAT)
        self.out = cv2.VideoWriter(path, codec, fps, (width, height))

    def save_video(self, result):
        self.out.write(result)


def normalize_image(data):
    image_data = cv2.resize(data, INPUT_SHAPE)
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    return image_data


def convert_numpy_and_select_range(data, num_objects):
    data = data.numpy()[0]
    data = data[0:int(num_objects)]

    return data


def is_image_big_enough(coordinates):
    x, w, y, h = coordinates
    return h - w >= IMAGE_SIZE_LIMIT and  y - x >= IMAGE_SIZE_LIMIT


def get_coordinates(frame):
    return (frame.x, frame.y, frame.w, frame.h)


def get_absolute(coor):
    return abs(coor[1]-coor[0])


def is_frame_different_with_prev_frame(frame, prev_frame, accu_num) -> bool:
    values = [(a, b) for a, b in zip(get_coordinates(frame), get_coordinates(prev_frame))]
    diff = sum(map(get_absolute, values))
    print(THREASHOLD ** accu_num, diff)
    if diff >= THREASHOLD ** accu_num:
        return True
    
    return False


def save_image_in_database(frame, video_model, cnt):
    if frame.array.any() and is_image_big_enough(frame):
        _, buf = cv2.imencode('.jpg', frame.array)
        content = ContentFile(buf.tobytes())
        image = Image(video=video_model, person=frame.person_number, coordinates=get_coordinates(frame))
        image.file.save(f"{cnt}.jpg", content)


def ObjectTracking(video, video_model, per_frame, output_video='1.avi', show_info=True, show_vid=False):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    

    number_of_people = 0
    people_images = defaultdict(int)
    output_path = os.path.join(BASE_DIR + '/deepsort/outputs/', output_video)
    
    # initialize deep sort
    model_filename = 'deepsort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    video_path = video

    saved_model_loaded = tf.saved_model.load('deepsort/checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    # _video = VideoSaving(vid, output_path)

    frame_num = 0
    cnt = 0
    dup_cnt = 0
    prev_frame = None

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            original_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_num +=1

        if frame_num % per_frame != 0:
            print('Frame #: ', frame_num)
            continue

        image_data = normalize_image(frame)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = convert_numpy_and_select_range(boxes, num_objects)
        scores = convert_numpy_and_select_range(scores, num_objects)
        classes = convert_numpy_and_select_range(classes, num_objects)

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names('deepsort/data/coco.names')

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        names = np.array(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            x, w, y, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            target_frame = Frame(x, y, w, h, str(track.track_id), original_frame[w:h, x:y, :])
            class_name = track.get_class()

            people_images[target_frame.person_number] += 1

            if int(target_frame.person_number) not in people_images:
                number_of_people += 1
            
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            
            if prev_frame and is_frame_different_with_prev_frame(target_frame, prev_frame, people_images[target_frame.person_number]):
                save_image_in_database(target_frame, video, str(people_images[target_frame.person_number]))
                dup_cnt = 0
            elif prev_frame and dup_cnt < 10:
                save_image_in_database(target_frame, video, str(people_images[target_frame.person_number]))
                dup_cnt += 1

            prev_frame = target_frame
            save_image_in_database(original_frame[w:h, x:y, :], track.track_id, video_model, people_images[track.track_id], (x, w, y, h))

            cv2.rectangle(frame, (x, w), (y, h), color, 2)
            cv2.rectangle(frame, (x, w-30), (y+(len(class_name)+len(str(track.track_id)))*17, w), color, -1)
            cv2.putText(frame, class_name + "-" + str(number_of_people), (x, w-10), 0, 0.75, (255, 255, 255), 2)
            
            # if enable info flag then print details about each track
            if show_info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (x, w, y, h)))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if show_vid:
            cv2.imshow("Output Video", result)

        # _video.save_video(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
