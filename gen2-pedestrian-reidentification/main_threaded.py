import argparse
import queue
import threading
from pathlib import Path

import cv2
import depthai
import numpy as np
from imutils.video import FPS

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug
camera = not args.video

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is False and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")

def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)
    #return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    if camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(544, 320)
        cam.setFps(4)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Person Detection Neural Network...")
    if camera:
        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013.blob").resolve().absolute()))
        if not camera:
            detection_nn.input.setBlocking(False)
        detection_nn_xout = pipeline.createXLinkOut()
        detection_nn_xout.setStreamName("detection_nn")
        detection_nn.out.link(detection_nn_xout.input)
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013.blob").resolve().absolute()))
        if not camera:
            detection_nn.input.setBlocking(False)
        detection_nn_xout = pipeline.createXLinkOut()
        detection_nn_xout.setStreamName("detection_nn")
        detection_in.out.link(detection_nn.input)
        detection_nn.out.link(detection_nn_xout.input)

    # NeuralNetwork
    print("Creating Person Reidentification Neural Network...")
    reid_in = pipeline.createXLinkIn()
    reid_in.setStreamName("reid_in")
    reid_nn = pipeline.createNeuralNetwork()
    reid_nn.setBlobPath(str(Path("models/person-reidentification-retail-0031.blob").resolve().absolute()))
    if not camera:
        reid_nn.input.setBlocking(False)
    reid_nn_xout = pipeline.createXLinkOut()
    reid_nn_xout.setStreamName("reid_nn")
    reid_in.out.link(reid_nn.input)
    reid_nn.out.link(reid_nn_xout.input)

    print("Pipeline created.")
    return pipeline


class Main:

    def __init__(self):
        self.device = depthai.Device(create_pipeline())
        print("Starting pipeline...")
        self.device.startPipeline()
        if camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)
        else:
            self.cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
            self.detection_in = self.device.getInputQueue("detection_in")
        self.reid_in = self.device.getInputQueue("reid_in")

        self.bboxes = []
        self.results = {}
        self.results_path = {}
        self.reid_bbox_q = queue.Queue()
        self.next_id = 0

        self.fps = FPS()
        self.fps.start()
        self.updateFrame = False
        self.updateFrameCount = 0
        self.boxCount = 0

    def det_thread(self):
        detection_nn = self.device.getOutputQueue("detection_nn")
        while True:
            bboxes = np.array(detection_nn.get().getFirstLayerFp16())
            bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

            self.boxCount = len(bboxes)
            self.updateFrameCount = 0
            for raw_bbox in bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                nn_data = depthai.NNData()
                nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                self.reid_in.send(nn_data)
                self.reid_bbox_q.put(bbox)
            self.updateFrame = True


    def reid_thread(self):
        reid_nn = self.device.getOutputQueue("reid_nn")
        while True:
            reid_result = reid_nn.get().getFirstLayerFp16()
            bbox = self.reid_bbox_q.get()

            for person_id in self.results:
                dist = cos_dist(reid_result, self.results[person_id])
                if dist > 0.5:
                    result_id = person_id
                    self.results[person_id] = reid_result
                    break
            else:
                result_id = self.next_id
                self.results[result_id] = reid_result
                self.results_path[result_id] = []
                self.next_id += 1

            cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
            x = (bbox[0] + bbox[2]) // 2
            y = (bbox[1] + bbox[3]) // 2
            self.results_path[result_id].append([x, y])
            cv2.putText(self.debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
            if len(self.results_path[result_id]) > 1:
                cv2.polylines(self.debug_frame, [np.array(self.results_path[result_id], dtype=np.int32)], False,
                              (255, 0, 0), 2)
            self.updateFrameCount = self.updateFrameCount + 1

    def should_run(self):
        return True if camera else self.cap.isOpened()

    def convert_cam(self, data):
        return np.array(data).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)

    def get_frame(self):
        if camera:
            return True, np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(
                np.uint8)
        else:
            return self.cap.read()

    def run(self):
        threading.Thread(target=self.det_thread, daemon=True).start()
        threading.Thread(target=self.reid_thread, daemon=True).start()
        first = True
        while self.should_run():
            read_correctly, self.frame = self.get_frame()
            if not camera:
                nn_data = depthai.NNData()
                nn_data.setLayer("input", to_planar(self.frame, (544, 320)))
                self.detection_in.send(nn_data)
            if first:
                self.debug_frame = self.frame.copy()
                first = False
            self.fps.update()

            if not read_correctly:
                break

            if self.updateFrame and self.updateFrameCount == self.boxCount:
                self.updateFrame = False
                aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
                cv2.imshow("Camera_view", cv2.resize(self.debug_frame, (int(900),  int(900 / aspect_ratio))))
                self.debug_frame = self.frame.copy()

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        self.fps.stop()
        print("FPS: {:.2f}".format(self.fps.fps()))
        self.cap.release()

Main().run()
