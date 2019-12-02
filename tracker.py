import cv2
from torchvision import transforms
import pickle as pkl
from PIL import Image
import numpy.linalg as lin

from additional_funtions.detecting.sort import *
from additional_funtions.detecting.models import *
from additional_funtions.detecting.utils import *


class Tracker:
    def __init__(self, filter_list, video=0, resolution=416, conf_threshold=0.6, nms_threshold=0.3):
        self.config_path = 'model_config/yolov3.cfg'
        self.weights_path = 'model_config/yolov3.weights'
        self.class_path = 'model_data/coco.names'

        self.resolution = resolution  # Detection resolution
        self.conf_threshold = conf_threshold  # Confidence Threshold
        self.nms_threshold = nms_threshold  # NMS Threshold

        self.current_index = 0
        self.indices_dict = dict()

        self.vid = cv2.VideoCapture(video)
        self.mot_tracker = Sort()
        self.time_slice = 0

        self.filter_list = filter_list
        self.colors = pkl.load(open("model_data/pallete", "rb"))
        self.classes = load_classes(self.class_path)

        self.model = Darknet(self.config_path, img_size=self.resolution)
        self.model.load_weights(self.weights_path)
        self.model.cuda()
        self.model.eval()

        self.corners = None
        self.criteria = None
        self.chess_ret = None
        self.is_get_chessboard_init = False
        self.inverse_matrix = None
        self.inv_rot_vec_matrix = None
        self.inverse_new_matrix = None

        self.initialize_chessboard()

    def __del__(self):
        cv2.destroyAllWindows()
        pass

    def start_tracking(self):
        time_standard = time.time()

        while True:
            frame = self.get_frame()

            if frame is None:
                print("Error - Wrong frame")
                break
            else:

                if self.is_get_chessboard_init:
                    pass
                else:
                    time_current = time.time()
                    time_difference = time_current - time_standard

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame)
                    detection_for_frame = self.perform_detection(frame_pil)

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_array = np.array(frame_pil)

                    pad_x = max(frame_array.shape[0] - frame_array.shape[1], 0) * (
                                self.resolution / max(frame_array.shape))
                    pad_y = max(frame_array.shape[1] - frame_array.shape[0], 0) * (
                                self.resolution / max(frame_array.shape))
                    unpad_h = self.resolution - pad_y
                    unpad_w = self.resolution - pad_x
                    img_x = frame_array.shape[0]
                    img_y = frame_array.shape[1]

                    if time_difference > self.time_slice:
                        if detection_for_frame is not None:
                            tracked_objects = self.mot_tracker.update(detection_for_frame.cpu())

                            for each_tracked_object in tracked_objects:
                                class_name = self.classes[int(each_tracked_object[5])]

                                if class_name in self.filter_list:
                                    self.draw_rectangle_to_frame(frame, each_tracked_object, (img_x, img_y),
                                                                 (unpad_h, unpad_w), (pad_x, pad_y))
                        time_standard = time.time()

            cv2.imshow('Stream', frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    def get_frame(self):
        ret, frame = self.vid.read()

        if not ret:
            return None
        else:
            return frame

    def perform_detection(self, img):
        ratio = min(self.resolution / img.size[0], self.resolution / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                             transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                             max(int((imh - imw) / 2), 0),
                                                             max(int((imw - imh) / 2), 0)),
                                                            (128, 128, 128)),
                                             transforms.ToTensor(),
                                             ])

        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.conf_threshold, self.nms_threshold)

        return detections[0]

    def get_positions(self, tags=None):
        if tags is None:
            tags = self.filter_list

    def get_real_position(self):
        pass

    def get_predict_position(self):
        pass

    def initialize_chessboard(self):
        tile_size = (31.5, 31.5)  # size cm
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        object_points = np.zeros((7 * 7, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

        frame = self.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        object_points = [object_points]  # make it to double array
        image_points = [corners]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        rvecsMatrix, J = cv2.Rodrigues(rvecs[0])
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        self.inverse_matrix = lin.inv(mtx)
        self.inv_rot_vec_matrix = lin.inv(rvecsMatrix)
        self.inverse_new_matrix = lin.inv(newcameramtx)

        self.chess_ret = ret
        self.corners = corners
        self.criteria = criteria

        self.is_get_chessboard_init = True

    def draw_rectangle_to_frame(self, frame, tracked_object, shapes, unpads, pads):
        unpad_h = unpads[0]
        unpad_w = unpads[1]
        imgX = shapes[0]
        imgY = shapes[1]
        pad_x = pads[0]
        pad_y = pads[1]

        x1, y1, x2, y2, obj_id, cls_pred = tracked_object
        object_id = self.get_index_for_object_id(object_id=str(obj_id))
        class_name = self.classes[int(cls_pred)]

        box_h = int(((y2 - y1) / unpad_h) * imgX)
        box_w = int(((x2 - x1) / unpad_w) * imgY)
        y1 = int(((y1 - pad_y // 2) / unpad_h) * imgX)
        x1 = int(((x1 - pad_x // 2) / unpad_w) * imgY)
        color = self.colors[int(object_id) % len(self.colors)]

        big_rect_start = (x1, y1)
        big_rect_end = (x1 + box_w, y1 + box_h)
        namecard_start = (x1, y1 - 35)
        namecard_end = (x1 + len(class_name) * 19 + 80, y1)
        text_point = (x1, y1 - 10)

        cv2.rectangle(frame, big_rect_start, big_rect_end, color, 4)  # 큰 사각형
        cv2.rectangle(frame, namecard_start, namecard_end, color, -1)  # 이름표
        cv2.putText(frame, class_name + "-" + str(int(object_id)), text_point, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)

    def get_index_for_object_id(self, object_id):
        if object_id in self.indices_dict:
            return self.indices_dict[object_id]
        else:
            self.indices_dict[object_id] = int(self.current_index)
            self.current_index += 1

            return self.indices_dict[object_id]


if __name__ == '__main__':
    person_tracker = Tracker(['person'])
    person_tracker.start_tracking()
