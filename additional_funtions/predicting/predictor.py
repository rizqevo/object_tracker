class Predictor:
    def __init__(self, time_slice):
        self.last_point = None
        self.time_slice = time_slice
        self.objects_last_point = dict()
        self.objects_predicted_point = dict()

    def update_last_point(self, x, y):
        self.last_point = [x, y]

    def remove_point(self, object_id):
        del self.objects_last_point[object_id]
        del self.objects_predicted_point[object_id]

    def get_predicted_point(self, object_id):
        return self.objects_predicted_point[object_id]

    def get_last_point(self, object_id):
        return self.objects_last_point[object_id]

    def predict_next_point(self, current_x, current_y, object_id=0):
        if object_id in self.objects_last_point.keys():
            moved_x = current_x - self.objects_last_point[object_id][0]
            moved_y = current_y - self.objects_last_point[object_id][1]

            self.objects_last_point[object_id] = [current_x, current_y]
            self.objects_predicted_point[object_id] = [current_x + moved_x, current_y + moved_y]

            return self.objects_predicted_point[object_id]
        else:
            self.objects_last_point[object_id] = [current_x, current_y]

            return current_x, current_y


class PredictorFill:
    def __init__(self, time_slice, frame_slice=4):
        self.last_point = None
        self.time_slice = time_slice
        self.frame_slice = frame_slice

    def update_last_point(self, x, y):
        self.last_point = [x, y]

    def get_moved_distance(self, current_x, current_y):
        if self.last_point is None:
            self.last_point = [current_x, current_y]
            return [0, 0]
        else:
            print(self.last_point[0], current_x)
            return [abs(self.last_point[0]-current_x), abs(self.last_point[1]-current_y)]

    def fill_with_last_point(self, current_x, current_y):
        if self.last_point is None:
            self.last_point = [current_x, current_y]
            return []
        else:
            current_point = [current_x, current_y]

            moved_x = current_point[0] - self.last_point[0]
            moved_y = current_point[1] - self.last_point[1]

            moved_x_divided = int(moved_x / self.frame_slice)
            moved_y_divided = int(moved_y / self.frame_slice)

            fill_list = []
            for i in range(self.frame_slice):
                x = self.last_point[0] + moved_x_divided * (i + 1)
                y = self.last_point[1] + moved_y_divided * (i + 1)
                fill_list.append([x, y])

            return fill_list
