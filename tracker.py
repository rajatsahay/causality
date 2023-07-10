import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = list(self.objects.values())
        object_ids = list(self.objects.keys())

        D = dist.cdist(input_centroids, centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register(centroids[col])

        return self.objects


def get_coords(video_id):
    video = cv2.VideoCapture(video_id)
    tracker = CentroidTracker()

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Perform object detection to obtain bounding boxes
        # and centroids for each detected object

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray)
        boxes = []
        centroids = []
        centroid_list = []
        frame_coords = []
        for keypoint in keypoints:pow
            x = int(keypoint.pt[0] - keypoint.size / 2)
            y = int(keypoint.pt[1] - keypoint.size / 2)
            w = int(keypoint.size)
            h = int(keypoint.size)

            box = (x, y, w, h)
            boxes.append(box)

            centroid = ((x+w)/2, (y+h)/2)
            centroids.append(centroid)

        # Update the centroid tracker
        # with the detected centroids
        objects = tracker.update(centroids)

        for object_id, centroid in objects.items():
            x, y = centroid
            centroid_list.append((object_id, x, y))
        
        frame_coords.append(centroid_list)
            #cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            #cv2.putText(frame, "ID {}".format(object_id),
                        #(int(x) - 10, int(y) - 10),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #cv2.imshow("Tracking", frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    return frame_coords
