from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from Inference import HumanPose
import cv2



class Danet():
    def __init__(self):
        self.two_d_model = self.load_model()
        self.image_path_to_infer = "./Examples/example_1.jpg"
        self.size = (640, 480)
        self.tree_d_model = HumanPose(predict_14=False, visualise=True)
        self.infer(self.image_path_to_infer)


    def load_model(self):
        return TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))

    def infer(self, image_path):
        img = cv2.imread(image_path)
        show = cv2.resize(img, (self.size[0], self.size[1]))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        humans = self.two_d_model.inference(show)
        show_image = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
        cv2.imshow("Image", show_image)
        cv2.waitKey(1000)
        show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
        return self.tree_d_model.DoTheInference(list(joints[0].values()))


if __name__ == '__main__':
    Danet()