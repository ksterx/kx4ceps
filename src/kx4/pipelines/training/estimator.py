#%%
import glob
from tkinter.filedialog import askdirectory

import cv2
import numpy as np
import torch
from mlflow import pytorch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from preprocessing.transformer import ScaleTransformer


#%%
class Estimator:
    """Estimate the 3D pose of surgical tools from the trained model.
    """
    def __init__(self, mode, ds_num, transform, run_id=None):
        """

        Args:
            mode (bool): Real-time->0, Movie->1
            ds_num (int): Dataset number.
            transform (object): The way how to transform images.
            run_id (str, optional): Mlflow experiment id. Defaults to None.
        """

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transform
        self.scaler = ScaleTransformer(ds_num)
        self.mode = mode

        if run_id is None:
            run_id = askdirectory(initialdir="mlruns")

        self.model = pytorch.load_model("mlruns/2/" + run_id +
                                        "/artifacts/model")

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, src):
        with torch.no_grad():
            if not self.mode:
                # TODO: np.ndarray -> PILimage
                pred = self.model()

            else:
                img = self.transform(src).unsqueeze(0).to(self.device)
                pred = self.model(img).detach().cpu()
                pred = self.scaler.inverse_transform(pred)

            return pred


#             cv2.drawMarker(im, (x_2d, y_2d),
#                            color=(255, 255, 0),
#                            markerType=cv2.MARKER_TILTED_CROSS,
#                            thickness=3)

#             txt0 = 'x: {:.2f} mm'.format(p[0][0])
#             txt1 = 'y: {:.2f} mm'.format(p[0][1])
#             txt2 = 'z: {:.2f} mm'.format(p[0][2])
#             txt3 = 'joint: {:.2f} deg'.format(p[2])

#             im = cv2.putText(im, txt0, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (
#                 255,
#                 255,
#                 0,
#             ), 2)
#             im = cv2.putText(im, txt1, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (
#                 255,
#                 255,
#                 0,
#             ), 2)
#             im = cv2.putText(im, txt2, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (
#                 255,
#                 255,
#                 0,
#             ), 2)
#             im = cv2.putText(im, txt3, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (
#                 255,
#                 255,
#                 0,
#             ), 2)

#             cv2.imshow('image', im)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
# -

run_id = "f1fd2eea388148da8bc7ea3bed378b7a"

# + active=""
#
# img_p = "Database/ds_024/train/img_00001.jpg"
# from PIL import Image
#
# im = Image.open(img_p)
#
# estimator = Estimator(mode=2, ds_num=24, run_id=run_id, transform=t)
# estimator(im)
# -

cmean = [0.485, 0.456, 0.406]
cstd = [0.229, 0.224, 0.225]
resize_shape = (224, 224)
transform = Compose([Resize(resize_shape), ToTensor(), Normalize(cmean, cstd)])


class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        with open("./evaluation/camera_config/camera_config.yaml") as f:
            cam_config = yaml.load(f, Loader=yaml.SafeLoader)
        w = cam_config["image_size"]["w"]
        h = cam_config["image_size"]["h"]
        self.mtx = np.load("evaluation/camera_config/intrinsic_parameter.npy")
        self.dist = np.load(
            "evaluation/camera_config/distortion_parameter.npy")
        self.cam_param, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h))

    def get_frame(self):
        ret, frame = self.video.read()
        if ret:
            frame = cv2.undistort(frame, self.mtx, self.dist, None,
                                  self.cam_param)
            return frame
        else:
            return None

    def terminate(self):
        self.video.release()


def main():
    print("""
    --- Estimation Mode ---
    0: Real-time
    1: Movie (Please convert the video into the images (with a software such as ffmpeg), and they must be on './evaluation/eval_dataset')
    """)
    mode = int(input("Mode: "))
    run_id = input("Run ID: ")
    ds_num = int(input("Dataset: #"))
    if not mode:
        print("Initializing camera...")
        cam = Camera()
        print("Loading model...")
        estimator = Estimator(mode=0, run_id=run_id)

        while True:
            frame = cam.get_frame()
            if not frame:
                print("No signal")
            else:
                out = estimator(frame)
                print(out)

        print("""
        --- Operation ---
        
        """)
        open_video, img = cam.get_img()

    else:  # Movie
        eval_dir = "./evaluation/eval_dataset/*.jpg"
        print("Loading model...")
        estimator = Estimator(
            mode=1,
            ds_num=ds_num,
            transform=transform,
            run_id=run_id,
        )

        for img_path in glob.glob(eval_dir):
            img = Image.open(img_path)
            out = estimator(img)
            print(out)

        pass


if __name__ == '__main__':
    main()
