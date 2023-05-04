import argparse

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


class FaceMesh:
    def __init__(self, config):
        self.config = OmegaConf.load(config)
        self.df = pd.read_csv(self.config.path_csv)

    def normalize(self, landmarks):
        output = []
        for landmark in landmarks:
            landmark = np.array(landmark)
            landmark = (landmark - landmark.min()) / (landmark.max() - landmark.min())
            output.append(landmark)
        return np.array(output)

    def get_facemesh(self, path):
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(
                cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            )
            x, y, z = [], [], []
            result = results.multi_face_landmarks[0]
            for lands in result.landmark:
                x.append(lands.x)
                y.append(lands.y)
                z.append(lands.z)
            landmark = self.normalize([x, y, z])
        return landmark

    def create_dataset(self):
        landmarks = []
        for i in range(len(self.df)):
            landmark = self.get_facemesh(self.df.iloc[i]["filepath"])
            landmarks.append(landmark)
        np.save(self.config.path_skeletal, np.array(landmarks))


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="File path for config file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    scraper = FaceMesh(args.config)
    scraper.create_dataset()
