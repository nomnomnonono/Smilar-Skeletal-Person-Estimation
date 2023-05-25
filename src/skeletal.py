import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


class FaceMesh:
    def __init__(self, config: str) -> None:
        """
        初期設定を行う

        Args:
            config (str): 設定ファイルパス
        """

        self.config = OmegaConf.load(config)
        self.df = pd.read_csv(self.config.path_csv)
        if os.path.exists(self.config.path_skeletal):
            self.reference = np.load(self.config.path_skeletal)

    def create_dataset(self) -> None:
        """フェイスメッシュデータセットの作成を行う"""

        landmarks = []
        for i in range(len(self.df)):
            landmark = self.get_facemesh(self.df.iloc[i]["filepath"])
            landmarks.append(landmark)
        np.save(self.config.path_skeletal, np.array(landmarks))

    def normalize(self, landmarks: list[list[float]]) -> np.ndarray:
        """
        顔ランドマークの正規化を行う

        Args:
            landmarks (list[list[float]]): 顔ランドマーク

        Returns:
            np.ndarray: 正規化後の顔ランドマーク
        """

        output = []
        for landmark in landmarks:
            landmark = np.array(landmark)
            landmark = (landmark - landmark.min()) / (landmark.max() - landmark.min())
            output.append(landmark)
        return np.array(output)

    def get_facemesh(self, path: str) -> np.ndarray:
        """
        フェイスメッシュを獲得する

        Args:
            path (str): 画像ファイルパス

        Returns:
            np.ndarray: ファイスメッシュのランドマーク位置
        """

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

    def estimate_similar_person(self, path: str, topK: str) -> pd.DataFrame:
        """
        類似骨格人物を検索する

        Args:
            path (str): 画像ファイルパス
            topK (str): 上位何件を取得するか

        Returns:
            pd.DataFrame: 検索結果
        """

        facemesh = self.get_facemesh(path)
        diff = abs(self.reference - facemesh).mean((1, 2))
        rank = np.argsort(diff)[0 : int(topK)]
        top = self.df.iloc[rank]
        return top.drop("filepath", axis=1)


if __name__ == "__main__":
    scraper = FaceMesh("config.yaml")
    scraper.create_dataset()
