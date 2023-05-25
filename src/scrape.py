import glob
import os
import shutil
import time
import urllib

import cv2
import mediapipe as mp
import pandas as pd
import requests
from bs4 import BeautifulSoup
from omegaconf import OmegaConf


class Scraper:
    def __init__(self, config: str) -> None:
        """
        初期設定を行う

        Args:
            config (str): 設定ファイルパス
        """

        self.config = OmegaConf.load(config)
        self.base_url = "https://hominis.media/person/"
        if os.path.exists(self.config.path_csv):
            self.df = pd.read_csv(self.config.path_csv)
            self.idx = len(self.df)
        else:
            self.df = pd.DataFrame([], columns=["filepath", "name", "url"])
            self.idx = 0
        os.makedirs(self.config.path_data, exist_ok=True)
        os.makedirs(self.config.path_garbage, exist_ok=True)

    def scrape(self) -> None:
        """スクレイピングを実行する"""

        html = requests.get(self.base_url, timeout=5)
        soup = BeautifulSoup(html.content, "html.parser")
        pages = soup.find_all("input", class_="selectButton")
        before = 0

        for page in pages:
            url = self.base_url + page.get("onclick").split("'")[1].replace(
                "/person/", ""
            )
            html = requests.get(url, timeout=5)
            soup = BeautifulSoup(html.content, "html.parser")
            people = soup.find_all("li", class_="card people")
            for person in people:
                name = person.find("p", class_="name").text
                img_url = (
                    person.find("p", class_="thumbnail")
                    .get("style")
                    .replace("background-image:url('", "")
                    .replace("');", "")
                )
                img_path = os.path.join(self.config.path_data, name + ".png")
                if os.path.exists(img_path):
                    continue
                try:
                    urllib.request.urlretrieve(img_url, img_path)
                    self.df.loc[self.idx] = {
                        "filepath": img_path,
                        "name": name,
                        "url": img_url,
                    }
                    self.idx += 1
                    time.sleep(1)
                except Exception:
                    continue

            imgs = glob.glob(os.path.join(self.config.path_data, "*.png"))
            assert len(imgs) == len(self.df)
            print(f"Get {len(imgs) - before} images")
            before = len(imgs)

        self.df.to_csv(self.config.path_csv, index=False)

    def post_processing(self) -> None:
        """顔ランドマーク取得ができない画像を除去する"""

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for file in glob.glob(os.path.join(self.config.path_data, "*.png")):
                image = cv2.imread(file)
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_face_landmarks:
                    shutil.move(
                        file,
                        os.path.join(self.config.path_garbage, os.path.split(file)[-1]),
                    )
                if len(results.multi_face_landmarks) > 1:
                    shutil.move(
                        file,
                        os.path.join(self.config.path_garbage, os.path.split(file)[-1]),
                    )

        idx = []
        for path in glob.glob(os.path.join(self.config.path_garbage, "*.png")):
            idx.append(
                self.df[
                    self.df["filepath"]
                    == os.path.join(self.config.path_data, os.path.split(path)[-1])
                ].index.values[0]
            )
        self.df = self.df.drop(idx)
        valid_images = glob.glob(os.path.join(self.config.path_data, "*.png"))
        assert len(valid_images) == len(self.df)
        self.df.to_csv(self.config.path_csv, index=False)


if __name__ == "__main__":
    scraper = Scraper("config.yaml")
    scraper.scrape()
    scraper.post_processing()
