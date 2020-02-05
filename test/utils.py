from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class TestDataSet:

    def __init__(self, path: Union[str, Path], number_of_videos: (int, int, int), min_length: int, max_length: int, overwrite=False,
                        labels=["Swipe Right", "Swipe Left", "Do Nothing"]) -> None:

        self.test_data_path = Path(path)
        test_data_video_path = self._create_directories(overwrite)

        self._create_labels_files(labels)

        video_number = 0
        for mode_number, mode in enumerate(["train", "test", "validation"]):
            data_file = self.test_data_path / f"jester-v1-{mode}.csv"
            for _ in range(0, number_of_videos[mode_number]):
                video_number += 1
                video_dir_path = test_data_video_path / str(video_number)
                self._create_video(max_length, min_length, video_dir_path)
                self._append_to_data_file(data_file, labels, mode, video_number)

    def remove(self):
        self._rm_recursive(self.test_data_path)

    def _append_to_data_file(self, data_file, labels, mode, video_number):
        with data_file.open("a") as f:
            random_label = np.random.choice(labels)
            if mode is not "validation":
                f.write(f"{video_number};{random_label}\n")
            else:
                f.write(f"{video_number}\n")

    def _create_directories(self, overwrite):
        if self.test_data_path.exists() and not overwrite:
            raise FileExistsError(
                f"Directory for test data already exists, please remove it. Path: {self.test_data_path}")
        elif self.test_data_path.exists() and overwrite:
            self.remove()
        self.test_data_path.mkdir()
        test_data_video_path = self.test_data_path / "20bn-jester-v1"
        test_data_video_path.mkdir()
        return test_data_video_path

    def _create_labels_files(self, labels):
        labels_file = self.test_data_path / "jester-v1-labels.csv"
        with labels_file.open("a") as f:
            f.writelines([label + "\n" for label in labels])

    def _create_video(self, max_length, min_length, video_dir_path):
        video_dir_path.mkdir()
        number_of_frames = np.random.randint(min_length, max_length)
        self._create_frames(number_of_frames, video_dir_path)

    def _create_frames(self, number_of_frames, video_dir_path):
        for frame_number in range(number_of_frames):
            frame_file_path = video_dir_path / f"{frame_number:05d}.jpg"
            random_rgb = tuple((np.random.randint(0, 255) for _ in range(3)))
            image = Image.new("RGB", (224, 224), random_rgb)
            image.save(frame_file_path)

    def _rm_recursive(self, pth: Path):
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                self._rm_recursive(child)
        pth.rmdir()


def project_dir() -> Path:
    return Path(__file__).parents[1]


if __name__ == '__main__':
    TestDataSet(project_dir() / "test_data", (10, 3, 3), 2, 5, True)
