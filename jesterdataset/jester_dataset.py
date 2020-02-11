from enum import Enum

from torch import randint
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class JesterDataset(Dataset):
    class FrameSelectStrategy(Enum):
        FROM_BEGINNING = 0
        FROM_END = 1
        RANDOM = 2

    class FramePadding(Enum):
        REPEAT_END = 0
        REPEAT_BEGINNING = 2

    def __init__(self, csv_file="./jester_data/jester-v1-train.csv", video_dir="./jester_data/20bn-jester-v1", frame_file_ending="jpg", number_of_frames=8,
                 frame_select_strategy=FrameSelectStrategy.RANDOM, frame_padding=FramePadding.REPEAT_END,
                 video_transform=None):
        """
        A pytorch dataset to load the 20BN-JESTER dataset or datasets in the same format.

        Args:
            csv_file: Path to the csv file, describing the videos. In jester this is eg. "1;Swipe Right" where 1 is
                the video folder name
            video_dir: Path to the directory containing the videos
            frame_file_ending: File ending of the video images, eg. "jpg" for file names like "0001.jpg"
            number_of_frames: The number of frames to get from one video.
            frame_select_strategy: When the video has more frames than number_of_frames, then the frame_select_strategy
                is used to select a subset of the frames.
            frame_padding: If the video does not have enough frames, the frame padding strategy is used to fill the frames.
            video_transform:  Videotransforms to apply to the frames. You can not use the usual torchvision.transforms
                as the same transform must be applied to all the frames of one video.
                Checkout https://github.com/hassony2/torch_videovision for video compatible transforms.

        Example:
            from jesterdataset import JesterDataset
            from torch_videovision.videotransforms.volume_transforms import ClipToTensor

            dataset = JesterDataset("./jester_data/jester-v1-train.csv", "./jester_data/20bn-jester-v1",
                                    video_transform=ClipToTensor())
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

            for i_batch, sample_batched in enumerate(dataloader):
                self.assertLessEqual(len(sample_batched), 4)
        """
        self.file_ending = frame_file_ending
        self.video_dir = video_dir
        self.number_of_frames = number_of_frames
        self.frame_select_strategy = frame_select_strategy
        self.frame_padding = frame_padding
        self.video_transform = video_transform
        self.data_description = self._read_csv(csv_file)

    def _read_csv(self, path):
        result = []
        with open(path, "r") as f:
            for line in f:
                line_contents = line.strip().split(";")
                if len(line_contents) == 1:
                    result.append([line_contents[0], None])
                else:
                    result.append(line_contents)
            return result

    def __getitem__(self, index):
        video_id, label = self.data_description[index]
        video_directory = Path(self.video_dir) / str(video_id)
        frame_files = list(Path(video_directory).glob(f"*.{self.file_ending}"))
        if len(frame_files) == 0:
            raise FileNotFoundError(f"Could not find any frames. There should be at least one frame in the directory "
                                    f"{video_directory}")
        frame_files = self._add_padding(frame_files, self.number_of_frames, self.frame_padding)
        frame_files = self._select_frames(frame_files, self.frame_select_strategy, self.number_of_frames)
        frames = [Image.open(frame_file).convert('RGB') for frame_file in frame_files]

        if self.video_transform:
            frames = self.video_transform(frames)

        return frames, label

    def __len__(self):
        return len(self.data_description)

    def _add_padding(self, frame_files, number_of_frames, frame_padding: FramePadding):
        difference = number_of_frames - len(frame_files)
        if difference > 0:
            if frame_padding == self.FramePadding.REPEAT_BEGINNING:
                frame_index_to_repeat = 0
            elif frame_padding == self.FramePadding.REPEAT_END:
                frame_index_to_repeat = -1
            else:
                raise ValueError("Frame Padding Type not supported")

            frame_files += [frame_files[frame_index_to_repeat] for _ in range(difference)]

        return frame_files

    def _select_frames(self, frame_files: list, frame_select_strategy: FrameSelectStrategy, number_of_frames: int):
        if len(frame_files) <= number_of_frames:
            return frame_files
        else:
            if frame_select_strategy == self.FrameSelectStrategy.FROM_BEGINNING:
                return frame_files[:number_of_frames]
            elif frame_select_strategy == self.FrameSelectStrategy.FROM_END:
                return frame_files[-number_of_frames:]
            elif frame_select_strategy == self.FrameSelectStrategy.RANDOM:
                difference = len(frame_files) - number_of_frames
                random_start_index = randint(0, difference, (1,)).item()
                end_index = random_start_index + number_of_frames
                return frame_files[random_start_index:end_index]
            else:
                raise ValueError("FrameSelectStrategy not supported.")







