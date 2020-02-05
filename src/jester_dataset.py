from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class JesterDataset(Dataset):

    def __init__(self, csv_file, root_dir, video_dir="20bn-jester-v1", file_ending="jpg", transform=None):
        self.file_ending = file_ending
        self.root_dir = root_dir
        self.video_dir_name = video_dir
        self.transform = transform
        self.dataDescription = self._read_csv(csv_file)

    def _read_csv(self, path):
        with open(path, "r") as f:
            return [line.strip().split(";") for line in f]

    def __getitem__(self, index):
        video_id, label = self.dataDescription[index]
        frame_files = Path(self.root_dir / self.video_dir_name / str(video_id)).glob(f"*.{self.file_ending}")
        frames = [Image.open(frame_file) for frame_file in frame_files]

        if self.transform:
            # frames = self.transform(frames) # TODO make sure the same transform is applied to the whole sequence
            frames = [self.transform(frame) for frame in frames]

        return frames, label

    def __len__(self):
        return len(self.dataDescription)
