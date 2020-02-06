from unittest import TestCase

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from test.utils import TestDataSet, project_dir
from jesterdataset import JesterDataset

from torch_videovision.videotransforms.volume_transforms import ClipToTensor


class TestJesterDataset(TestCase):

    def setUp(self) -> None:
        self.data_dir = project_dir() / "test_data"
        self.min_number_of_frames = 10
        self.max_number_of_frames = 20
        self.train_test_validation_size = (20, 10, 10)
        self.testDataset = TestDataSet(self.data_dir, self.train_test_validation_size, self.min_number_of_frames,
                                       self.max_number_of_frames, True)
        self.testDataset.create()

    def tearDown(self) -> None:
        self.testDataset.remove()

    def test_JesterDataset(self):
        dataset = JesterDataset(self.data_dir / "jester-v1-train.csv", self.data_dir, video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[0])
        for frames, label in dataset:
            self.assertGreaterEqual(len(frames), self.min_number_of_frames)
            self.assertLessEqual(len(frames), self.max_number_of_frames)
            self.assertIn(label, self.testDataset.labels)

    def test_JesterDataset_with_dataloader(self):

        dataset = JesterDataset(self.data_dir / "jester-v1-train.csv", self.data_dir, video_transform=ClipToTensor())

        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=4)

        for i_batch, sample_batched in enumerate(dataloader):
            self.assertLessEqual(len(sample_batched), 4)
