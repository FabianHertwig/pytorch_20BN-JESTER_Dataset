from unittest import TestCase
from torch.utils.data.dataloader import DataLoader
from torch_videovision.videotransforms.volume_transforms import ClipToTensor

from test.utils import TestDataSet, project_dir
from jesterdataset import JesterDataset


class TestJesterDataset(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = project_dir() / "test_data"
        cls.min_number_of_frames = 10
        cls.max_number_of_frames = 20
        cls.train_test_validation_size = (20, 10, 10)
        cls.testDataset = TestDataSet(cls.data_dir, cls.train_test_validation_size, cls.min_number_of_frames,
                                      cls.max_number_of_frames, True)
        cls.testDataset.create()

        cls.train_csv_file = cls.data_dir / "jester-v1-train.csv"
        cls.validation_csv_file = cls.data_dir / "jester-v1-validation.csv"
        cls.video_dir = cls.data_dir / "20bn-jester-v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.testDataset.remove()

    def test_JesterDataset_without_padding(self):
        number_of_frames = 10
        self.assertLessEqual(number_of_frames, self.min_number_of_frames)
        dataset = JesterDataset(self.train_csv_file, self.video_dir, number_of_frames=number_of_frames,
                                video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[0])
        for frames, label in dataset:
            self.assertEqual(len(frames), number_of_frames)
            self.assertIn(label, self.testDataset.labels)

    def test_JesterDataset_with_validation_set(self):
        number_of_frames = 10
        self.assertLessEqual(number_of_frames, self.min_number_of_frames)
        dataset = JesterDataset(self.validation_csv_file, self.video_dir, number_of_frames=number_of_frames,
                                video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[2])
        for frames, label in dataset:
            self.assertEqual(len(frames), number_of_frames)
            self.assertIsNone(label)

    def test_JesterDataset_with_padding(self):
        number_of_frames = 15
        self.assertGreater(number_of_frames, self.min_number_of_frames)
        dataset = JesterDataset(self.train_csv_file, self.video_dir, number_of_frames=number_of_frames,
                                video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[0])
        for frames, label in dataset:
            self.assertEqual(len(frames), number_of_frames)
            self.assertIn(label, self.testDataset.labels)

    def test_JesterDataset_with_random_frame_select(self):
        number_of_frames = 5
        self.assertLess(number_of_frames, self.min_number_of_frames)
        dataset = JesterDataset(self.train_csv_file, self.video_dir, number_of_frames=number_of_frames,
                                frame_select_strategy=JesterDataset.FrameSelectStrategy.RANDOM, video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[0])
        for frames, label in dataset:
            self.assertEqual(len(frames), number_of_frames)
            self.assertIn(label, self.testDataset.labels)

    def test_JesterDataset_with_frame_select_from_beginning(self):
        number_of_frames = 5
        self.assertLess(number_of_frames, self.min_number_of_frames)
        dataset = JesterDataset(self.train_csv_file, self.video_dir, number_of_frames=number_of_frames,
                                frame_select_strategy=JesterDataset.FrameSelectStrategy.FROM_BEGINNING,
                                video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[0])
        for frames, label in dataset:
            self.assertEqual(len(frames), number_of_frames)
            self.assertIn(label, self.testDataset.labels)

    def test_JesterDataset_with_frame_select_from_end(self):
        number_of_frames = 5
        self.assertLess(number_of_frames, self.min_number_of_frames)
        dataset = JesterDataset(self.train_csv_file, self.video_dir, number_of_frames=number_of_frames,
                                frame_select_strategy=JesterDataset.FrameSelectStrategy.FROM_END, video_transform=None)
        self.assertEqual(len(dataset), self.train_test_validation_size[0])
        for frames, label in dataset:
            self.assertEqual(len(frames), number_of_frames)
            self.assertIn(label, self.testDataset.labels)

    def test_JesterDataset_with_dataloader(self):

        dataset = JesterDataset(self.train_csv_file, self.video_dir, video_transform=ClipToTensor())

        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=4)

        for i_batch, sample_batched in enumerate(dataloader):
            self.assertLessEqual(len(sample_batched), 4)
