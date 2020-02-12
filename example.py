from torch.utils.data.dataloader import DataLoader
from jesterdataset import JesterDataset
from torchvideotransforms.volume_transforms import ClipToTensor

dataset = JesterDataset("./jester_data/jester-v1-train.csv", "./jester_data/20bn-jester-v1",
                        video_transform=ClipToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    batch, label = sample_batched
    assert(len(batch) <= 4)
    print(f"Batch number {i_batch} has a batch size of {len(batch)}")
