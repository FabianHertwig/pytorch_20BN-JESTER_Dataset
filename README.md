# Pytorch 20BN-JESTER Dataset
A [Pytorch Dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) to load the 
[20BN-JESTER hand gesture dataset](https://20bn.com/datasets/jester) or datasets that have the same format.

## Usage

Install this package

    pip install git+https://github.com/FabianHertwig/pytorch_20BN-JESTER_Dataset
    
You will also need the video_transforms from this repo:

    pip install git+https://github.com/hassony2/torch_videovision

Code Example:

    from jesterdataset import JesterDataset
    from torch_videovision.videotransforms.volume_transforms import ClipToTensor
    
    dataset = JesterDataset("./jester_data/jester-v1-train.csv", "./jester_data/20bn-jester-v1", 
                            video_transform=ClipToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        self.assertLessEqual(len(sample_batched), 4)
