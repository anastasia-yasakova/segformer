import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from pathlib import Path
from typing import Tuple


class Wound(Dataset):
    """
    num_classes: 18
    """
    # explain the purpose of the model
    # where is it, how big it is,
    # give examples of what each of segments are
    # people who are familiar: segmentation
    # medical background: application site, trying to identify different areas in a an image
    # in the wound we are looking for different types of tissues
    # get the story
    CLASSES = ['Boundary','PeriWoundPerimeter','WoundPerimeter','Epithellialization','Granulation','Hypergranulation','NecroticSlough','Eschar','OtherWound','DamagedToeNail','HealthyToeNail','Oedematous','Erythematous','OtherSkinUnbroken','Maceration','Excoriation','OtherSkinBroken','HealthySkin']
    
    PALETTE = torch.tensor([[192, 192, 192],[0, 183, 235],[0, 255, 255],[255, 255, 0],[212, 175, 55],[127, 255, 212],[138, 43, 226],[204, 255, 0],[220, 208, 255],[0, 250, 154],[255, 69, 0],[255, 165, 0],[30, 144, 255],[221, 160, 221],[0, 255, 0],[0, 128, 128],[252, 15, 192],[220, 20, 60]])

    ID2TRAINID = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
                  17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: -1}
    
    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        img_path = Path(root) / 'leftImg8bit' / split
        self.files = list(img_path.rglob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")

        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label.squeeze().numpy()).long()

    def encode(self, label: Tensor) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)
        # for id, trainid in self.ID2TRAINID.items():
        #     label[label == id] = trainid
        # return label

    def decode(self, label: Tensor) -> Tensor:
        return self.PALETTE[label.to(int)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms as T
    from torchvision.utils import make_grid
    from transforms import Compose, RandomResizedCrop, Normalize

    root = 'C:\\Users\\sithu\\Documents\\Datasets\\CityScapes'
    transform = Compose([RandomResizedCrop((1024, 1024)), Normalize()])

    dataset = CityScapes(root, split="train", transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    print('=========================')
    print(image.shape, label.shape)
    print(label.unique())
    label[label==255] = 0
    labels = [dataset.decode(lbl).permute(2, 0, 1) for lbl in label]
    labels = torch.stack(labels)

    inv_normalize = T.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    image = inv_normalize(image)
    image *= 255
    images = torch.vstack([image, labels])
    
    plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
    plt.show()
