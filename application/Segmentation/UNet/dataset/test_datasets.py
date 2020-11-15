import unittest

import torch
from custom import CustomSegDataset

class TestCustomSegDataset(unittest.TestCase):

    def test_init(self):
        d = CustomSegDataset(
            images_dir = '/home/aliclotho/GitRepos/unet/data/membrane/train/image')
        self.assertEqual(len(d.images), len(d.masks))

    def test_dataset(self):
        d = CustomSegDataset(
            images_dir = '/home/aliclotho/GitRepos/unet/data/membrane/train/image')
        sample_loader = torch.utils.data.DataLoader(d, batch_size=1,
                                                    shuffle=False, num_workers=2)
        dataiter = iter(sample_loader)
        im, mask = next(dataiter)
        self.assertIs(type(im), torch.Tensor)
        self.assertIs(type(mask), torch.Tensor)
        self.assertEqual(im.shape, mask.shape)


if __name__ == '__main__':
    unittest.main()
    
