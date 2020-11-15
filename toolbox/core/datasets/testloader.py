import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import (
    has_file_allowed_extension, default_loader)
from typing import Any, Callable, List, Optional, Tuple, cast


def make_testset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = path
                instances.append(item)
    return instances


class TestDataset(VisionDataset):
    """For evaluate model, it will not identify image belongs to which class 
    and not make its label."""
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(TestDataset, self).__init__(
            root, transform=transform)

        samples = make_testset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in folders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.samples = samples


    def __getitem__(self, index: int) -> Tuple[Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)


def collate_fn(batch):
    return batch


if __name__ == "__main__":
    # TEST CODE
    import tqdm
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)

    args = parser.parse_args()

    testset = TestDataset(args.data, extensions=('png', 'jpg', 'jfif'))
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,
                            collate_fn=collate_fn)

    print(f'size: {len(testset)}')
    for i, (data) in tqdm.tqdm(enumerate(testloader)):
        continue

    print('Done.')
