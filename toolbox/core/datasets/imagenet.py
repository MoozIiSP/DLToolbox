import os
import collections
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg

ARCHIVE_META = {
    'imagenet_object_localization':
        ('imagenet_object_localization.tar.gz', '8ea90d1ac9fe7d7591eaf68ee9f0949f'),
    'LOC_sample_submission.csv.zip':
        ('LOC_sample_submission.csv.zip', '81d18fba33690eacceac71b0239f6184'),
    'LOC_synset_mapping.txt.zip':
        ('LOC_synset_mapping.txt.zip', 'cb35635ffd94f38d1e8db360139a8570'),
    'LOC_train_solution.csv.zip':
        ('LOC_train_solution.csv.zip', 'e87df8030fa38f82e948f89a97e3f46d'),
    'LOC_val_solution.csv.zip':
        ('LOC_val_solution.csv.zip', '90c78c3ba27368beb57a68ef5a4ba92a'),
}

SYNSET_MAPPING_FILE = 'LOC_synset_mapping.txt'

class ImageNetK(ImageFolder):

    def __init__(self, root, split="train", **kwargs):

        self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        self.data_dir = os.path.join(self.root, "Data/CLS-LOC")
        wnid_to_classes = load_synset_mapping(self.root)
        super(ImageNetK, self).__init__(self.split_folder, **kwargs)

        # wnid, class labels
        self.wnids = wnid_to_classes.keys()
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    @property
    def split_folder(self):
        return os.path.join(self.data_dir, self.split)


def load_synset_mapping(root, file=None):
    if file is None:
        file = SYNSET_MAPPING_FILE
    file = os.path.join(root, file)

    if os.path.exists(file):
        mapping = {}
        with open(file, 'r') as f:
            for entry in f.readlines():
                wnid, labels = entry.rstrip('\n').split(maxsplit=1)
                labels = [s.strip() for s in labels.split(',')]
                mapping[wnid] = labels
        return mapping
    else:
        msg = ("The mapping file {} is not present in the root directory or is corrupted."
               "This file is provided by ImageNet dataset at Kaggle.")
        raise RuntimeError(msg.format(file, root))


if __name__ == '__main__':
    import time
    from torchvision import transforms

    dataset = ImageNetK(
        '/run/media/mooziisp/仓库/datasets/Kaggle-ILSVRC/ILSVRC',
        split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()])
    )
    print('loaded.')
    print(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8
    )

    ITERITION = 100

    print('read data from disk...')
    record = []
    tik = time.perf_counter()
    for idx, (data, label) in enumerate(loader):
        tok = time.perf_counter()
        record.append(tok - tik)
        if idx == ITERITION:
            break
        elif idx == 0:
            print('batch size: ', len(data))
            print('data: ')
            print(data.shape)
            print(data[0], label[0])
    print('test done.')
    print(f'load data: max {max(record):.02f}, min {min(record):.02f}, avg {sum(record)/ITERITION:.02f}')
