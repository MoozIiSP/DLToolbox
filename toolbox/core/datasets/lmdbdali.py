import os
import torch
import numpy as np
from torch.utils import data

from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


DUMMY_LABEL = -1


class CaffePipeline(Pipeline):
    def __init__(self, db_folder, batch_size, num_threads, device_id):
        super(CaffePipeline, self).__init__(batch_size,
                                            num_threads,
                                            device_id)
        self.input = ops.CaffeReader(path=db_folder)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu",
                                 image_type=types.RGB,
                                 resize_shorter=224)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        output = self.resize(images)
        output = self.cmnp(output, crop_pos_x=self.uniform(),
                           crop_pos_y=self.uniform())
        return [output, self.labels]

    def iter_setup(self):
        pass


class Caffe2Pipeline(Pipeline):
    def __init__(self, db_folder, batch_size, num_threads, device_id):
        super(Caffe2Pipeline, self).__init__(batch_size,
                                         num_threads,
                                         device_id)
        self.input = ops.Caffe2Reader(path = db_folder)
        self.decode= ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (224, 224),
                                            image_type = types.RGB,
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        outputs = self.cmnp(images, crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform())
        return (outputs, self.labels)

    def iter_setup(self):
        pass


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from torchvision import transforms

    pipe = CaffePipeline(
        db_folder='/run/media/mooziisp/仓库/datasets/Kaggle-ILSVRC/ILSVRC/Data/CLS-LOC/val-lmdb',
        batch_size=32,
        num_threads=8,
        device_id=0
    )

    pipe.build()

    loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
    print('loaded.')

    # ITERITION = 200

    print(f'read data from disk...')
    record = []
    iter = 0
    tik = time.perf_counter()
    for idx, data in tqdm(enumerate(loader)):
        tok = time.perf_counter()
        record.append(tok - tik)
        # if idx == ITERITION:
        #     break
        if idx == 0:
            print('batch size: ', len(data))
            print('data: ')
            print(data[0]["data"].shape)
            print(data[0]["data"], data[0]["label"])
        iter += 1
    print('test done.')
    print(f'load data: max {max(record):.02f}, min {min(record):.02f}, avg {sum(record) / iter:.02f}')
