#!/usr/bin/env python3

import time
import random
import threading

cargo = []


class Producer(threading.Thread):
    def __init__(self, threadId, desc, resource, batch_size):
        super(Producer, self).__init__()
        self.threadId = threadId
        self.desc = desc
        self.resource = resource
        self.batch_size = batch_size
    def run(self):
        print(f'{self.desc} {self.threadId} start to produce from resource.')
        while self.resource:
            print(f'cargo cap. {len(cargo)}', end='')
            time.sleep(5)       # 25.6/s
            for i in range(self.batch_size):
                if self.resource:
                    cargo.append(self.resource.pop())
            print(f' -> {len(cargo)}')
        print(f'{self.desc} {self.threadId} stop to produce due to lack of resource.')

class Consumer(threading.Thread):
    def __init__(self, threadId, desc, batch_size, epoch):
        super(Consumer, self).__init__()
        self.threadId = threadId
        self.desc = desc
        self.batch_size = batch_size
        self.epoch = epoch
    def run(self):
        print(f'{self.desc} {self.threadId} start to consumer from cargo.')
        while dataset:
            if cargo:
                print(f'cargo cap. {len(cargo)}', end='')
                time.sleep(0.05) # 2560/s
                for i in range(self.batch_size):
                    cargo.pop()
                print(f' -> {len(cargo)}')
            else:
                print(f'cargo no content, waiting.')
                time.sleep(5)
        print(f'{self.desc} {self.threadId} stop to consumer from cargo.')


dataset = [i for i in range(5120)]
print('dataset len: ', len(dataset))

workers = [Producer(i, 'produce worker', dataset, 128) for i in range(16)]
for w in workers:
    w.start()

processers = [Consumer(i, 'processer', 128, 10) for i in range(1)]
for p in processers:
    p.start()

print('all thread start finished.')

for w in workers:
    w.join()
for p in processers:
    p.join()

print('main thread finished.')
