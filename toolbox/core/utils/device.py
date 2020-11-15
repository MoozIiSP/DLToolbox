import torch
import torch.backends.cudnn as cudnn


def get_device(logger, index=None, cudnn_benchmark=False):
    """Get device installed on the computer, which is CPU or GPU."""
    if not torch.cuda.is_available():
        logger.info('not found CUDA device and enable CPU mode')
        return torch.device('cpu')

    count = torch.cuda.device_count()
    logger.info('found {:d} CUDA devices and enable GPU mode'.format(count))
    # FIXME Get all devices - but only support one device to train
    devices = []
    for i in range(count):
        dev_prop = torch.cuda.get_device_properties(i)
        tot_mem = dev_prop.total_memory / 1024 / 1024
        logger.info('found device {}: {dev_prop.name}, '
                    'Computing cap. {dev_prop.major}.{dev_prop.minor}, '
                    'Total mem {tot_mem:.2f}MB'.format(
            i, dev_prop=dev_prop, tot_mem=tot_mem))
        devices.append(i)
    if cudnn_benchmark:
        logger.info('enable cudnn.benchmark to improve performance')
        cudnn.benchmark = True
    if len(devices) == 1:
        return devices[0]
    return devices