import os
import torch
from pynvml import *

# Multi GPU
#print(f"GPUs: {torch.cuda.device_count()}")
#for id in range(torch.cuda.device_count()):
#    print(f"Device: {torch.cuda.get_device_name(id)}")
#    print(f"Capability: {torch.cuda.get_device_capability(id)}")
#    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(id)}")
#    print(f"Currect Memory Allocated: {torch.cuda.memory_allocated(id)}")

def init():
    """init of import package"""
    nvmlInit()
    print(f"Driver Version: {nvmlSystemGetDriverVersion().decode('utf-8')}")
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        print(f"Device {i}:\t{nvmlDeviceGetName(handle).decode('utf-8')}")
        print(f"  | Mem Usage:\ttotal {bytefmt(meminfo.total)}, free {bytefmt(meminfo.free)}, used {bytefmt(meminfo.used)}")
        print(f"  | PCIe Supp:\tPCIe{nvmlDeviceGetMaxPcieLinkGeneration(handle)}x{nvmlDeviceGetMaxPcieLinkWidth(handle)}")
        print(f"  | Power    :\t{nvmlDeviceGetPowerUsage(handle)/1000:.2f}/{nvmlDeviceGetPowerManagementLimit(handle)/1000:.2f}")
        print(f"  | Temp     :\t{nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)}C")
        print(f"  | Util     :\t{nvmlDeviceGetUtilizationRates(handle)}")
        print(f"  | Processes:\t")
        for p in nvmlDeviceGetComputeRunningProcesses(handle):
            print(f"     | pid {p.pid} used {bytefmt(p.usedGpuMemory)}")
        #nvmlDeviceGetIndex


def bytefmt(size):
    """shorten byte length.

    Args:
      size (int): bytes of object.

    Returns:
      res (str): format string
    """
    fmt = ['B', 'KB', 'MB', 'GB', 'PB']
    res = ''
    for unit in fmt:
        res = f"{size:.2f}{unit}"
        if size < 1024:
            break
        size /= 1024
    return res


def device_meminfo(device):
    """get memory information of the device

    Args:
      device (int): index of device

    Returns:
      info (str): memory information
    """
    handle = nvmlDeviceGetHandleByIndex(device)
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    return meminfo, f'device {device}: mem total {bytefmt(meminfo.total)}, free {bytefmt(meminfo.free)}, used {bytefmt(meminfo.used)}'


def process_meminfo(device):
    """get memory information of processes that run on the device,
    matching currect program."""
    pids = []
    handle = nvmlDeviceGetHandleByIndex(device)
    print(f'device {device}:')
    for p in nvmlDeviceGetComputeRunningProcesses(handle):
        if os.getpid() == p.pid:
            pids.append((p, f"  | pid {p.pid} used {bytefmt(p.usedGpuMemory)}"))
    return pids


def shutdown():
    """Wrapper of nvmlShutdown"""
    nvmlShutdown()


init()
