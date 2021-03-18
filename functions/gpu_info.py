#Copyright (C) 2020 BardzoFajnyZespol

#from tensorflow.python.client import device_lib
#import tensorflow as tf

def get_gpu_name():
    devices = device_lib.list_local_devices()
    gpu_name = 'No detected'
    if tf.test.is_gpu_available():
        for dev in devices:
            if "device:GPU" in dev.name:
                gpu_info = dev.physical_device_desc
                start = 'name: '
                end = ', pci'
                gpu_name = gpu_info[gpu_info.find(start) + len(start):gpu_info.rfind(end)]
    return gpu_name