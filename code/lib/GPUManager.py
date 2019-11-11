#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:13:40 2017
# Last modify: Mon June 12

@author: diegothomas
"""

import pyopencl as cl
import imp
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    
class GPUManager():
    # Constructor
    def __init__(self):
        self.platform = cl.get_platforms()[0]
        self.devices = self.platform.get_devices()
        self.context = cl.Context([self.devices[1]])
        self.queue = cl.CommandQueue(self.context)
        self.programs = {}
        
    #Print info
    def print_device_info(self):
        print ('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
        print('=' * 60)
        print('Platform - Name: ' + self.platform.name)
        print('Platform - Vendor: ' + self.platform.vendor)
        print('Platform - Version: ' + self.platform.version)
        print('Platform - Profile: ' + self.platform.profile)
        
        for device in self.devices:
            print('   ' + '-' * 56)
            print('   Device - Name:  ' + device.name)
            print('   Device - Type:  ' + cl.device_type.to_string(device.type))
            print('   Device - Max Clock Speed:  {0} MHz' .format(device.max_clock_frequency))
            print('   Device - Compute Units:  {0}' .format(device.max_compute_units))
            print('   Device - Local Memory:  {0:.0f} KB' .format(device.local_mem_size/1024.0))
            print('   Device - Constant Memory:  {0:.0f} KB' .format(device.max_constant_buffer_size/1024.0))
            print('   Device - Global Memory:  {0:.0f} GB' .format(device.global_mem_size/1073741824.0))
            print('   Device - Max Buffer/Image Size:  {0:.0f} MB' .format(device.max_mem_alloc_size/1048567.0))
            print('   Device - Max Work Group Size:  {0:.0f}' .format(device.max_work_group_size))
        print('\n')
        
        
    #Load kernels
    def load_kernels(self):
        return