#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:24:42 2017

@author: diegothomas
"""

"""Script for P-Avatar."""
import Tkinter as tk
import imp
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )
GPU = imp.load_source('GPUManager', APP_ROOT+'/lib/GPUManager.py')

def main(GPUManager):
    
    ''' Create Menu to load data '''
    M = imp.load_source('Menu', APP_ROOT+'/lib/Menu.py')
    root = tk.Tk()
    menu_app = M.Menu(root)
    menu_app.mainloop()
    
    '''　If a data path has been selected:
        Create the main application and start the loop'''
    if (len(menu_app.filename) > 0):
        A = imp.load_source('App', APP_ROOT+'/lib/Application.py')
        root = tk.Tk()
        app = A.Application(menu_app.filename, GPUManager, root)
        app.mainloop()
    
    return 0


if __name__ == '__main__':
    GPUManager = GPU.GPUManager()
    GPUManager.print_device_info()
    GPUManager.load_kernels()
    main(GPUManager)
    exit(0)