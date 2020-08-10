#!/usr/bin/env python
import os
import glob

files = glob.glob('./CarND-Behavioral-Cloning-P3/data/IMG/*')

for f in files:
    os.remove(f)
    #print(f)