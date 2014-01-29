#!/usr/bin/env python
import sys
import pylab
import math
import numpy as np
import string
from matplotlib import pyplot

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
          description='Show the correspondences between hash and descriptors',
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('csv_file', 
          help='file with hash matching and descriptor matching')
  
  args = parser.parse_args()

  # Save graph edges file into global
  csv_file = args.csv_file
  data = pylab.loadtxt(csv_file, delimiter=',')

  # Init figure
  fig, ax = pylab.subplots()
  ax.grid(True)
  ax.set_title("Hash vs Descriptors Matching")
  ax.set_xlabel("Hash")
  ax.set_ylabel("Descriptors Matchings")
  ax.plot(data[:,0], data[:,1], marker='o', ls='')
  pylab.xlim(0, 1000000000000)

  pylab.show()