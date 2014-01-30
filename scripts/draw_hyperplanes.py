#!/usr/bin/env python
import sys
import pylab
import math
import numpy as np
import string
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
          description='Draws the 2D projection of the hyperplanes.',
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('hyperplanes', 
          help='file with the hyperplanes')
  parser.add_argument('descriptors', 
          help='file with the descriptors')
  
  args = parser.parse_args()

  # Prepare the figure
  ax1 = pyplot.figure().gca(projection='3d')
  ax1.grid(True)

  # Load the data
  hp_data = pylab.loadtxt(args.hyperplanes, delimiter=',')
  desc_data = pylab.loadtxt(args.descriptors, delimiter=',')

  # Plot the descriptors
  desc_max = np.amax(desc_data);
  desc_min = np.amin(desc_data);
  ax1.plot(desc_data[:,0], desc_data[:,1], desc_data[:,2], 'g', marker='o', linestyle='None')

  # Plot the centroid
  #ax1.plot(hp_data[0,0], hp_data[0,1], hp_data[0,2], 'r', marker='o', linestyle='None')  

  # Plot the hyperplane lines
  desc_max = np.amax(desc_data);
  desc_min = np.amin(desc_data);
  rg = np.linspace(desc_min, desc_max, num=20)
  for i in range(len(hp_data)-1):
    d = hp_data[i+1,3]
    xx, yy = np.meshgrid(rg, rg)
    z = (-hp_data[i+1,0] * xx - hp_data[i+1,1] * yy - d) * 1. /hp_data[i+1,2]
    ax1.plot_surface(xx, yy, z, 'b')

  pyplot.draw()
  pylab.show()