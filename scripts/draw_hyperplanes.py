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
  fig1 = pylab.figure()
  ax1 = fig1.gca()
  ax1.grid(True)

  # Load the data
  hp_data = pylab.loadtxt(args.hyperplanes+"_a.txt", delimiter=',')
  desc_data = pylab.loadtxt(args.descriptors+"_a.txt", delimiter=',')
  desc_data_b = pylab.loadtxt(args.descriptors+"_b.txt", delimiter=',')

  # Plot the descriptors
  desc_max = np.amax(desc_data);
  desc_min = np.amin(desc_data);
  ax1.plot(desc_data[:,0], desc_data[:,1], 'g', marker='o', linestyle='None')
  ax1.plot(desc_data_b[:,0], desc_data_b[:,1], 'm', marker='+', linestyle='None')

  # Plot the centroid
  ax1.plot(hp_data[0,0], hp_data[0,1], 'r', marker='o', linestyle='None')  

  # Plot the hyperplane lines
  desc_max = np.amax(desc_data);
  desc_min = np.amin(desc_data);
  x = np.linspace(desc_min, desc_max, num=20)
  for i in range(len(hp_data)-1):
    m = hp_data[i+1,0]
    n = hp_data[i+1,1]
    y = m*x + n
    ax1.plot(x, y, 'b')
    

  pyplot.draw()
  pylab.show()