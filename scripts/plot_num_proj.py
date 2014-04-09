#!/usr/bin/env python
import sys
import pylab
import math
import os
import numpy as np
import string
import matplotlib.pyplot as plt


class Error(Exception):
  """ Base class for exceptions in this module. """
  pass


def calc_match_mean(h, matches, size):
  """ Compute the match percentage per hash """
  
  # Sort the hash
  idx = np.argsort(h)
  h = h[idx]
  matches = matches[idx]

  # Return the mean
  return np.mean(matches[0:size])

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
          description='Show the correspondences between hash and descriptors.',
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('dir', 
          help='Directory where the user has saved several output files of the hash_matching node.')
  parser.add_argument('-s', '--size', type=int, default=20, 
          help='How many images per bucket.')
  
  args = parser.parse_args()

  # Log
  print "Directory: ", args.dir
  print "Bucket Size: ", args.size
  
  files_dir = args.dir
  if (files_dir[-1] != "/"):
    files_dir = files_dir + "/"

  # Loop dir
  proj = []
  matchings = []
  time_h1 = []
  time_h2 = []
  time_h3 = []
  for subdir, dirs, files in os.walk(args.dir):
    for file in files:
      # Detect the reference field
      images = np.genfromtxt(files_dir+file, dtype='str', delimiter=',', usecols=(0,1))
      itm_idx = np.where(images[:,1] != images[0,0])

      # Read the file
      data = pylab.loadtxt(files_dir+file, delimiter=',', usecols=(2,3,4,5,6,7,8,9))
      matches = data[itm_idx[0],0]
      h = data[itm_idx[0],3]

      # Get the number of projections
      file_parts = file.split('_');
      proj.append(int(file_parts[2]))

      # Calculate the matching percentage
      mp = calc_match_mean(h, matches, args.size)
      matchings.append(mp)

      time_h1.append(np.mean(data[:,4])*1000)
      time_h2.append(np.mean(data[:,5])*1000)
      time_h3.append(np.mean(data[:,6])*1000)

  # Convert to numpy
  proj = np.array(proj)
  matchings = np.array(matchings)
  time_h1 = np.array(time_h1)
  time_h2 = np.array(time_h2)
  time_h3 = np.array(time_h3)

  # Sort the hash
  idx = np.argsort(proj)
  proj = proj[idx]
  matchings = matchings[idx]
  time_h1 = time_h1[idx]
  time_h2 = time_h2[idx]
  time_h3 = time_h3[idx]

  # Figure
  width = 0.5 * (proj[1] - proj[0])
  f, (ax) = plt.subplots(1, 1, sharey=True)

  # Bars
  ax.bar(proj, matchings, align='center', width=width)
  ax.set_xlabel("Number of Projections")
  ax.set_ylabel('Mean Feat. Matchings')
  ax.grid(True)

  # Execution time
  ax1 = ax.twinx()
  ax1.plot(proj, time_h1, marker='o', ls='-', color='k', linewidth=2.0, label="Hyperplanes Runtime")
  ax1.plot(proj, time_h2, marker='o', ls='-', color='r', linewidth=2.0, label="Histogram Runtime")
  ax1.plot(proj, time_h3, marker='o', ls='-', color='g', linewidth=2.0, label="Projections Runtime")
  ax1.set_ylabel('Execution time (ms)')
  ax1.legend()

  plt.show()