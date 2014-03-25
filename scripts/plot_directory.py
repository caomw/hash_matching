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


def calc_match_percentage(h, matches, thresh, size):
  """ Compute the match percentage per hash """
  
  # Sort the hash
  idx = np.argsort(h)
  h = h[idx]
  matches = matches[idx]

  matches_percentage = []
  sum_c_matches = 0
  sum_g_matches = 0
  c = 0
  n = 0
  for i in range(len(matches)):
    sum_c_matches = sum_c_matches + 1
    if (matches[i] >= thresh):
      sum_g_matches = sum_g_matches + 1

    # Reset the counter if needed
    if (c >= size):
      matches_percentage.append(100*sum_g_matches/sum_c_matches)
      sum_c_matches = 0
      sum_g_matches = 0
      c = 0;
      n = n + 1

    # Increase the counter
    c = c + 1

  # Convert to numpy and return
  return np.array(matches_percentage)


def calc_match_mean(h, matches, thresh, size):
  """ Compute the match mean per hash """
  
  # Sort the hash
  idx = np.argsort(h)
  h = h[idx]
  matches = matches[idx]

  matches_mean = []
  for i in range(len(matches)):
    end = i*size+size
    if (end > len(matches)-1):
      end = len(matches)-1
    matches_mean.append(np.mean(matches[i*size: end]))

  # Convert to numpy and return
  matches_mean = np.array(matches_mean)
  matches_mean = matches_mean[~np.isnan(matches_mean)]
  return matches_mean[0:-1]


def get_axes(x, y):
  "Build the axes for the barchart"

  # Get the null indices
  idx = []
  for i in range(len(y)):
    if (y[i] != 0):
      idx.append(i)

  idx = np.array(idx)
  x = range(1, len(x[idx])+1)
  y = y[idx]
  return x, y



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
          description='Show the correspondences between hash and descriptors.',
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('dir', 
          help='Directory where the user has saved several output files of the hash_matching node.')
  parser.add_argument('-s', '--size', type=int, default=40, 
          help='How many images per bucket.')
  parser.add_argument('-t', '--thresh', type=int, default=300, 
          help='The matching threshold to considerate an image as a success or not.')
  
  args = parser.parse_args()

  # Log
  print "Directory: ", args.dir
  print "Bucket Size: ", args.size
  print "Descriptor Matching Threshold: ", args.thresh

  files_dir = args.dir
  if (files_dir[-1] != "/"):
    files_dir = files_dir + "/"

  # Loop dir
  matches = np.array([])
  h1 = np.array([])
  h2 = np.array([])
  h3 = np.array([])
  for subdir, dirs, files in os.walk(args.dir):
    for file in files:
      # Detect the reference fiel
      images = np.genfromtxt(files_dir+file, dtype='str', delimiter=',', usecols=(0,1))
      itm_idx = np.where(images[:,1] != images[0,0])

      data = pylab.loadtxt(files_dir+file, delimiter=',', usecols=(2,3,4,5,6,7,8,9))
      matches = np.concatenate([matches, data[itm_idx[0], 0]])
      h1 = np.concatenate([h1, data[itm_idx[0], 1]])
      h2 = np.concatenate([h2, data[itm_idx[0], 2]])
      h3 = np.concatenate([h3, data[itm_idx[0], 3]])

  # Compute the matches percentage
  mp_1 = calc_match_mean(h1, matches, args.thresh, args.size)
  mp_2 = calc_match_mean(h2, matches, args.thresh, args.size)
  mp_3 = calc_match_mean(h3, matches, args.thresh, args.size)

  # Generate the histogram bins
  bins = np.arange(0, len(matches), args.size);
  center = (bins[:-1] + bins[1:]) / 2
  x1, y1 = get_axes(center, mp_1)
  x2, y2 = get_axes(center, mp_2)
  x3, y3 = get_axes(center, mp_3)

  # Get the width of every figure
  w1 = 0.7 * (x1[1] - x1[0])
  w2 = 0.7 * (x2[1] - x2[0])
  w3 = 0.7 * (x3[1] - x3[0])

  # Figure
  f1, (ax11, ax12, ax13) = plt.subplots(1, 3, sharey=True)

  # Hash 1
  ax11.plot(np.log(1 + h1), matches, marker='o', ls='')
  ax11.set_title(str(len(matches)) + " Samples (Hash Hyperplanes)")
  ax11.set_xlabel("Hash Matching")
  ax11.set_ylabel("Descriptor Matches")
  ax11.grid(True)

  # Hash 2
  ax12.plot(np.log(1 + h2), matches, marker='o', ls='')
  ax12.set_title(str(len(matches)) + " Samples (Hash Hystogram)")
  ax12.set_xlabel("Hash Matching")
  ax12.grid(True)

  # Hash 3
  ax13.plot(np.log(1 + h3), matches, marker='o', ls='')
  ax13.set_title(str(len(matches)) + " Samples (Hash Projections)")
  ax13.set_xlabel("Hash Matching")
  ax13.grid(True)

  # Figure
  f2, (ax21, ax22, ax23) = plt.subplots(1, 3, sharey=True)

  # Setup figure
  xmin = 0
  if (len(mp_1) > 150):
    xmin = -4
  title = "Success percentage (%)"
  if (np.amax(mp_1)>=100):
    title = "Mean feat. matching"

  # Hash 1
  barlist1 = ax21.bar(x1, y1, align='center', width=w1)
  ax21.set_title("Hyperplanes")
  ax21.set_xlabel("Hash Matching")
  ax21.set_ylabel(title)
  ax21.set_xlim(xmin, len(mp_1))
  ax21.grid(True)
  for i in range(len(barlist1)):
    barlist1[i].set_color('b')

  # Hash 2
  barlist2 = ax22.bar(x2, y2, align='center', width=w2)
  ax22.set_title("Hystogram")
  ax22.set_xlabel("Hash Matching")
  ax22.set_xlim(xmin, len(mp_2))
  ax22.grid(True)
  for i in range(len(barlist2)):
    barlist2[i].set_color('b')

  # Hash 3
  barlist3 = ax23.bar(x3, y3, align='center', width=w3)
  ax23.set_title("Projections")
  ax23.set_xlabel("Hash Matching")
  ax23.set_xlim(xmin, len(mp_3))
  ax23.grid(True)
  for i in range(len(barlist3)):
    barlist3[i].set_color('b')

  # Gloabl title
  files_dir_parts = files_dir.split('/');
  fig = plt.gcf()
  fig.suptitle("Dataset: " + files_dir_parts[-2] + " (" + str(len(matches)) + " Samples). Bucket Size: " + str(args.size), fontsize=14)

  plt.show()