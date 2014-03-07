#!/usr/bin/env python
import sys
import pylab
import math
import numpy as np
import string
from matplotlib import pyplot


class Error(Exception):
  """ Base class for exceptions in this module. """
  pass


def normalize_vector(v):
  """ Normalizes the vector v """
  v_max = np.amax(v)
  if v_max == 0.0:
    return v

  v_n = v/np.amax(v);
  return v_n


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
          description='Show the correspondences between hash and descriptors.',
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('csv_file', 
          help='The output of hash_matching node: file with hash matching and descriptor matching.')
  
  args = parser.parse_args()

  # Save graph edges file into global
  csv_file = args.csv_file
  data = pylab.loadtxt(csv_file, delimiter=',', usecols=(2,3,4,5,6,7,8,9))

  # Normalize vectors
  matches = data[:,0]
  h1 = normalize_vector(data[:,1])
  h2 = normalize_vector(data[:,2])
  h3 = normalize_vector(data[:,3])

  # Init figure
  fig1 = pylab.figure()
  ax = fig1.gca()
  ax.grid(True)

  ax.set_title("Hash Matching vs Descriptor Matching")
  ax.set_xlabel("Hash Matching")
  ax.set_ylabel("Descriptor Matching")
  ax.plot(h1, matches, 'b', marker='o', ls='', label='Hyperplanes')
  ax.plot(h2, matches, 'r', marker='o', ls='', label='Histogram')
  ax.plot(h3, matches, 'g', marker='o', ls='', label='Projections')
  ax.legend()
  pylab.show()