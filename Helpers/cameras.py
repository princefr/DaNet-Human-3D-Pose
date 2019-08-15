
"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from . import  data_utils
from . import viz
import tensorflow as tf
import hashlib
import operator
import torch




def deterministic_random(min_value, max_value, data):
  digest = hashlib.sha256(data.encode()).digest()
  raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
  return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def normalize_screen_coordinates(X, w, h):
  assert X.shape[-1] == 2

  # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
  return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
  assert X.shape[-1] == 2
  # Reverse camera frame normalization
  return (X + [1, h / w]) * w / 2

def project_to_2d(X, camera_params):
  """

  :param X: 3D points in *camera space* to transform (N, *, 3)
  :param camera_params:
  :return:
  """


  assert X.shape[-1] == 3
  assert len(camera_params.shape) == 2
  assert camera_params.shape[-1] == 9
  assert X.shape[0] == camera_params.shape[0]

  while len(camera_params.shape) < len(X.shape):
    camera_params = np.expand_dims(camera_params, axis=1)

  f = camera_params[..., :2]
  c = camera_params[..., 2:4]
  k = camera_params[..., 4:7]
  p = camera_params[..., 7:]



  XX = X[..., :2] / X[..., 2:]
  r2 = np.sum(X[..., :2] * 2, axis=len(XX.shape)-1, keepdims=True)



  r_final = np.array([r2, r2**2, r2**3])
  r_final = r_final.reshape(r_final.shape[1], r_final.shape[2], r_final.shape[3], r_final.shape[0])

  radial = 1 + np.sum(k * r_final, axis=len(r2.shape)-1, keepdims=True)
  tan = np.sum(p*XX, axis=len(XX.shape)-1, keepdims=True)



  XXX = XX * (radial + tan) + p * r2
  value_to_return = f * XXX + c
  return np.squeeze(value_to_return, axis=0)



def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3



  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2



  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
  tan = p[0]*XX[1,:] + p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2


def qrot(q, v):
  """
  :param q:
  :param v:
  :return:
  """
  assert q.shape[-1] == 4
  assert v.shape[-1] == 3
  assert q.shape[:-1] == v.shape[:-1]

  qvec = q[..., 1:]

  uv = np.cross(qvec, v, axis=len(q.shape) -1)
  uuv = np.cross(qvec, uv, axis=len(q.shape) -1)
  data = (v + 2 * (q[..., :1] * uv + uuv))


  return data

def q_inverse(q, inplace=False):
  if inplace:
    q[..., 1:] *= -1
    return q
  else:
    w = q[..., :1]
    xyz = q[..., 1:]
    return np.concatenate((w, -xyz), axis=len(q.shape)-1)

def world_to_camera(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam:
  """
  R = q_inverse(R)
  R = np.tile(R, (*P.shape[:-1], 1))

  q = qrot(R, P - T)  # rotate and translate
  return  q

def camera_to_world(P, R, T):
  # need to invert this one.
  brah = np.tile(R, (*P.shape[:-1], 1))

  x = qrot(brah, P) + T
  return  x

def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  print(P.shape, "this is the p sh")

  assert len(P.shape) == 3
  assert P.shape[2] == 3


  X_cam = R.dot(P.T - T ) # rotate and translate

  return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T

def load_camera_params( hf, path ):
  """Load h36m camera parameters

  Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  R = hf[ path.format('R') ][:]
  R = R.T

  T = hf[ path.format('T') ][:]
  f = hf[ path.format('f') ][:]
  c = hf[ path.format('c') ][:]
  k = hf[ path.format('k') ][:]
  p = hf[ path.format('p') ][:]

  name = hf[ path.format('Name') ][:]
  name = "".join( [chr(item) for item in name] )

  return R, T, f, c, k, p, name

def load_cameras( bpath='cameras.h5', subjects=[1,5,6,7,8,9,11] ):
  """Loads the cameras of h36m

  Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  with h5py.File(bpath,'r') as hf:
    for s in subjects:
      for c in range(4): # There are 4 cameras in human3.6m
        rcams[(s, c+1)] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

  return rcams
