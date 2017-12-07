import numpy as np
# import rospy
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

class SLAM(object):
  """Simultaneous Localization And Mapping."""

  def show(self):
    """Returns an image visualizing the map and localization."""
    raise(NotImplementedError())


  def predict(self, w, v, dt):
    """Predicts state given angular velocity, velocity, and time.
    Args
      w  :: Float, angular velocity
      v  :: Float, velocity
      dt :: Float, time since last update
    """
    raise(NotImplementedError())


  def update(self, features):
    """Updates state given features.
    Arg
      features :: [(id, r, phi)], april tag ids and poses in bot frame
    """
    raise(NotImplementedError())


class EKF_SLAM(SLAM):
  """Extended Kalman Filter SLAM.

  `self.state` is an (1+N, 3) numpy array where N is number of features
  """
  def __init__(self, x_stdev=0.1, y_stdev=0.1, th_stdev=0.0314):

    # TODO make this rosparams in the node
    self.x_stdev =  x_stdev
    self.y_stdev = y_stdev
    self.th_stdev = th_stdev

    # Process noise
    self.R = np.diag([x_stdev, y_stdev, th_stdev])

    # Measurement noise TODO: parameters for this
    self.Q = np.diag([x_stdev, y_stdev])
    self.S = np.diag([x_stdev, y_stdev]) / 100

    self.mean = np.zeros(3, dtype='float64')
    self.cov = np.zeros((3,3), dtype='float64')
    self.N = 0
    self.features = {}

  def featureMean(self,j):
    return self.mean[3 + 2 * j : 3 + 2 * j + 2]

  def featureCov(self,j):
    return self.cov[3 + 2*j: 5 + 2*j, 3 + 2*j: 5 + 2*j]

  def show(self, sigma=3):
    """Returns a 3 sigma visualization about every point."""

    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot robot position
    plotCovariance(ax,
      self.mean[0:2],
      self.cov[0:2, 0:2],
      sigma, 'b','bot', self.mean[2])

    # Plot map features
    for tag in self.features:
      i = self.features[tag]
      plotCovariance(ax,
        self.featureMean(i),
        self.featureCov(i),
        sigma, 'r', str(tag))

    plt.xlim((-5,5))
    plt.ylim((-5,5))
    plt.show()


  def predict(self, w , v, dt):
    """Predicts robot pose given angular velocity, velocity, and time.
    See Probablistic Robotics Chapter 10 for maths.
    """
    # See Probablistic robots table 10.1 line 3
    th = self.mean[2]
    bot_pose_change = np.array([v * np.sin(th), v * np.cos(th), w]) * dt
    self.mean[0:3] += bot_pose_change

    # See Probablistic robots table 10.1 lines 2
    Fx = np.zeros((3, 2 * self.N + 3))
    Fx[0:3,0:3] = np.identity(3)
    FxT = Fx.transpose()

    # See Probablistic robots table 10.1 lines 4
    G = np.zeros((3,3))
    G[0,2] = v * (np.cos(th) - np.cos(th + w * dt))
    G[1,2] = v * (np.sin(th) - np.sin(th + w * dt))
    G = np.identity(2 * self.N + 3) + FxT.dot(G).dot(Fx)

    # See Probablistic robots table 10.1 line 5
    self.cov = G.dot(self.cov).dot(G.transpose())
    self.cov += FxT.dot(self.R).dot(Fx)

  def augment(self, r, phi, featureID):
    """Augment mean and covariance matrix with the new landmark."""
    ux, uy, th = self.mean[0:3]
    lx = ux + r * np.sin(th + phi)
    ly = uy + r * np.cos(th + phi)
    self.mean = np.append(self.mean,[lx, ly])

    U = np.zeros((2, 3 + self.N * 2))
    U[0,0] = U[1,1] = 1
    U[0,2] = r *  np.cos(th + phi)
    U[1,2] = r * -np.sin(th + phi)
    UT = U.transpose()
    self.cov = np.vstack([
      np.hstack([self.cov       , self.cov.dot(UT)]),
      np.hstack([U.dot(self.cov), U.dot(self.cov).dot(UT) + self.S])
    ]) # TODO: don't use self.R there should be another set of parameters

    self.features[featureID] = self.N
    self.N += 1

  def single_update(self, r, phi, featureID):
    """Updates mean and covariance given this observation."""
    j = self.features[featureID]

    delta = self.featureMean(j) - self.mean[0:2]
    q = delta.dot(delta)
    rq = np.sqrt(q)
    dx, dy = delta

    # Innovation
    z = np.array([r, phi])
    zhat = np.array([rq, np.arctan2(dy, dx) - self.mean[2]])

    # Table 10.1 line 15
    Fxj = np.zeros((5, 3 + 2 * self.N))
    Fxj[0:3,0:3] = np.identity(3)
    Fxj[3:5, 2*j+3: 2*j+5] = np.identity(2)

    # Table 10.1 line 16
    H = (np.array(
          [[-rq * dx, -rq * dy, 0 , rq * dx, rq * dy],
           [dy      , dx      , -q, -dy    , -dx    ]]) / q
        ).dot(Fxj)
    HT = H.transpose()

    # Kalman gain
    K = self.cov.dot(HT).dot(np.linalg.inv(H.dot(self.cov).dot(HT) + self.Q))

    # Update
    self.mean += K.dot(z - zhat)
    self.cov = (np.identity(self.mean.size) + K.dot(H)).dot(self.cov)


  def update(self, features):
    """Updates state given features.
    See Probablistic Robotics Chapter 10 for maths.
    """
    for featureID, r, phi in features:
      if featureID in self.features:
        self.single_update(r,phi, featureID)
      else:
        self.augment(r, phi, featureID)




def plotCovariance(ax, xy, cov, sigma, color, name, angle=None):
  """Plots one point and its 3 sigma covariance matrix."""

  x, y = xy
  evals, evecs = np.linalg.eigh(cov)

  ell = Ellipse(xy = xy,
                width = evals[0] * sigma * 2,
                height= evals[1] * sigma * 2,
                angle = np.rad2deg(np.arccos(evecs[0, 0])),
                alpha = 0.1,
                color = color)
  ax.add_artist(ell)

  plt.annotate(name,xy)

  if angle: plt.arrow(x, y, np.sin(angle), np.cos(angle) )

  plt.scatter(x, y, color=color)



# TODO
# * Node
#   - april tag to x, y, relative pose
#   - visual odometry
#   - publish image
# * EKF-SLAM class
#   - Update
#   - Predict
#   - Show
#
# message types for the april tag detector
#
# AprilTagDetectionArray :: [Detections]
# Detection :: (Int ID, Float size, PoseStamped pose)
# PoseStamped :: (Pose pose, Header header)
# Header :: (UInt32 timestamp, String referenceFrame)
# Pose :: (Point position, Quaternion orientation)
# Point :: (Float64 x, Float64 y, Float64 z)
# Quaternion :: (Float64 x, Float64 y, Float64 z, Float64 w)
#
# a Quaternion is a way of expressing rotation using an extended form of the
# complex numbers.
