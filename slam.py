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
    self.Q = np.diag([x_stdev, y_stdev]) / 100
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
    # See Probablistic robots table 10.1 line 3 (modified for out model)
    th = self.mean[2]
    bot_pose_change = np.array([v * np.cos(th), v * np.sin(th), w]) * dt
    self.mean[0:3] += bot_pose_change
    self.mean[2] %= 2*np.pi

    # Linearized error of process model
    G = np.identity(2 * self.N + 3)
    G[0,2] = -v * np.sin(th) * dt
    G[1,2] =  v * np.cos(th) * dt
    G[2,2] = w * dt

    # Add Gaussian noise to Robot pose
    R = np.zeros((2 * self.N + 3, 2 * self.N + 3))
    R[0:3,0:3] = self.R

    # Update uncertainty
    self.cov = G.dot(self.cov).dot(G.transpose()) + R * dt

  def augment(self, r, phi, featureID):
    """Augment mean and covariance matrix with the new landmark."""
    ux, uy, th = self.mean[0:3]
    lx = ux + r * np.cos(th + phi)
    ly = uy + r * np.sin(th + phi)
    self.mean = np.append(self.mean,[lx, ly])
    # U is the linearized function that propagates state error to new landmark
    U = np.zeros((2, 3 + self.N * 2))
    U[0:2,0:2] = np.identity(2)
    U[0,2] = r * -np.sin(th + phi)
    U[1,2] = r *  np.cos(th + phi)
    UT = U.transpose()
    # Extend covariance matrix
    self.cov = np.vstack([
      np.hstack([ self.cov       , self.cov.dot(UT)                 ]),
      np.hstack([ U.dot(self.cov), U.dot(self.cov).dot(UT) + self.S ])])
    # Add to feature list
    self.features[featureID] = self.N
    self.N += 1

  def single_update(self, r, phi, featureID):
    """Updates mean and covariance given this observation."""
    j = self.features[featureID]

    delta = self.featureMean(j) - self.mean[0:2]
    dx, dy = delta
    q = delta.dot(delta)
    rq = np.sqrt(q)

    # Innovation
    z = np.array([float(r), phi])
    zhat = np.array([rq, np.arctan2(dy, dx) - self.mean[2]])

    # Table 10.1 line 15
    Fxj = np.zeros((5, 3 + 2 * self.N))
    Fxj[:3, :3] = np.identity(3)
    Fxj[3:5, 2*j+3: 2*j+5] = np.identity(2)

    # Table 10.1 line 16

    print("\nTHIS IS H\n")

    print("q %f, rq %f, dx %f, dy %f"%(q,rq,dx,dy))

    H = (np.array(
          [[-rq * dx, -rq * dy, 0 , rq * dx, rq * dy],
           [ dy     , -dx     , -q, -dy    , dx     ]]) / q
        ).dot(Fxj)
    HT = H.transpose()

    np.set_printoptions(precision=4,suppress=True)

    print(z, zhat)
    print(H)

    # Kalman gain
    K = self.cov.dot(HT).dot(np.linalg.inv(H.dot(self.cov).dot(HT) + self.Q))

    print("K")
    print(K)

    print("z-zhat")
    print(z-zhat)

    print("K(z-zhat)")
    print(K.dot(z-zhat))

    print("(I - KH)")
    print((np.identity(self.mean.size) - K.dot(H)))

    print("\nTHIS IS H\n")

    # Update
    self.mean += K.dot(z - zhat)
    self.mean[2] %= 2*np.pi
    self.cov = (np.identity(self.mean.size) - K.dot(H)).dot(self.cov)

  def update(self, features):
    """Updates state given features.
    """
    for featureID, r, phi in features:

      if featureID in self.features:
        print("Single update:(%f, %f, %s)" % (r,phi,featureID))
        self.single_update(r,phi, featureID)

      else:
        print("Augment:(%f, %f, %s)" % (r,phi,featureID))
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

  if angle != None:
    plt.arrow(x, y, np.cos(angle), np.sin(angle) )

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
