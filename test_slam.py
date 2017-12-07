import numpy as np
from slam import *

def test_predict():
  x = EKF_SLAM()
  np.testing.assert_allclose(x.mean, np.array([0,0,0]))

  x.predict(0, 1, 1)
  np.testing.assert_allclose(x.mean, np.array([0,1,0]))

  x.predict(np.pi, 0, 0.5)
  np.testing.assert_allclose(x.mean, np.array([0,1,np.pi/2]))

  x.predict(0,4, 0.25)
  np.testing.assert_allclose(x.mean, np.array([1,1,np.pi/2]))


def test_update():
  x = EKF_SLAM()
  x.predict(0, 1, 1)
  x.show()
  x.predict(np.pi, 0, 0.5)
  x.show()
  x.predict(0,4, 0.25)
  x.show()
  x.update([(0, 1, np.pi/4)])
  x.show()
  x.predict(np.pi, 0, 0.5)
  x.show()
  x.predict(0, 1, 1)
  x.show()
  x.predict(0, 1, 1)
  x.show()



if __name__ == '__main__':
  print("Testing `predict`...")
  test_predict()

  test_update()

  print("All tests passed.")
