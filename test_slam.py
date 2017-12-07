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


def execute_path(path, show = False):
  x = EKF_SLAM()
  for isPredict, args in path:
    if isPredict:
      w, v, dt = args
      x.predict(w, v, dt)
    else:
      x.update(args)
    if show:
      print(x.mean)
      x.show()

def test_update():
  path = [
    (True,  (0,1,1)),
    # (False, [('A',0.5,0)]),
    (True,  (np.pi/2, 0.5,1)),
    (False, [('B', 1, 0)]),
    (True,   (0, 0.5, 1)),
    # (False, [('B', 0.5, 0)]),
    (True,  (0 , 0.5,1)),
    (True,  (np.pi/2, 0 ,1)),
    (False, [('C', 1, 0)]),

    (True,   (0, 0.5, 1)),
    # (False, [('C',0.5,0)]),
    (True,  (np.pi/2, 0.5, 1)),
    (True,  (0, 0.5, 1)),
    (True,  (np.pi/2, 0.5, 1)),
  ]

  execute_path(path, True)




if __name__ == '__main__':
  print("Testing `predict`...")
  test_predict()

  test_update()

  print("All tests passed.")
