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
  for args in path:
    if type(args) == list:
      x.update(args)

    else:
      w, v, dt = args
      x.predict(w, v, dt)

    if show:
      np.set_printoptions(precision=4,suppress=True)
      print("Mean:")
      print(x.mean)
      print("Cov:")
      print(x.cov)
      x.show()


def test_path2():
  path = [
    (0,0,1),
    [('A',2,0),('B',3,1)],
    (0,1,1),
    (np.pi/2, 1,1),
    (0, 1, 1),
    (0 , 1,1),
    (np.pi/2, 0 ,1),
    (0, 1, 1),
    (np.pi/2, 1, 1),
    (0, 1, 1),
    (np.pi/2, 1, 1),
    [('A',2,0),('B',3,1)]
  ]

  execute_path(path, True)

def test_path1():
  """Straight line with landmarks."""
  path = [(0,0,1)]
  path.append([('A',4,0)])
  path.append((0,1,1))
  path.append([('A',3,0)])
  path.append((0,1,1))
  path.append([('A',2,0)])
  path.append((0,1,1))
  path.append([('A',1,0)])

  execute_path(path, True)

def test_path2():
  """Go in a circle"""
  path = [ (np.pi/10, 0.3, 1)] * 20
  execute_path(path, True)

def test_path3():
  """Go in an arc, see a landmark, """
  path = [ (np.pi, 3, 0.1)] * 2
  path += [[('A',2,0)]]
  path.append((0,1,1))
  path += [[('A',1,0)]]
  execute_path(path, True)

def test_path4():
  """Square with 1 landmark"""
  path = [
    (0, 0, 1),
    [('A', 3, 0)],
    (0, 1, 1),
    [('A', 2, 0)],
    (np.pi/2, 1, 1),
    (0, 1, 1),
    (np.pi/2, 1, 1),
    (0, 1, 1),
    (np.pi/2, 1, 1),
    (0, 1, 1),
    (np.pi/2, 1, 1),
  ] * 4
  execute_path(path,True)





if __name__ == '__main__':
  print("Testing `predict`...")
  #test_predict()

  #test_path1()
  test_path4()

  print("All tests passed.")
