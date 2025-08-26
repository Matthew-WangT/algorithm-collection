import numpy as np

def euler2Matrix(roll, pitch, yaw):
  R_x = np.array([[1, 0, 0],
                  [0, np.cos(roll), -np.sin(roll)],
                  [0, np.sin(roll), np.cos(roll)]])
  R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                  [0, 1, 0],
                  [-np.sin(pitch), 0, np.cos(pitch)]])
  R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]])
  return R_z @ R_y @ R_x

class AnkleInfo:
  D = 0.15
  d = D/2.0
  h1 = 0.2
  h2 = 0.2
  r = 0.08
  
  delta_z = 0.00 # TODO: 跟十字轴的z距离
  
  p_lu_3 = np.array([-r, +d, 0.0])
  p_ru_3 = np.array([-r, -d, 0.0])
  
  p_la_1 = np.array([0, +d, h1])
  p_ra_1 = np.array([0, -d, h2])  
  p_lb_1 = np.array([-r, +d, h1])
  p_rb_1 = np.array([-r, -d, h2])


class Ankle:
  def __init__(self, info: AnkleInfo):
    self.info = info

  def get_p_lu_1(self, pitch, roll):
    R = euler2Matrix(roll, pitch, 0)
    return R @ self.info.p_lu_3

  def get_p_ru_1(self, pitch, roll):
    R = euler2Matrix(roll, pitch, 0)
    return R @ self.info.p_ru_3
  
  def inv(self, pitch, roll):
    p_lu_1 = self.get_p_lu_1(pitch, roll)
    # p_ru_1 = self.get_p_ru_1(pitch, roll)
    
    d_ly = self.info.p_la_1[1] - p_lu_1[1]
    l_xz = np.sqrt(self.info.h1**2 - d_ly**2)
    
    delta_x = self.info.p_la_1[0] - p_lu_1[0]
    delta_z = self.info.p_la_1[2] - p_lu_1[2]
    delta_l = np.sqrt(delta_x**2 + delta_z**2)
    
    alpha = np.arctan2(delta_x, delta_z)
    beta = np.arccos((delta_l**2 + self.info.r**2 - l_xz**2) / (2 * self.info.r * delta_l))
    # print(f"alpha: {alpha}")
    # print(f"beta: {beta}")
    phi_l = alpha + beta - np.pi/2
    
    return phi_l

def __main__():
  info = AnkleInfo()
  ankle = Ankle(info)
  print(ankle.get_p_lu_1(0, 0))
  print(ankle.get_p_ru_1(0, 0))
  print('='*10)
  phi_l0 = ankle.inv(0, 0)
  print(f'phi_l0: {phi_l0}')
  print(ankle.inv(-0.2, 0))
  print(ankle.inv(0.0, -0.2))

if __name__ == "__main__":
  __main__()
