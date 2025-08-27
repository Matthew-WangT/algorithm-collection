import casadi as ca

def euler_to_rotmat(roll, pitch, yaw):
    """
    欧拉角 (ZYX, yaw-pitch-roll) 转旋转矩阵
    输入: roll, pitch, yaw (casadi.SX 或 casadi.MX 或 float)
    输出: 3x3 旋转矩阵 (casadi.SX)
    """
    cr = ca.cos(roll)
    sr = ca.sin(roll)
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cy = ca.cos(yaw)
    sy = ca.sin(yaw)

    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp,   cp*sr,             cp*cr)
    )
    return R


class AnkleInfo:
    D = 0.12
    d = D / 2.0
    h1 = 0.49
    h2 = 0.37
    r = 0.12

    delta_z = 0.00  # TODO: 跟十字轴的z距离

    p_lu_3 = ca.vertcat(-r, +d, 0.0)
    p_ru_3 = ca.vertcat(-r, -d, 0.0)

    p_la_1 = ca.vertcat(0, +d, h1)
    p_ra_1 = ca.vertcat(0, -d, h2)
    p_lb_1 = ca.vertcat(-r, +d, h1)
    p_rb_1 = ca.vertcat(-r, -d, h2)


class Ankle:
    def __init__(self, info: AnkleInfo):
        self.info = info

    def get_p_lu_1(self, pitch, roll):
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ self.info.p_lu_3

    def get_p_ru_1(self, pitch, roll):
        R = euler_to_rotmat(roll, pitch, 0)
        return R @ self.info.p_ru_3

    def inv(self, pitch, roll):
        p_lu_1 = self.get_p_lu_1(pitch, roll)
        p_ru_1 = self.get_p_ru_1(pitch, roll)
        phi_l = self._get_phi(p_lu_1, self.info.p_la_1, self.info.h1, self.info.r)
        phi_r = self._get_phi(p_ru_1, self.info.p_ra_1, self.info.h2, self.info.r)
        return phi_l, phi_r

    @staticmethod
    def _get_phi(p_u, p_a, h, r):
        d_ly = p_a[1] - p_u[1]
        l_xz = ca.sqrt(h**2 - d_ly**2)

        delta_x = p_a[0] - p_u[0]
        delta_z = p_a[2] - p_u[2]
        delta_l = ca.sqrt(delta_x**2 + delta_z**2)

        alpha = ca.arctan2(delta_x, delta_z)
        beta = ca.acos((delta_l**2 + r**2 - l_xz**2) / (2 * r * delta_l))

        phi = alpha + beta - ca.pi / 2
        return phi

    def _jacobian_func(self):
        roll = ca.SX.sym('roll')
        pitch = ca.SX.sym('pitch')
        phi_l, phi_r = self.inv(pitch, roll)
        jac = ca.jacobian(ca.vertcat(phi_l, phi_r), ca.vertcat(pitch, roll))
        func = ca.Function('jacobian', [pitch, roll], [jac])
        return func

    def jacobian(self, roll, pitch):
        func = self._jacobian_func()
        return func(roll, pitch)


def __main__():
    info = AnkleInfo()
    ankle = Ankle(info)

    print("p_lu_1 @ (0,0):", ankle.get_p_lu_1(0, 0))
    print("p_ru_1 @ (0,0):", ankle.get_p_ru_1(0, 0))
    print('='*20)

    phi_l0 = ankle.inv(0, 0)
    print(f'phi_l0: {phi_l0}')

    print("phi_l(-0.2, 0):", ankle.inv(-0.2, 0))
    print("phi_l(0, -0.2):", ankle.inv(0.0, -0.2))
    
    print("jacobian:", ankle.jacobian(0.1, 0.2))


if __name__ == "__main__":
    __main__()
