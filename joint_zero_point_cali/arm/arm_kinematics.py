from pinocchio import casadi as cpin
import pinocchio as pin
import casadi as cs
from pinocchio.robot_wrapper import RobotWrapper

import os
import numpy as np

class ArmKinematics:
    def __init__(self, urdf_path):
        # 加载机器人模型
        origin_robot = RobotWrapper.BuildFromURDF(urdf_path)
        mixed_jointsToLockIDs = [
            "zarm_r1_joint" ,
            "zarm_r2_joint" ,
            "zarm_r3_joint" ,
            "zarm_r4_joint" ,
            "zarm_r5_joint" ,
            "zarm_r6_joint" ,
            "zarm_r7_joint" ,
        ]
        self.robot = origin_robot.buildReducedRobot(
            list_of_joints_to_lock=mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * origin_robot.model.nq),
        )

        # 创建 CasADi 符号变量
        # print("nq: ", origin_robot.model.nq)
        # print("nv: ", origin_robot.model.nv)
        print("nq_reduced: ", self.robot.model.nq)
        print("nv_reduced: ", self.robot.model.nv)

        self.cmodel = cpin.Model(self.robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = cs.SX.sym('q', self.robot.model.nq)
        self.cv = cs.SX.sym('v', self.robot.model.nv)
        cTf = cs.SX.sym("tf_l", 4, 4)
        
        L_hand_id = self.robot.model.getFrameId("zarm_l7_link")
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # cpin.forwardKinematics(self.cmodel, self.cdata, self.cq)
        # cpin.updateFramePlacement(self.cmodel, self.cdata, L_hand_id)  # 更新末端帧位姿
        # cpin.computeFrameJacobian(self.cmodel, self.cdata, self.cq, L_hand_id, cpin.ReferenceFrame.WORLD)
        # J = self.cdata.J

        L_hand_pose = self.cdata.oMf[L_hand_id]
        
        J_pos = cs.jacobian(L_hand_pose.translation, self.cq)
        J_rot = cs.jacobian(cpin.log3(L_hand_pose.rotation), self.cq)
        J = cs.vertcat(J_pos, J_rot)

        # print("L_hand_pos: ", L_hand_pose.translation)
        # print("J: ", J)
        self.kinematics = cs.Function('fk', 
                                [self.cq], 
                                [L_hand_pose.translation, L_hand_pose.rotation, J],
                                ['q'],
                                ['pos', 'rot', 'jacobian'])

    def FK(self, q):
        pos, rot, J = self.kinematics(q)
        # print("se3_pose: ", se3_pose)
        # pos, rot = se3_pose.translation, se3_pose.rotation
        return np.array(pos.full()).flatten(), np.array(rot.full()), np.array(J.full())
    
    def jacobian(self, q):
        J = cs.jacobian(self.kinematics(self.cq)[0], self.cq)
        return np.array(J.full())


if __name__ == "__main__":
    n_poses = 3
    q_list = [np.random.uniform(-np.pi, np.pi, 7) for _ in range(n_poses)]
    print("q_list: ", q_list)
    
    urdf_path = os.path.join(os.path.dirname(__file__), "robot.urdf")
    arm_kinematics = ArmKinematics(urdf_path)
    for q in q_list:
        pos, rot, J = arm_kinematics.FK(q)
        print("pos: ", pos)
        # print("rot: ", rot)
        # print("J: ", J)
    true_bias = np.array([0.02, 0.03, 0.05])