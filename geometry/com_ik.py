import numpy as np
from pydrake.all import (
    RigidTransform, 
    InverseKinematics, 
    RotationMatrix, 
    RollPitchYaw, 
    Solve, 
    MultibodyPlant,
    Context,
    ComPositionConstraint
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, StartMeshcat, Meshcat
from pydrake.multibody.tree import JointIndex

# 假设 TO_RADIAN 已经定义
TO_RADIAN = np.pi / 180.0 

class CoMIK:
    def __init__(self, plant, frames_name_dict, tol=1e-6, solver_tol=1e-6):
        self.plant_ = plant
        self.frames_name_dict_ = frames_name_dict
        self.tol_ = tol
        self.solver_tol_ = solver_tol
        self.plant_context_ = plant.CreateDefaultContext()
        self.prev_q_sol = np.zeros(plant.num_positions())
        self.locked_joint_names_ = []

    def add_locked_joints(self, joint_names):
        """添加需要锁定的关节名称"""
        self.locked_joint_names_.extend(joint_names)

    def add_com_position_constraint(self, ik, r_des):
        # In Python pydrake, InverseKinematics class doesn't have AddCenterOfMassPositionConstraint
        # We need to construct ComPositionConstraint manually and add it to the program.
        # ComPositionConstraint evaluates f(q, r) = CoM(q) - r = 0
        # So we introduce new variables r, and constrain r to be within r_des +/- tol.
        
        prog = ik.get_mutable_prog()
        
        # 1. Create new decision variables for CoM position 'r' (3 variables)
        r = prog.NewContinuousVariables(3, "com_r")
        
        # 2. Create the CoM constraint
        # model_instances=None means all bodies except world
        constraint = ComPositionConstraint(
            self.plant_,
            None, 
            self.plant_.world_frame(),
            self.plant_context_
        )
        
        # 3. Add constraint to program: CoM(q) == r
        # Constraint expects variables [q; r]
        # ik.q() returns q variables
        vars = np.concatenate([ik.q(), r])
        prog.AddConstraint(constraint, vars)
        
        # 4. Add bounding box constraint on r: r_des - tol <= r <= r_des + tol
        lb = r_des - np.full(3, self.tol_)
        ub = r_des + np.full(3, self.tol_)
        prog.AddBoundingBoxConstraint(lb, ub, r)

    def solve(self, end_pose_dict, q0):
        """
        Solves the Inverse Kinematics problem.
        
        This method combines the two C++ overload logics:
        1. Simple Mode: If arm arguments are None (matches C++ solve(pose, q0, q_sol))
        2. Complex/Arm Mode: If arm arguments are provided (matches C++ solve(pose, l_rot, r_rot..., q0, ...))
        """
        
        ik = InverseKinematics(self.plant_, self.plant_context_)
        prog = ik.get_mutable_prog()
        q_vars = ik.q()

        # 应用关节锁定约束
        for name in self.locked_joint_names_:
            if self.plant_.HasJointNamed(name):
                joint = self.plant_.GetJointByName(name)
                if joint.num_positions() == 1:
                    idx = joint.position_start()
                    val = q0[idx]
                    prog.AddBoundingBoxConstraint(val, val, q_vars[idx])
            else:
                print(f"Warning: Joint '{name}' not found, skipping lock.")
        
        for frame_name in end_pose_dict.keys():
            link_name = self.frames_name_dict_[frame_name]
            frame = self.plant_.GetFrameByName(link_name)
            
            # Orientation (magic number check from C++)
            ik.AddOrientationConstraint(
                self.plant_.world_frame(),
                RotationMatrix(RollPitchYaw(end_pose_dict[frame_name]['ori'])),
                frame,
                RotationMatrix(RollPitchYaw(0, 0, 0)),
                self.tol_
            )
            
            # Position
            if frame_name == "torso":
                self.add_com_position_constraint(ik, end_pose_dict[frame_name]['pos'])
            else:
                ik.AddPositionConstraint(
                    frame,
                    np.zeros(3),
                    self.plant_.world_frame(),
                    end_pose_dict[frame_name]['pos'] - self.tol_,
                    end_pose_dict[frame_name]['pos'] + self.tol_
                )

        # Common Solve Logic
        prog.SetInitialGuess(ik.q(), q0)
        
        # Solver Options (Commented out in C++, but here is how you would set them in Python)
        # prog.SetSolverOption(SnoptSolver().solver_id(), "Major feasibility tolerance", 1e-7)
        
        result = Solve(prog)
        
        if not result.is_success():
            print("Failed solution")
            # In C++ it returns q0 on failure in the complex method, or just false in simple.
            # We will follow the simple method behavior of returning success flag and result.
            return False, q0
        else:
            q_sol = result.GetSolution(ik.q())
            self.prev_q_sol = q_sol
            return True, q_sol

def initialize_robot(plant, plant_context):
    # 获取末端帧名称
    end_frames_name_dict = {
        "torso": "torso_yaw_link",
        "lfoot": "left_ankle_roll_link",
        "rfoot": "right_ankle_roll_link",
    }
    # 计算左脚相对于躯干的位姿
    frame_l_foot = plant.GetFrameByName(end_frames_name_dict["lfoot"])
    frame_torso = plant.GetFrameByName(end_frames_name_dict["torso"])
    foot_in_torso = frame_l_foot.CalcPose(plant_context, frame_torso)
    
    # 计算右脚相对于躯干的位姿
    frame_r_foot = plant.GetFrameByName(end_frames_name_dict["rfoot"])
    r_foot_in_torso = frame_r_foot.CalcPose(plant_context, frame_torso)
    
    p0_foot = np.array([0, foot_in_torso.translation()[1], 0])
    print(f"p0_foot: {p0_foot}")
    
    # 获取当前关节位置
    q0 = plant.GetPositions(plant_context)
    
    # 调整高度 (索引 6 通常是浮动基座的 Z 轴位置)
    q0[6] = -foot_in_torso.translation()[2] 
    
    # 设置新的位置到 context
    plant.SetPositions(plant_context, q0)
    
    # 计算质心
    r0 = plant.CalcCenterOfMassPositionInWorld(plant_context)
    
    # 打印脚宽
    print(f"foot width: {foot_in_torso.translation()[1] - r_foot_in_torso.translation()[1]}")
    
    # 设置特定关节的初始角度
    initial_joint_name = ["left_knee_pitch_joint", "right_knee_pitch_joint"]
    initial_joint_pos = [0.10, 0.10]
    
    for name, pos in zip(initial_joint_name, initial_joint_pos):
        # pydrake 中 GetJointByName 返回 Joint 对象
        joint = plant.GetJointByName(name)
        joint.set_angle(plant_context, pos)
        
    # 更新 q0
    q0 = plant.GetPositions(plant_context)
    print(f"initial torso pose: {q0[:7]}")
    
    torsoY = 0.0 * TO_RADIAN
    torsoP = 0.0 * TO_RADIAN
    com_z = 0.483
    step_width = 0.2
    # 获取末端帧名称
    end_pose_dict = {
        "torso": {
            "ori": np.array([0, torsoP, torsoY]),
            "pos": np.array([0.05, 0.0, com_z])
        },
        "lfoot": {
            "ori": np.array([0, 0, 0]),
            "pos": np.array([p0_foot[0], step_width / 2, p0_foot[2]])
        },
        "rfoot": {
            "ori": np.array([0, 0, 0]),
            "pos": np.array([p0_foot[0], -step_width / 2, p0_foot[2]])
        }
    }
    # --- IK 部分 ---
    ik = CoMIK(plant, end_frames_name_dict, tol=1e-3, solver_tol=1e-6)
    
    # 锁定手臂关节（名称包含 "arm" 的关节）
    left_arm_joints = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint","left_shoulder_yaw_joint", "left_elbow_pitch_joint"]
    right_arm_joints = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint","right_shoulder_yaw_joint", "right_elbow_pitch_joint"]
    locked_joints = ["torso_yaw_joint"] + left_arm_joints + right_arm_joints
            
    if locked_joints:
        print(f"Locking {len(locked_joints)} arm joints: {locked_joints}")
        ik.add_locked_joints(locked_joints)
    
    # 求解 IK
    result, q = ik.solve(end_pose_dict, q0)
    
    if not result:
        print(f"torso: {end_pose_dict['torso']['ori']}, {end_pose_dict['torso']['pos']}")
        print(f"lfoot: {end_pose_dict['lfoot']['ori']}, {end_pose_dict['lfoot']['pos']}")
        print(f"rfoot: {end_pose_dict['rfoot']['ori']}, {end_pose_dict['rfoot']['pos']}")
        raise RuntimeError("Failed to IK0!")

    print('='*20)
    plant.SetPositions(plant_context, q)
    r1 = plant.CalcCenterOfMassPositionInWorld(plant_context)
    print(f"final com: {r1}")
    # 计算躯干相对于右脚的高度
    torso_in_rfoot = frame_torso.CalcPose(plant_context, frame_r_foot)
    print(f"torso height: {torso_in_rfoot.translation()[2]}")

    return q

def load_robot_from_mjcf(mjcf_path, meshcat=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    
    # 创建解析器
    parser = Parser(plant)
    
    # 读取文件内容
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    # 使用 AddModelsFromString 加载修改后的内容
    # file_type 应为文件扩展名（不带点），如 'xml'
    file_type = mjcf_path.split('.')[-1]
    parser.AddModelsFromString(mjcf_content, file_type)
    
    plant.Finalize()
    
    # 添加可视化
    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat,
                                       MeshcatVisualizerParams())
                                       
    diagram = builder.Build()
    
    return plant, diagram

if __name__ == "__main__":
    # 启动 Meshcat
    meshcat = Meshcat()
    print(f"Open Meshcat at: {meshcat.web_url()}", flush=True)
    
    path = '/home/seyan/zyuon/zyuon-hrdf/robots/zhaplin-21dof/exported/robot.xml'
    plant, diagram = load_robot_from_mjcf(path, meshcat=meshcat)
    q = initialize_robot(plant, plant.CreateDefaultContext())
    print(f'l leg q: {q[7:13]}')
    print(f'r leg q: {q[13:19]}')
    
    # 可视化结果
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)
    
    # 保持运行以查看可视化
    import time
    print(f"Meshcat URL: {meshcat.web_url()}", flush=True)
    while True:
        time.sleep(1.0)
