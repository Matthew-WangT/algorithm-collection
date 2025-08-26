# 踝关节运动学解算程序
参照论文[^1]，编写踝关节的运动学逆解，并基于casadi计算雅可比，用于速度和扭矩映射。目前分别用numpy和casadi实现了运动学部分。

<p align="center">
    <img src="img/ankle_ik.png" width="300">
</p>

---
[1] W.-S. Jang, D.-Y. Kim, Y.-S. Choi, and Y.-J. Kim, “Self-Contained 2-DOF Ankle-Foot Prosthesis With Low-Inertia Extremity for Agile Walking on Uneven Terrain,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 8134–8141, Oct. 2021, doi: 10.1109/LRA.2021.3098931.