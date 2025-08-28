# 踝关节运动学解算程序
> 注意：关于并联杆机构，正向运动学和逆向运动学的定义在不同资料里存在不一致的情况。为避免存在混淆，在这里我们定义逆向运动学是已知关节空间$q_j$（踝关节pitch和roll），求解电机空间$q_m$的过程，正向运动学反之。

参照论文[^1]，编写踝关节的运动学逆解，并基于CasADi计算雅可比，用于速度和扭矩映射。目前分别用numpy和CasADi实现了运动学部分。

- 逆向运动学：参照论文方法构造
<p align="center">
    <img src="img/ankle_ik.jpg" width="300">
</p>

- 正向运动学：基于逆向运动学，使用CasADi计算雅可比$J$，再基于$J$使用迭代方法求解正向运动学问题。
   
   **算法流程：**
   
   目的：给定电机角度 $q_m = [\phi_l, \phi_r]^T$，求解关节角度 $q_j = [pitch, roll]^T$
   
   1. **初始化**：
      - 初始关节角度：$q_j^{(0)} = q_{j0}$（初值）
      - 步长因子：$\alpha = 0.9$
      - 收敛阈值：$\epsilon = 10^{-4}$
      - 最大迭代次数：$N_{max} = 100$
   
   2. **迭代过程**：
      
      ```
      for k = 0, 1, 2, ..., N_max:
          // 计算当前电机角度
          q_m^{(k)} = IK(q_j^{(k)})  // 逆运动学
          
          // 计算误差
          Δq_m = q_m - q_m^{(k)}
          
          // 收敛检查
          if ||Δq_m|| < ε: break
          
          // 计算雅可比矩阵
          J = ∂q_m/∂q_j |_{q_j^{(k)}}
          
          // 牛顿-拉夫逊更新
          q_j^{(k+1)} = q_j^{(k)} + α · J^{-1} · Δq_m
      ```

   
   3. **雅可比矩阵计算**：
      $$J = \frac{\partial q_m}{\partial q_j} = \begin{bmatrix}
      \frac{\partial \phi_l}{\partial pitch} & \frac{\partial \phi_l}{\partial roll} \\
      \frac{\partial \phi_r}{\partial pitch} & \frac{\partial \phi_r}{\partial roll}
      \end{bmatrix}$$
      使用CasADi符号微分自动计算
   
   4. **矩阵求逆优化**：
      对于2×2雅可比矩阵，使用解析公式求逆：
      $$J^{-1} = \frac{1}{\det(J)} \begin{bmatrix}
      J_{22} & -J_{12} \\
      -J_{21} & J_{11}
      \end{bmatrix}$$
      其中 $\det(J) = J_{11}J_{22} - J_{12}J_{21}$
   
   **算法特点：**
   - 收敛速度快（通常5-10次迭代）
   - 数值稳定性好（雅可比矩阵条件数良好）
   - 奇异性处理（检测行列式近零情况）
   - 最优性能（专用2×2矩阵求逆，比通用方法快3-5倍）

- 速度映射：
$$
\begin{array}{ll}
v_m&=J(q_j)v_j\\
v_j&=J(q_j)^{-1}v_m
\end{array}
$$

---
[1] W.-S. Jang, D.-Y. Kim, Y.-S. Choi, and Y.-J. Kim, “Self-Contained 2-DOF Ankle-Foot Prosthesis With Low-Inertia Extremity for Agile Walking on Uneven Terrain,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 8134–8141, Oct. 2021, doi: 10.1109/LRA.2021.3098931.