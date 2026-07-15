# 双臂 Rizon4 Base 布局优化（Capability Map + IRM 查表式）

离线求解两臂安装布局 `(d, theta)`：`d` = 两臂 base 间距，`theta` = 向内 roll
（保持现有水平安装 `rpy="±theta 1.57 0"`，与 `pedestal_to_*_base` 一致）。

方法：单臂能力图（FK 1e7 采样，体素 5cm × 32 方向 × 8 roll，姿态感知）
预计算一次 → 每个候选布局把任务点变换到 base 系查表评分（秒级全网格扫描）
→ top-k 布局用直接 IK（DLS 多 seed + 查表 q warm-start）精评定最终排序。

## 运行

```bash
~/miniconda3/envs/retarget_nn/bin/python -m base_placement.run_pipeline all
# 或分步: build-map / scan / refine，参数见 base_placement/config.yaml
```

输出在 `out_base_placement/`：`report.md`（最优布局+指标）、
`figs/scan_heatmaps.png`（主交付物）、`figs/capmap_*_D.png`、
`figs/best_layout_3d.png`、`scan_results.json`、`report.json`。

### 操作区中心 p 搜索模式（独立工作流，不进 `all`）

布局 `(d,θ)` 固定为 `region_search.d_fixed/theta_fixed`，反过来搜索
操作区 AABB 最佳中心 `p=(px,pz)`（py 恒为 0：双臂对称布置 + 对称任务集
使 Score 关于 py=0 镜像对称；pz 为相对桌面基准的偏移，`pz_range` 单值时
退化为纯 x 的 1D 搜索）。任务点在局部系只生成一次，候选间仅刚性平移，
能力图与评分公式复用，无需重建。

```bash
python -m base_placement.run_pipeline scan-region    # p 网格粗扫
python -m base_placement.run_pipeline refine-region  # top-k 候选 p 直接 IK 精评
```

输出：`scan_region_results.json`、`figs/scan_region_heatmaps.png`
（pz 单值时为"指标 vs px"曲线图）、`report_region.md/json`、
`figs/best_region_3d.png`。

## 模块

| 文件 | 职责 |
|---|---|
| `robot_model.py` | URDF 单臂子链、base 系 FK/雅可比、`make_base_pose(d,θ)` |
| `capability_map.py` | 能力图构建（并行 FK 采样）/ 查询 / 存取 / 自检 |
| `capsules.py` | 连杆胶囊近似，胶囊-平面（桌面）与胶囊-胶囊（臂间）闭式距离 |
| `task_points.py` | 合成任务点生成；UMI 轨迹 loader 接口（TODO） |
| `layout_eval.py` | 查表式布局评分 `Score(d,θ)` + 分项指标 + 网格扫描 |
| `refine_ik.py` | top-k 直接 IK 全评估（可达/可操作度/条件数/桌面/自碰撞） |
| `visualize.py` | 热力图与 3D 诊断图（dataviz 规范配色） |
| `run_pipeline.py` | CLI 入口 |

## 关键约定与已知取舍

- **EEF 姿态约定：x 轴 = 抓取接近方向**（URDF `end_effector_joint`
  的 rpy 使然），z 轴 = 手指闭合方向。姿态 bin 与任务点生成都按此。
- FK 采样能力图对边缘 bin 保守漏报（查表可达率 ≤ 真实 IK 可达率，
  交叉验证见 report）；bin 代表 q 是"最佳可操作度"解，碰撞初筛偏悲观，
  两者都由 top-k 直接 IK 精评兜底。
- 桌面裕度不含 gripper 段（抓取本来就要贴近桌面），臂间碰撞含全部胶囊。
- 臂间碰撞用 left[i]-right[i] 任务点配对近似"双臂同时作业"。
- `lambda_overlap=0`：交叠区质量只输出不进总分；双手协同任务多时在
  config 里调大并重跑 scan（秒级）。
- 实机部署前把 `robot.urdf_path` 换成 flexiv_rdk `Model::SyncURDF()`
  导出的单机标定 URDF。
- 设计决策全记录见 `../design_note.md` 与拷问结论（memory）。
