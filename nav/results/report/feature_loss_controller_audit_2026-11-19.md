Feature loss、controller 与 policy 核对（2026-11-19）

核对范围：2026-11-13 报告中的 yf 实验。本说明基于本地报告、导出的结果表和评估代码；截至本次整理，尚未完成 Torch 原始日志复核，因此没有把本地配置默认值当成服务器实际运行参数的独立证明，也没有重新计算每个 seed 的原始日志。队友当前提出“未收敛”的具体 run 尚未确认。

**可以给出的结论：验证集 feature MSE 已接近平台，但按现有汇总脚本的 2% 判据，尚不能称为已经收敛。**

这里的 feature MSE 是 `pred_loss` 在 `loss.pred=mse` 下的值：预测未来的 projected embedding 与真实未来图像的 projected embedding 之间的均方误差。它不包括 SIGReg 和 inverse-dynamics（B1）损失。

| 比较组 | epoch 7 的 val pred_loss | epoch 10 | 原汇总中的变化 | 原汇总的判定 |
|---|---:|---:|---:|---|
| 不加 B1：s4_xysc_fs8_h3 | 0.0599 | 0.0581 | −3.0% | still decreasing |
| B1 λ=1：s5inv_xysc_fs8_h3_w1 | 0.0833 | 0.0812 | −2.5% | still decreasing |

数值是 seed 3072、1、2 的均值。`summarize_epochs.py` 使用 `abs(relative_change) < 0.02` 才标记为 `flat`，默认比较 epoch 7 与 10。报告采用约 3% 的变化量描述“已收敛”，与脚本的定义不一致。这支持“接近平台、仍缓慢下降”，不支持无条件的“已完全收敛”。该 2% 规则是工程判据，不是数学收敛证明；未满足它也不等于训练发散。

仅凭三 seed 的均值，不能断言每个 seed 都达到平台。训练集 feature loss 与验证集 feature loss 也必须分别报告；原报告描述 train 还在下降，不能把 val 近似平台扩大为整个训练已经收敛。

来源：[含归一化指标的 Stage 6 汇总](../stage6/stage6_epochs.md)、[判定代码](https://github.com/fanxinyusarah/lewm-navigation-capstone/blob/080e49752c12f5784f67014c0105b7717269a831/yf/nav/summarize_epochs.py)、[11-13 报告源文件](source_2026-11-13/report.html)。

**三类评估使用的动作来源不同。**

| 评估 | controller / policy | 输入与评价对象 |
|---|---|---|
| Feature MSE / 多步 rollout 误差 | 不在线运行导航 controller；使用离线数据集记录的动作 | 历史真实特征和记录的动作 → 预测未来特征，与同一轨迹真实未来特征比较 |
| 模拟器分支排序 | 穷举候选动作序列，再由 world model 打分选取 | 主实验 5 种转角 × 3 步，共 125 条候选，从相同状态执行；按终点最短路径距离检验排序 |
| 闭环导航 | LeWM + CEM model-predictive control（MPC），代码类为 CEMPolicy | CEM 搜索未来动作序列，执行最优序列的第一步，再用新观测重新规划 |

因此，feature loss 曲线本身没有一个正在与模拟器交互的 evaluation policy。闭环 controller 的性能是另一个实验，不能用它来定义训练 loss 是否收敛。离线数据采集时具体使用了哪个版本的行为 policy，还需要数据生成记录；当前已读的评估代码不能证明这一点。

**11-13 报告与本地闭环代码一致的设置如下。服务器输出中的 `args` 仍需核对。**

| 项目 | 设置 |
|---|---|
| 模型 | xysc、fs8、history H=3；不加 B1 / B1 λ=1，各 3 个 seed |
| checkpoint | 报告注明 epoch 10；代码默认加载 run 目录中 epoch 编号最大的权重 |
| Policy | 冻结的 LeWM + 在线 CEM 搜索，没有另外训练 actor policy 网络 |
| MPC horizon | 3 个模型步；H=3 是历史长度，horizon=3 是规划长度，概念不同 |
| CEM | 每次迭代 300 个候选，30 个 elite，3 次迭代；上一次均值序列移位作为 warm start |
| 规划代价 | 第 3 步预测特征与目标图像特征的余弦距离 `1 − cosine_similarity` |
| 优化变量 | 每步位移参数与转角，再转为局部 `(dx, dy, dtheta)` waypoint；限制在训练动作范围 |
| 执行 | 只执行最优序列的第一步，然后重新观测、重新规划 |
| 底层执行器 | 自定义 Habitat waypoint executor：转向—直走—转向，直走由 Habitat `move_forward` 执行并处理碰撞 |
| 物理控制限制 | `enable_physics=False`；旋转直接设置 agent 朝向，没有实现真实机器人的速度/PID 跟踪控制 |
| 历史动作反馈 | 模拟器提供的实际实现位移 `realized`；不是纯图像估计的运动反馈 |
| 环境与样本 | 报告为 Habitat-sim 0.3.3、7 个 val 场景；每场景每距离档 20 回合，3 档，总计每个 policy/run 420 回合 |
| 对照 policy | random、straight、euclid_greedy（知道坐标）、geo_greedy（查询模拟器最短路径的 oracle） |

来源：[MPC 与 policy 代码](https://github.com/fanxinyusarah/lewm-navigation-capstone/blob/080e49752c12f5784f67014c0105b7717269a831/yf/nav/sim/mpc.py)、[模型加载与距离函数](https://github.com/fanxinyusarah/lewm-navigation-capstone/blob/080e49752c12f5784f67014c0105b7717269a831/yf/nav/sim/lewm_sim.py)、[底层 waypoint 执行](https://github.com/fanxinyusarah/lewm-navigation-capstone/blob/080e49752c12f5784f67014c0105b7717269a831/yf/nav/sim/habitat_bridge.py)、[提交脚本](https://github.com/fanxinyusarah/lewm-navigation-capstone/blob/080e49752c12f5784f67014c0105b7717269a831/yf/nav/slurm/sim.sbatch)。

执行器不是连续圆弧的物理跟踪器：代码用转向—直线—转向达到指定终点；其途中路径和真实圆弧不同，碰撞表现可能受影响。报告中“沿圆弧执行”的表述应按这一实现细化。

**成功率必须附带停止判据。**

B1 的 1–2 m 成功率 57.9% 指轨迹曾达到 geodesic distance < 0.5 m；首次自主停止成功率为 24.8%。它们不是同一个指标。停止阈值 tau 在训练场景标定。当前代码记录首次阈值触发时刻，但不会仅因为一次错误停止触发就立即终止轨迹，而是继续用于到达评估；因此 57.9% 不能解释为真实自主停止 policy 的任务成功率。

起点朝向和目标相机朝向的构造也使用最短路径信息，历史动作由模拟器的真实实现位移提供。教授面前应称为带这些辅助条件的模拟器评估，不应称为已经验证的纯视觉真实机器人导航。

来源：[闭环结果及指标定义](../sim/mpc_summary.md)、[运行与 metrics 实现](https://github.com/fanxinyusarah/lewm-navigation-capstone/blob/080e49752c12f5784f67014c0105b7717269a831/yf/nav/sim/mpc.py)。

**队友的结果尚不能与上述结果合并。**

仓库已有 [B1_ALIGNMENT.md](../protocol/B1_ALIGNMENT.md)，记录了当时两套 pipeline 的训练量、batch size、场景划分和验证采样不一致。主线历史脚本的 30 epoch 含 `limit_train_batches=100`，不能直接与 yf 的 10 个完整 epoch 比较。这是历史配置证据，不能据此认定队友现在使用的 run 仍然相同，也不能断言其未收敛就是由其中某一个差异造成。

待补证据：队友具体 run 和 loss 字段；双方逐 seed 的原始 `metrics.csv` 和 resolved `config.yaml`；Torch 闭环 JSON 保存的实际 `args`、policy 名称及 checkpoint 身份；必要时以相同总更新步数、样本量和固定验证集复核。

可向教授使用的当前表述：

> For the yf runs in the November 13 report, the validation feature-prediction MSE is near a plateau but does not satisfy the existing <2% tail-change criterion: it decreases by about 3.0% without B1 and 2.5% with B1 from epoch 7 to 10, averaged over three seeds. We therefore do not claim strict empirical convergence or convergence of every individual run. Feature loss is evaluated offline using recorded dataset actions. The separate closed-loop navigation evaluation uses a frozen LeWM with CEM-based MPC, a three-step planning horizon, and cosine distance to the goal embedding. A custom Habitat turn–move-forward–turn executor applies the selected waypoint. Reported reachability and autonomous-stop success are separate metrics. The teammate's run and the server-side run metadata still need to be checked before making a joint claim.
