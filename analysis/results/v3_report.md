# c_core v3 有效退化模型拟合报告

## 结论

`c_core` 的 9 张 JPEG 已严格按文件时间顺序绑定到 9 张源标定图。三张独立置换响应图与红/灰阶图联合后，灰阶设计矩阵秩为 30/30，条件数 13.59；R/G/B 三通道无约束 tone 序列均单调，因此此前不可识别的 R tone 与 R 空间衰减在本标定序列内已经恢复。灰阶 MAE 为 0.031。

前向域行和约束颜色矩阵细化后的颜色块 MAE 为 0.115，状态为 `not_reliable_under_current_3x3_model`。这一定义是 JPEG 路径级有效矩阵，不解释为 OLED、镜片或相机 ISP 的独立物理矩阵。

首尾全黑图分别以相邻定位图配准，二次背景场拟合 MAE 为 0.0042。源图中心背景 RGB 从 [0.0745, 0.0743, 0.0667] 变为 [0.1556, 0.1596, 0.1421]，因此背景必须写成随时间变化的低频场，不能使用单个常数。

G/B 斜边共有 9/16 个 ESF 通过稳健拟合。窄边缘核心 FWHM 中位数为 G 4.37 px、B 5.76 px。该数值描述有效核的窄核心；点目标得到的宽翼和 ISP 振铃仍需保留在非高斯核模型中。

## 当前模型

\[
Z_t(\mathbf u)=\mathcal D_s\!\left[k_{\mathbf u,t}*\mathcal W_{H_t,\delta_2}(X)\right],
\qquad
\widehat Y_t=\mathcal J_{96,4:2:0}\!\left\{\operatorname{clip}_{[0,1]}\!\left[b_t(\mathbf u)+g_t\odot m_t(\mathbf u)\odot f\!\left(AZ_t(\mathbf u)\right)+\varepsilon_t(\mathbf u)\right]\right\}.
\]

其中 \(H_t\) 逐帧估计，\(\delta_2\) 沿用 v2 的二次几何残差；\(f\)、\(A\)、\(g_t\)、\(m_t\) 由四张联合光度图估计；\(b_t\) 为首尾二次场插值并允许帧内黑块校正；G/B 的斜边核心与 v2/v2.1 点目标宽翼共同约束 \(k_{\mathbf u,t}\)。

## 尚未分离

当前 47 mm 自动成像域的核心路径级方程已经闭合。仍未分离的是跨姿态/环境的参数分布、多反射路径、核的光学/离焦/运动/ISP 分解，以及 shot/read noise 与 JPEG/去噪残差；这些不能由本批 9 张 JPEG 强行物理解耦。
