# Shouldersurfing Smart Glass

#### Background and motivation

- 现在有好几款智能眼镜，通过type\-C的线可以与手机/笔记本电脑相连，然后可以直接在眼镜里面看手机和笔记本的屏幕，比如阅读文档，这样做既便利又能在很多公众场合保护隐私和信息安全。

- 但我们发现，这种眼镜并不能全方位保护隐私和信息的安全，眼镜中的OLED屏幕内容依然能够从眼镜两侧的位置泄露出来，因为这些眼镜不是全包的，且光会经过反射从眼镜下方透出来，所以如果一个恶意主体坐在佩带眼镜者的旁边，将手机放在身侧，朝向眼镜的方向，可以录制到眼镜里面的内容，我们目前想做的就是通过这个现象来窃取眼镜里所显示的文字内容。

- 但这个任务并不简单，首先，因为眼镜中的OLED屏幕十分小，所以录制下来的内容是模糊的，其次，因光的衰减和反射损失，录制到的视频会发生颜色和形态的变化，也不利用直接看出内容。



#### Related/existing works?

Any existing studies related to the above task? Discuss and show their limitations



#### Our solution

为了解决以上问题：

- 先对窥视到的退化图像建立路径级有效前向模型：

    - **建模范围。** 当前观测是手机在超级微距模式下输出的 JPEG，而不是 RAW。OLED 发光、镜片反射、手机镜头、传感器、ISP 和 JPEG 中若干作用在现有数据下相互耦合，因此本文拟合的是“清晰显示内容到指定可读反射路径 JPEG”的有效模型，不把不可分离的量伪装成独立物理参数。47 mm 与 69--70 mm 是两个独立条件域；新增设备、角度或距离时复用同一方程，只新增域参数。

    - **完整版总模型。** 令 \(X_j\in[0,1]^{H\times W\times3}\) 为第 \(j\) 张清晰 sRGB 显示图，\(Y_{d,p,j,t}\) 为域 \(d\)、反射路径 \(p\)、burst 帧 \(t\) 的观测，则

    \[
    \widehat Y_{d,p,j,t}
    =\mathcal J_{Q_d}\!\left\{
    \operatorname{clip}_{[0,1]}\!\left[
    g_{d,p,j,t}\!\left(
    A_{d,p}\,\mathcal D_{s_{d,p}}\!\left[
    k_{d,p,\mathbf u}*\left(m_{d,p}(\mathbf u)\odot
    \mathcal W_{H_{d,p,j,t}}(X_j)\right)
    \right]+b_{d,p,j,t}(\mathbf u)
    \right)+\varepsilon_{d,p,j,t}
    \right]\right\}.
    \]

    当同一照片含多条可见反射时，进入相机响应前的光学项扩展为

    \[
    Z_{d,j,t}(\mathbf u)=\sum_{p=1}^{P_d}\alpha_{d,p}(\mathbf u)\,
    \mathcal D_{s_{d,p}}\!\left[k_{d,p,\mathbf u}*
    \left(m_{d,p}(\mathbf u)\odot\mathcal W_{H_{d,p,j,t}}(X_j)\right)\right].
    \]

    当前数据只可靠拟合主要可读路径，故本轮令 \(P_d=1\)；多路径权重 \(\alpha_p\) 留待下一轮单亮点扫描后估计。

    | 符号 | 含义 | 参数层级 |
    |---|---|---|
    | \(\mathcal W_H\) | 裁剪、镜像、旋转、非等比缩放和透视的局部投影 | 域/路径级 + 帧间小抖动 |
    | \(m(\mathbf u)\) | 镜片路径的空间透射/衰减场 | 域/路径级 |
    | \(k_{\mathbf u}\) | 合并光学、离焦、轻微运动和 ISP 锐化后的有效 PSF | 域/路径级，可空间变化 |
    | \(\mathcal D_s\) | 相机采样与重采样 | 域/路径级 |
    | \(A\) | 有效 RGB 通道混合 | 域/路径级 |
    | \(g\) | 自动曝光、白平衡、tone mapping 和 gamma 的逐通道单调响应 | 帧级干扰量 |
    | \(b(\mathbf u)\) | 环境光、镜框散射、flare 和 JPEG 黑电平 | 帧级/低频空间项 |
    | \(\varepsilon\) | 信号相关、通道相关的 JPEG 空间残差噪声 | 帧级随机量 |
    | \(\mathcal J_Q\) | JPEG 量化与色度子采样 | 编码级 |

    - **拟合目标。** 几何、PSF、光度和噪声先分模块估计，最后才做小范围联合细化，避免几何、模糊和 tone curve 相互补偿：

    \[
    \mathcal L(\theta)=
    \lambda_I\rho\!\left(M\odot(Y-\widehat Y)\right)
    +\lambda_{\nabla}\rho\!\left(M\odot(\nabla Y-\nabla\widehat Y)\right)
    +\lambda_E\left|w_E(Y)-w_E(\widehat Y)\right|+\mathcal R(\theta),
    \]

    其中 \(M\) 排除遮挡、饱和和其他反射路径，\(\rho\) 为稳健 Charbonnier/Huber 损失，\(w_E\) 为边缘宽度，\(\mathcal R\) 约束 PSF 非负、归一化及空间平滑。

    - **当前可识别参数。** 结果由 `analysis/run_fit_v1.py` 从 12 组、共 120 张 4096 x 3072 JPEG 自动生成，完整值和状态见 `analysis/results/v1_parameters.json`。

    | 条件域 | 局部尺度（相机 px / 源图 px） | 有效模糊 | 状态与解释 |
    |---|---:|---:|---|
    | 47 mm 主路径 | \(0.4898\times0.4192\) | 点阵 PSF FWHM \(7.82\times9.68\) 相机 px；约 \(15.97\times23.10\) 源图 px；主轴约 \(49.99^\circ\) | 尺度和 PSF 为 `estimated`；10 帧尺度 IQR 分别为 0.4736--0.4924、0.4114--0.4253，FWHM IQR 为 6.90--8.55、9.00--10.60 px |
    | 69 mm 主路径 | \(0.4738\times0.5046\) | 棋盘边缘梯度 FWHM 约 \(6.00\) 相机 px | 尺度为 `estimated`；模糊仅为 `provisional`，因为没有该焦段的孤立点 PSF；尺度 IQR 为 0.4696--0.4793、0.4989--0.5077 |

    两个域的 JPEG 量化表完全一致，等价于 `quality=96`、YCbCr 4:2:0。47 mm 颜色模型的留一色条误差为 0.127（归一化 JPEG 值），且出现非物理负增益，因此颜色、背景和噪声均保留为 `provisional`，默认合成时不自动应用。3 px 点阵在现有拍摄中不可可靠检测，只能作为分辨率下限证据。

    - **仍不可识别的参数。** 下列量不能由当前 JPEG 强行确定：

        1. 绝对源图裁剪、镜像状态、完整单应矩阵 \(H\)，以及各反射副本的路径数、\(H_p\) 和权重 \(\alpha_p\)：现有棋盘是周期图案，且缺少与目标紧邻拍摄的非对称定位点。
        2. 完整空间变化 PSF \(k_{\mathbf u}\) 与独立的光学/离焦/运动/ISP 锐化分量：只有有限视场位置，69 mm 没有孤立点阵。
        3. 唯一的 gamma/tone curve、3 x 3 颜色矩阵 \(A\)、空间衰减 \(m(\mathbf u)\) 与真实黑电平：自动曝光和自动白平衡开启，不同图案的曝光/ISO 不一致。
        4. shot noise、read noise、固定图案噪声与 JPEG/去噪残差的独立参数：只有处理后 JPEG，没有 RAW/DNG，灰阶数和同曝光帧数不足。
        5. 与快门时间和手机角速度条件化的运动模糊核：当前 burst 没有受控运动或同步 IMU。
        6. 跨设备、跨距离、跨姿态、跨环境光的参数分布，以及 OCR 内容泛化误差：当前仅一副眼镜、一个手机、两个邻近焦段和一张固定文本图。

    经典的模糊--采样--噪声--JPEG 链可参考 [Real-ESRGAN](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html) 与 [BSRGAN degradation model](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.html)；RAW 噪声和 ISP 可识别性参考 [Unprocessing Images](https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html)；多重反射的移位衰减模型参考 [Reflection Removal Using Ghosting Cues](https://openaccess.thecvf.com/content_cvpr_2015/html/Shih_Reflection_Removal_Using_2015_CVPR_paper.html)。

    - **下一轮必拍数据。** 每个“完整标定姿态”都必须锁定曝光、ISO、对焦、白平衡和焦段，关闭 HDR、美颜、夜景和场景优化；标定图与目标图之间不得移动相机或眼镜，并保存原始 EXIF、pose ID 和 path ID。

    | 优先级 | 必拍图像 | 每姿态帧数 | 解锁的方程项 |
    |---|---|---:|---|
    | P0 | 非对称定位点阵；在每个目标前后各拍一次 | 各 10 | 绝对裁剪、镜像、\(H_{p,t}\)、帧间抖动和路径身份 |
    | P0 | 单亮点扫描：白/R/G/B 亮点依次放在屏幕 5 x 5 位置 | 每位置每颜色 10 | 多路径数 \(P\)、每路 \(H_p\)、\(\alpha_p\)、空间变化 PSF 与色散 |
    | P0 | 同一帧内的黑块 + 16 级灰阶 + RGB/CMY/灰色色块，并带四角 fiducial | 10 | \(g\)、\(A\)、\(m(\mathbf u)\)、黑电平；消除跨图案自动曝光混淆 |
    | P0 | 3 x 3 视场位置的 3/7 px 稀疏点和水平/垂直/对角斜边，均带 fiducial | 各 10 | \(k_{\mathbf u}\)、各向异性 MTF、ISP overshoot/ringing |
    | P0 | 棋盘及其逐像素反相图、全黑图；同一曝光且不移动 | 各 10 | 分离加性 flare/背景 \(b\) 与乘性衰减 \(m\) |
    | P1 | 每个 ISO/曝光档的全黑 burst 与多个灰阶 burst | 各 30 | \(\operatorname{Var}(\varepsilon_c\mid\mu_c)=a_c\mu_c+b_c\)、固定图案噪声 |
    | P1 | 上述所有关键标定图的 RAW/DNG + JPEG 同步对 | 各 10 | 传感器噪声、ISP、tone curve 与 JPEG 的分层参数 |
    | P1 | 静止三脚架、受控平移/旋转和不同快门时间的点阵 | 各 10 | 条件运动核 \(k_{motion}(\tau,\omega)\) |
    | P1 | 至少 20 张独立随机文本图，字号/字体/颜色覆盖任务分布 | 每张 5--10 | 独立内容验证和 OCR 泛化，不参与 PSF 拟合 |

    泛用性采集建议分两层：每台设备至少选择 9 个锚点姿态做完整 P0/P1 标定，再随机选择至少 30 个覆盖姿态只拍“非对称点阵 + 随机文本”；覆盖近/中/远距离、左右 yaw、上下 pitch 和暗光/办公室/侧向强光三类环境。至少采集 3 台手机或 3 副眼镜、3 个独立 session，按设备和 session 划分训练/测试，不能把同一 burst 拆到两侧。

- 首先，进行多角度拍摄和图像预处理：

    - 第一步，拍摄多角度下眼镜内部的屏幕视频

    - 第二步，利用眼镜框与屏幕在亮度上的区别，进行屏幕检测，将屏幕切割出来；

    - 第三步，多帧筛选，筛选出较高质量和视角互补的N个帧

    - 第四步，对每一帧图像内的内容进行分词/分字，利用英文单词间的空格和中文字体之间的空隙进行分割。

- 然后，基于退化建模的理论，进行大量合成数据的构建，这一步是为了节省采集大量数据的人力时间和成本，合成数据20万\~50万样本主力训练集

    - 准备语料、字体、渲染参数

        - 语料准备，包括业务相关字符串（结合具体任务，比如面向金融/支付类、个人身份信息，基本都是数字\+字母的组合）和通用字符串辅助（维基百科、新闻语料切片）。

        - 字体：字体多样性是泛化的关键，面向具体任务，收集常见的字体

        - 渲染参数：字号16\~64 px，字间距\-2\~5 px，字符颜色全色域随机 \+ 黑白偏置，粗体/斜体30% 概率等

    - 退化模拟由 `analysis/degradation/model.py` 的前向算子执行，参数从 `v1_parameters.json` 按条件域采样，不再使用与设备无关的固定经验范围：

        1. 先采样 device/domain/path，再采样域级的尺度、PSF 和 JPEG 参数。
        2. 只有状态为 `estimated` 的参数默认启用；`provisional` 参数必须通过显式开关进入消融实验，`not_identifiable` 参数不能用单点伪装成拟合值。
        3. 当前 47 mm 域从 burst IQR 采样局部尺度和各向异性 PSF；69 mm 域只采样局部尺度，棋盘边缘宽度作为暂定模糊范围。
        4. 当前实拍 JPEG 固定使用等效 quality 96 和 4:2:0；未来收集其他手机后再按设备条件化 JPEG 表，而不是统一随机到低质量区间。
        5. 颜色、噪声、多路径、运动模糊和空间衰减在补拍完成前只作为独立鲁棒性增强，不计入“物理匹配合成集”。
        6. 合成/实拍比较同时报告 ROI 内边缘 FWHM、尺度、颜色误差、噪声方差和 OCR/CER；未配准图像不报告误导性的全图 PSNR/SSIM。

    - 多帧序列生成采用“共享结构参数 + 帧级干扰量”：同一序列共享清晰底图、域、路径、基础投影和 PSF；每帧只采样 burst 实测的亚像素抖动、曝光/白平衡漂移和随机残差。当前尚未可靠拟合的帧间运动范围不写成固定常数，待紧邻非对称点阵和受控运动数据采集后更新。

- 接下来，特征级多帧融合识别模型，也就是融合多帧的内容，进行目标信息的识别，整体模型流程如下：

    N 帧输入  \(N, 3, 32, W\)

    │

    ▼  共享 CNN Backbone \(各帧独立编码,权重共享\)

    │

    N 个特征序列  \(N, W', C\)

    │

    ▼  多帧特征融合模块  ★核心 例如用帧间 Cross\-Attention：每个时间步t，把N帧的同位置特征做自注意力

    │

    单个融合特征序列  \(W', C\)

    │

    ▼  BiLSTM / Transformer Encoder

    │

    上下文特征  \(W', C\)

    │

    ▼  CTC Head

    │

    字符串

    备注：

    真实数据利用方式：

    采集几千条真实数据

    不要从零训：先在合成数据上预训练 30 epoch

    微调阶段：合成 \+ 真实按 10:1 混采，学习率降到 1/10 （参数可能需要调整）

    测试集：真实 300 条，绝对不能进训练

    如果有真实无标注数据：用 Pseudo\-Labeling——模型先预测，置信度 \> 0\.9 的当伪标签加入训练

网络的参考架构：https://poe\.com/preview/HO98LHbObY765Cnf3FEK
