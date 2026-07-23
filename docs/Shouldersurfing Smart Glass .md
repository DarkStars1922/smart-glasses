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

    - **建模范围。** 当前观测是手机在超级微距模式下输出的 JPEG，而不是 RAW。OLED 发光、镜片反射、手机镜头、传感器、ISP 和 JPEG 中若干作用在现有数据下相互耦合，因此本文拟合的是“清晰显示内容到指定可读反射路径 JPEG”的有效模型，不把不可分离的量伪装成独立物理参数。MIX Flip 的 47 mm 长焦微距只能通过自动成像链路进入，因此 47 mm 是包含自动曝光、自动白平衡和自动对焦的条件域；69--70 mm 数码裁切及 23 mm 手动主摄均作为其他独立域，参数不能混用。

    - **完整版总模型。** 令 \(X_j\in[0,1]^{H\times W\times3}\) 为第 \(j\) 张清晰 sRGB 显示图，\(Y_{d,p,j,t}\) 为域 \(d\)、反射路径 \(p\)、burst 帧 \(t\) 的观测，则

    \[
    Z_{d,p,j,t}(\mathbf u)=\mathcal D_{s_{d,p}}\!\left[
    k_{d,p,\mathbf u,t}*\mathcal W_{H_{d,p,j,t},\delta_{d,p}}(X_j)
    \right],
    \]

    \[
    \widehat Y_{d,p,j,t}
    =\mathcal J_{Q_d}\!\left\{
    \operatorname{clip}_{[0,1]}\!\left[
    b_{d,p,j,t}(\mathbf u)+g_{d,p,j,t}\odot m_{d,p,j,t}(\mathbf u)\odot
    h_{d,p,j,t}\!\left(C_{d,p}\!\left(Z_{d,p,j,t}(\mathbf u)\right)\right)
    +\varepsilon_{d,p,j,t}(\mathbf u)
    \right]\right\}.
    \]

    当同一照片含多条可见反射时，进入相机响应前的光学项扩展为

    \[
    Z_{d,j,t}(\mathbf u)=\sum_{p=1}^{P_d}\alpha_{d,p}(\mathbf u)\,
    \mathcal D_{s_{d,p}}\!\left[k_{d,p,\mathbf u,t}*
    \mathcal W_{H_{d,p,j,t},\delta_{d,p}}(X_j)\right].
    \]

    当前数据只可靠拟合主要可读路径，故本轮令 \(P_d=1\)；多路径权重 \(\alpha_p\) 留待下一轮单亮点扫描后估计。

    | 符号 | 含义 | 参数层级 |
    |---|---|---|
    | \(\mathcal W_{H,\delta}\) | \(H\) 表示裁剪、镜像和透视；\(\delta(\mathbf u)\) 表示镜片/光路引起的低阶曲面几何残差 | 姿态/帧级 \(H\) + 路径级二次残差场 |
    | \(m(\mathbf u)\) | 颜色混合后的逐通道有效空间透射/衰减场 | `c_core` 标定序列内逐帧二次 `log m` 已估计；跨姿态分布未定 |
    | \(k_{\mathbf u,t}\) | 合并光学、离焦、轻微运动和 ISP 锐化后的有效核；ISP 可使核含窄锐化/振铃分量 | 域/路径级基础核 + 自动对焦引起的帧级变化，可空间变化 |
    | \(\mathcal D_s\) | 相机采样与重采样 | 域/路径级 |
    | \(C\) | 黑/白规范化后的联合有效颜色算子 | v6 在基准、重复、左右水平和下移姿态的留一会话验证支持共享 3 x 3 矩阵；稠密 LUT 被 v4 留出误差否决 |
    | \(h_t\) | 固定黑、白并由中灰锚点确定拐点的逐帧单调 tone response | 每张标定图可估计；跨自动曝光/白平衡状态的分布未定 |
    | \(g\) | 自动曝光和白平衡造成的逐通道帧增益 | 帧级干扰量 |
    | \(b(\mathbf u)\) | 环境光、镜框散射、flare 和 JPEG 黑电平 | 首尾二次空间场已估计；帧间插值为暂定项 |
    | \(\varepsilon\) | 信号相关、通道相关的 JPEG 空间残差噪声 | 帧级随机量 |
    | \(\mathcal J_Q\) | JPEG 量化与色度子采样 | 编码级 |

    - **拟合目标。** 几何、PSF、光度和噪声先分模块估计，最后才做小范围联合细化，避免几何、模糊和 tone curve 相互补偿：

    \[
    \mathcal L(\theta)=
    \lambda_I\rho\!\left(M\odot(Y-\widehat Y)\right)
    +\lambda_{\nabla}\rho\!\left(M\odot(\nabla Y-\nabla\widehat Y)\right)
    +\lambda_E\left|w_E(Y)-w_E(\widehat Y)\right|+\mathcal R(\theta),
    \]

    其中 \(M\) 排除遮挡、饱和和其他反射路径，\(\rho\) 为稳健 Charbonnier/Huber 损失，\(w_E\) 为边缘宽度。对纯光学核，\(\mathcal R\) 约束非负、归一化和空间平滑；对包含 ISP 锐化的有效核，只约束 DC 增益为 1、负旁瓣有界和空间平滑。

    - **当前可识别参数。** `analysis/run_fit_v2.py` 已处理 `c_C` 的 42 张实拍，`analysis/run_fit_v21.py` 拟合了 `c_follow` 的 6 张补拍，`analysis/run_fit_v3.py` 按时间顺序拟合了 `c_core` 的 9 张核心标定图，`analysis/run_fit_v4.py` 拟合了 `c_color` 的 6 张联合颜色标定图，`analysis/run_fit_v5.py` 联合评估了基准、重复和左右水平姿态，`analysis/run_fit_v6.py` 再加入 `c_color_pose_down`，共评估 30 张跨姿态图。当前结果见 `analysis/results/v6_parameters.json` 与 `analysis/results/v6_report.md`；`analysis/results/v5_parameters.json`、`analysis/results/v4_parameters.json`、`analysis/results/v3_parameters.json` 和 `analysis/results/v2_1_parameters.json` 继续保留作水平姿态、单姿态颜色、tone、点目标宽翼和自动对焦时变核的历史对照。v1 的 B 组 120 张数据仍作为独立基线，不与本次不同姿态的坐标直接混合。

    | 47 mm 自动域点阵 | 源方点 | 局部主尺度（Jacobian 奇异值，相机 px / 源图 px） | 有限方点反卷积后的 PSF FWHM（相机 px） | 二次场/单应中位残差 | 状态 |
    |---|---:|---:|---:|---:|---|
    | W | 15 x 15 | \(0.2549\times0.5835\) | \(7.98\times10.24\) | 1.00 / 5.25 px | 几何与 PSF `estimated`，24/25 点有效 |
    | R | 15 x 15 | \(0.2389\times0.5516\) | \(8.39\times9.84\) | 1.30 / 3.78 px | `estimated`，23/25 点有效；49 mm EXIF 已按 \(47/49\) 归一化 |
    | G | 31 x 31 | \(0.2752\times0.6541\) | \(5.27\times12.56\) | 1.79 / 9.74 px | 几何 `provisional`，19/25 点有效；PSF 对窗口敏感 |
    | B | 31 x 31 | \(0.2645\times0.6034\) | \(5.76\times12.98\) | 1.21 / 6.66 px | 几何 `estimated`，24/25 点有效；PSF 对窗口敏感 |

    二次曲面场在四种点阵上均显著优于单应，因此 \(\delta(\mathbf u)\) 是已识别的必要项。31 x 31 补拍将 G/B 点中位对比度提高到 0.725/0.793，已解决“点基本看不见”的问题；但反卷积 minor 轴仍随测量窗口变化。`05` 连拍接受 38 个可靠小点，时变 PSF FWHM 中位数为 \(10.97\times12.72\) px，IQR 为 9.25--12.38 / 10.76--14.33 px；前 6 帧低对比、后 4 帧锁焦，证明 \(k_{\mathbf u,t}\) 不能退化为固定核。边缘梯度核心中位数仅 3 px，与宽点 PSF 并存，说明 ISP 锐化/振铃必须由经验有效核描述。

    `c_core` 的三张独立置换响应图与红/灰阶图固定 \(f_c(0)=0,f_c(1)=1\)、每帧 `mean(log m)=0` 后，灰阶设计矩阵满秩 30/30，条件数 13.59。R/G/B 三条无约束 tone 序列均单调，灰阶联合 MAE 为 0.031，因此 R/G/B tone 与标定帧内的二次 `log m` 均已可估计。v3 的固定 \(f(Ax)\) 颜色块 MAE 仍为 0.115，这一旧候选只保留作历史诊断。

    `c_color` 的三张置换训练色卡联合设计矩阵满秩 43/43，条件数 17.85，观测拟合 MAE 为 0.0294。EXIF 显示训练帧 ISO 为 1000/640/800，且中灰相对白色的响应随帧明显变化，因此仅使用逐帧乘性增益会把 ISP tone 漂移误算成颜色误差。v4 在每帧只用 K000/K128/K255 锚点估计 \(h_{d,p,j,t}\)，24 个 `V_` 颜色严格不参与训练。独立留出 JPEG 域 MAE 为：受约束 3 x 3 矩阵 0.0437，3 x 3 x 3 三线性 LUT 0.0493，故当前选择前者。所得联合有效矩阵为

    \[
    C(x)=
    \begin{bmatrix}
    0.4054&0.1087&0.4860\\
    0.1926&0.6240&0.1834\\
    0.0591&0.5498&0.3911
    \end{bmatrix}x.
    \]

    三行系数均非负且行和为 1。该矩阵是在锚点 tone 规范化后的路径级有效颜色域中定义的，不解释为 OLED、镜片或相机 ISP 的独立物理矩阵。

    v5 继续使用 `c_color_repeat`、`c_color_pose_left` 和 `c_color_pose_right`。每个序列的三张训练色卡只拟合训练节点，第 4 张色卡保持独立留出；右侧竖拍图逆时针旋转 90 度后进入统一坐标系。四组训练节点汇总矩阵在四组留出图上的 JPEG 域 MAE 均值/最大值为 0.0415/0.0556，留一会话交叉验证均值/最大值为 0.0420/0.0559，而逐姿态独立矩阵均值为 0.0412。共享矩阵与姿态专属矩阵几乎没有性能差距，故状态为 `supported_across_sampled_poses`。当前用于水平姿态泛化的汇总矩阵更新为

    \[
    C_{\rm pooled}(x)=
    \begin{bmatrix}
    0.3866&0.1376&0.4758\\
    0.1925&0.5775&0.2300\\
    0.0766&0.4683&0.4551
    \end{bmatrix}x.
    \]

    v6 将 `c_color_pose_down` 作为第五个完整会话：四张色卡和定位标记均完整可见，色卡帧逆时针旋转 90 度进入源图方向，前三张训练、第四张仍只作独立留出。五会话汇总矩阵在五张留出图上的 JPEG 域 MAE 均值/最大值为 0.0380/0.0565，下移姿态为 0.0235；留一会话均值/最大值为 0.0384/0.0571，状态仍为 `supported_across_sampled_poses`。当前汇总矩阵更新为

    \[
    C_{\rm pooled}^{(v6)}(x)=
    \begin{bmatrix}
    0.3888&0.1561&0.4551\\
    0.1851&0.5663&0.2486\\
    0.0840&0.4232&0.4928
    \end{bmatrix}x.
    \]

    因此 \(C\) 可在本次采样的水平及下移姿态间共享；几何映射、逐帧锚点 tone、增益、空间衰减和背景仍必须按姿态/帧条件化。本批下移色卡没有硬截断，只验证了下移姿态的颜色泛化，尚不能识别可见性边界或遮挡掩膜。

    作为历史对照，`c_follow` v2.1 的灰阶/颜色 MAE 为 0.022/0.136，v2 原全黑 burst 的 R 通道背景中位数为 0.1549；这些数值不覆盖本轮按时间配对的 v3 结果。

    首尾全黑图的二次背景场拟合 MAE 为 0.0042，源图中心背景 RGB 从 0.0745/0.0743/0.0667 漂移到 0.1556/0.1596/0.1421，证明 \(b_t(\mathbf u)\) 必须同时具有时间和空间变化。G/B 斜边共有 9/16 个 ESF 通过稳健拟合，窄边缘核心 FWHM 中位数为 G 4.37 px、B 5.76 px；它们与点目标得到的宽翼并存，继续支持含 ISP 锐化/振铃的非高斯有效核。

    - **仍不可识别的参数。** 下列量不能由当前 JPEG 强行确定：

        1. F11 截掉外侧定位角造成的绝对边界缺失已由内缩定位图解决；`c_core` 首尾中心检查点仍有 21.4/25.9 px 非线性残差，所以逐帧单应必须和已拟合二次几何残差场一起使用。
        2. 联合颜色矩阵已经通过基准、重复、左右水平和一次下移姿态的留一会话验证；向上姿态、真实硬截断边界、距离变化、跨环境和跨设备分布仍未确定。逐帧锚点 tone、空间衰减和背景必须保持条件化；首尾黑图只有两个时间端点，序列内部的时间插值仍是暂定项。
        3. G/B 斜边窄核心已有稳定估计，但点目标在 18/22/26 px 测量窗口下的宽翼仍变化明显，因此完整非高斯空间 PSF 场尚未成为稳定单值。
        4. 光学、离焦、运动和 ISP 锐化各自的核：JPEG 只能支持合并有效 \(k_{\mathbf u,t}\)，且没有真实对焦马达位置、同步 IMU 或 RAW/DNG。
        5. shot noise、read noise、固定图案噪声与 JPEG/去噪残差的独立参数，以及多反射路径数量和权重，仍不能由当前 JPEG 分开。

    经典的模糊--采样--噪声--JPEG 链可参考 [Real-ESRGAN](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html) 与 [BSRGAN degradation model](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.html)；RAW 噪声和 ISP 可识别性参考 [Unprocessing Images](https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html)；多重反射的移位衰减模型参考 [Reflection Removal Using Ghosting Cues](https://openaccess.thecvf.com/content_cvpr_2015/html/Shih_Reflection_Removal_Using_2015_CVPR_paper.html)。

    当前 47 mm 自动域已完成单姿态、左右水平和一次下移姿态的泛化验证。若继续扩展，下一轮只需要覆盖向上姿态/真实截断边界、距离或环境变化，不应再重复本轮水平与下移姿态。

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

    - 退化模拟由 `analysis/degradation/model.py` 的前向算子执行。参数装配采用“各参数块最新有效结果”，而不是把最高版本号当成对全部旧参数的替代：W/R 基础几何、点目标 PSF 和自动对焦时变范围来自 `v2_parameters.json`；G/B 点目标宽翼来自 `v2_1_parameters.json` 且保持 `provisional`；背景、单姿态光度和 G/B 斜边核心的后续校正来自 `v3_parameters.json`；v4 只保留颜色模型选择和独立留出证据；共享颜色矩阵及水平/下移姿态条件范围使用最新的 `v6_parameters.json`。各版本按参数状态合并，不再使用与设备无关的固定经验范围：

        1. 先采样 device/domain/path/pose，再采样 \(H\) 和已拟合的二次几何残差场 \(\delta\)。
        2. 只有状态为 `estimated` 的参数默认启用；`provisional` 参数必须通过显式开关进入消融实验，`not_identifiable` 参数不能用单点伪装成拟合值。
        3. 当前 47 mm 域按归一化视场位置采样 W/R 点目标 PSF，并按 `05` burst IQR 采样自动对焦状态；G/B 同时使用 v3 斜边窄核心与 v2.1 点目标宽翼，采用非高斯有效核或核心/翼混合模型。
        4. 当前实拍 JPEG 固定使用等效 quality 96 和 4:2:0；未来收集其他手机后再按设备条件化 JPEG 表，而不是统一随机到低质量区间。
        5. v6 的跨水平及下移姿态汇总颜色矩阵进入参数匹配合成集；逐帧 K000/K128/K255 tone、空间衰减和时间变化二次背景按 v6 实测会话范围采样。3 x 3 x 3 LUT 因 v4 留出误差更高而不启用。独立噪声、多路径、可见性边界和分解运动核仍不计入参数匹配合成集。
        6. 合成/实拍比较同时报告 ROI 内边缘 FWHM、尺度、颜色误差、噪声方差和 OCR/CER；未配准图像不报告误导性的全图 PSNR/SSIM。

    - 生成实现由协作者负责。当前仓库不提供批量生成器，只提供已经拟合的参数文件和 `analysis/degradation/model.py` 中的路径级前向算子。协作者负责参数适配、文本渲染、批量调度、数据划分和落盘；不得把尚未实现的批量入口写成当前仓库已有功能。低层调用契约为：输入是形状为 `(H,W,3)`、数值位于 `[0,1]` 的 float RGB，`output_size` 按 `(width,height)` 给出，输出仍是 `[0,1]` float RGB。

        ```python
        from analysis.degradation.model import DegradationParameters, degrade

        params = DegradationParameters(...)  # 由协作者的参数适配层构造
        degraded = degrade(image, params, seed=seed, encode_jpeg=True)
        ```

        参数适配层必须按下表读取，不再从报告正文抄录四舍五入后的数值：

        | `DegradationParameters` 项 | 参数来源与使用规则 |
        |---|---|
        | `homography`、`geometric_residual_coefficients` | 从 `v2_parameters.json` 的 `fit.geometry_and_spatial_psf` 和所选 v6 会话的 `response_grid_refinement` 读取。JSON 中单应为源像素到相机像素坐标，而前向算子接收归一化单应；若先裁出 ROI，应先左乘 ROI 平移，再按 `H_n = S_o^{-1} H_roi S_s` 转换，其中 `S_s = diag(W_s-1,H_s-1,1)`、`S_o = diag(W_o-1,H_o-1,1)`。几何残差也必须换算成输出宽高归一化位移。 |
        | `blur_fwhm_px`、`blur_angle_deg` | 默认使用 v2 中状态为 `estimated` 的 W/R 点目标以及 `fit.temporal_psf` 的同一空间样本/帧；G/B 窄核心取 v3 斜边结果，v2.1 点目标宽翼仍为 `provisional`，只在显式消融中启用。FWHM 两轴、角度和位置必须成组抽样。当前算子中的各向异性高斯核是基线近似，若协作者实现核心/翼混合核，必须在元数据记录 `kernel_mode`。 |
        | `color_matrix` | 固定读取 `v6_parameters.json` 的 `fit.cross_pose_generalization.pooled_color_matrix`；不得继续使用 v3 的旧颜色候选或 v4 中被留出误差否决的 LUT。 |
        | `tone_curve_levels`、`tone_curve_rgb`、`gain_rgb` | 先选 v6 会话和标定帧，再读取该帧 `anchor_tone_normalization` 的 `frame_midpoint_response_rgb` 与 `frame_gain_after_white_anchor_rgb`。使用三节点 `(0,0.5,1)`，每通道响应为 `(0, midpoint, 1)`；此时 `gamma_rgb=(1,1,1)`，避免重复施加 gamma。 |
        | `attenuation_coefficients` | v6 `training_node_fit.spatial_log_coefficients` 描述的是源图坐标中的 `log m`，而当前前向算子字段是输出坐标中的六项直接乘子。应先在目标网格计算 `exp(log m)`，再转换/拟合到输出六项基；禁止把这些数组原样复制到 `attenuation_coefficients`。 |
        | `background_coefficients`、`bias_rgb` | 优先从所选 v6 会话的 `temporal_quadratic_background.coefficients` 抽取一个实测端点场，并转换到输出坐标；v3 背景仅作单姿态对照。首尾端点之间的插值仍是暂定假设，默认不随机外推；`bias_rgb` 保持零，避免与背景常数项重复。 |
        | `noise_slope`、`noise_intercept` | 当前默认均为零。现有 JPEG 未分离 shot/read noise、去噪与压缩残差，不得填入任意经验噪声；如作消融，须单独标记 `provisional_noise=true`。 |
        | `jpeg_quality`、`jpeg_subsampling` | 当前设备固定为 `96` 和 Pillow 的 `2`（4:2:0）。参数匹配主数据集不得随机降到其他质量。 |

        采样时先从 `baseline/repeat/pose_left/pose_right/pose_down` 选择 `pose_session`，再在该会话内选择帧。共享矩阵 `C` 可以跨会话复用，但单应、tone、增益、空间衰减和背景必须保留会话/帧条件关系，禁止跨会话独立打乱后重新组合。对于一个 N 帧序列，`source_id`、device/domain/path、基础姿态和结构级 PSF 共享；帧级漂移、自动对焦状态、tone 与增益只在同一 `pose_session` 的实测支持内变化。

        每个样本必须写一条可回放的 JSONL 元数据，至少包含 `sample_id`、`sequence_id`、`frame_index`、`source_id`、`seed`、`model_version="v6"`、`domain="47mm"`、`path="primary_readable"`、`pose_session`、`parameter_sources`、完整的实际参数、`clean_path` 和 `degraded_path`。相同清晰图、元数据和 seed 必须逐像素复现相同的未压缩结果；训练、验证、测试集先按 `source_id` 切分，再扩增退化版本，防止同一文本内容跨集合泄漏。

        生成器验收不以肉眼相似为准。至少检查固定 seed 可复现、元数据可回放、输出有限且位于 `[0,1]`，并在配准 ROI 中将尺度、边缘 FWHM、背景 RGB、颜色 MAE 和 OCR/CER 分布与实拍比较。当前支持范围仅为本设备、47 mm、主要可读路径及已采样的水平/下移姿态；向上硬截断、可见性掩膜、多路径、距离、环境和跨设备变化不得默认混入主训练集。

    - 多帧序列生成采用“共享结构参数 + 帧级干扰量”：同一序列共享清晰底图、域、路径、基础投影、二次几何残差和空间 PSF 场；每帧从 `05` 实测分布采样视场漂移与自动对焦 PSF 状态，从 `00` 的可靠配准帧采样逐通道响应漂移。未分离的运动核和物理噪声不写成固定常数。

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
