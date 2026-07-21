# Smart-glasses effective degradation fit v1

## Scope

This report fits the selected readable reflection path in JPEG space. It does not separate OLED, lens, sensor, and ISP physics.

The path-level effective forward model is:

\[
\widehat Y_{d,p,j,t}=\mathcal J_{Q_d}\left\{\operatorname{clip}_{[0,1]}\left[g_{d,p,j,t}\left(A_{d,p}\mathcal D_{s_{d,p}}\left[k_{d,p,\mathbf u}*\left(m_{d,p}(\mathbf u)\odot\mathcal W_{H_{d,p,j,t}}(X_j)\right)\right]+b_{d,p,j,t}(\mathbf u)\right)+\varepsilon_{d,p,j,t}\right]\right\}.
\]

For multiple visible copies, replace the single optical path by \(Z(\mathbf u)=\sum_p\alpha_p(\mathbf u)\mathcal D_{s_p}[k_{p,\mathbf u}*(m_p\odot\mathcal W_{H_p}(X))]\). The current data supports only the selected primary path.

## Dataset and encoding

- Groups: `12`; frames: `120`.
- JPEG quality 96; chroma subsampling `4:2:0`.
- Quantization-table SHA-256: `4fc84b9dccb32e42b18b786b0fe3c74267269ab6d2b4db34cba7085e11d7cb04`.

## 47 mm domain

- Domain ID: `supermacro_47_primary`.
- Local scale (camera/source): `0.4898 x 0.4192`.
- Effective blur FWHM (camera px): `7.82 x 9.68`.
- Scale status: `estimated`; blur status: `estimated`; color status: `provisional`; noise status: `provisional`.

### Identifiability

- `display_camera_response`: `not_identifiable`. auto exposure and white balance vary across separately displayed targets
- `independent_sensor_noise`: `not_identifiable`. JPEG-only bursts confound sensor noise with demosaicing, denoising, sharpening, and compression
- `absolute_reflection_geometry`: `not_identifiable`. captured geometry charts are periodic and lack an asymmetric absolute orientation marker

## 69 mm domain

- Domain ID: `supermacro_69_primary`.
- Local scale (camera/source): `0.4738 x 0.5046`.
- Effective blur FWHM (camera px): `6.00 x 6.00`.
- Scale status: `estimated`; blur status: `provisional`; color status: `not_identifiable`; noise status: `not_identifiable`.

### Identifiability

- `display_camera_response`: `not_identifiable`. auto exposure and white balance vary across separately displayed targets
- `independent_sensor_noise`: `not_identifiable`. JPEG-only bursts confound sensor noise with demosaicing, denoising, sharpening, and compression
- `absolute_reflection_geometry`: `not_identifiable`. captured geometry charts are periodic and lack an asymmetric absolute orientation marker

## Validation and diagnostic interpretation

- ROI overlays verify that configuration, rather than core code, owns path coordinates.
- Comparison figures show a canonical effective synthesis beside a resized real ROI.
- Their unregistered difference is diagnostic only and is not reported as independent PSNR/SSIM validation.
- The fixed text target is held out from PSF fitting; its current role is qualitative content-domain validation.

## 不可识别的参数

- Absolute crop, mirror state, full homography, reflection-path count, and path weights are unresolved because the geometry chart is periodic.
- A unique response curve, color matrix, spatial attenuation, and black level are unresolved because auto exposure/white balance changed across targets.
- Shot/read/fixed-pattern noise cannot be separated from ISP denoising, sharpening, and JPEG without RAW and controlled gray bursts.
- Spatially varying PSF, controlled motion blur, and cross-device/pose distributions require broader calibrated captures.

## 下一轮拍摄

| Priority | Capture | Frames per pose | Identifies |
|---|---|---:|---|
| P0 | 每个目标前后的非对称定位点阵 | 各 10 | 绝对裁剪、镜像、\(H_{p,t}\)、路径身份 |
| P0 | 单亮点扫描：白/R/G/B 亮点放在屏幕 5 x 5 位置 | 每位置每颜色 10 | \(P, H_p, \alpha_p\)、空间 PSF、色散 |
| P0 | 同一帧内黑块 + 16 级灰阶 + RGB/CMY/灰色色块和 fiducial | 10 | \(g,A,m,b\) |
| P0 | 3 x 3 位置稀疏点和水平/垂直/对角斜边 | 各 10 | \(k_{\mathbf u}\)、MTF、ringing |
| P0 | 棋盘、逐像素反相棋盘、全黑图；同曝光不移动 | 各 10 | 加性 flare 与乘性衰减 |
| P1 | 每个 ISO/曝光档的黑场和多个灰阶 burst | 各 30 | \(\operatorname{Var}(\varepsilon_c\mid\mu_c)=a_c\mu_c+b_c\) |
| P1 | 关键标定图的 RAW/DNG + JPEG 同步对 | 各 10 | 传感器、ISP、tone curve、JPEG 分层 |
| P1 | 受控运动、不同快门时间的点阵 | 各 10 | 条件运动核 |
| P1 | 至少 20 张独立随机文本图 | 每张 5--10 | OCR/内容泛化验证 |

Generalization protocol: for each device, collect full calibration at nine anchor poses and asymmetric-grid plus random-text pairs at at least 30 coverage poses. Cover near/mid/far distance, yaw, pitch, and dark/office/strong-side-light conditions; use at least three devices or glasses and three independent sessions.
