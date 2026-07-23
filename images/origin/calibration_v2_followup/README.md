# v2 拟合后的最小补拍

本目录中的 5 张 PNG 均为 `2500 x 1600`、8-bit RGB。`00` 定位图需要在同一姿态开始和结束各拍一次，因此当前姿态实际拍 6 张。

## 当前姿态：6 张

1. `00_inset_asymmetric_locator.png`：普通单拍 1 张。
2. `01_permuted_response_A.png`：普通单拍 1 张。
3. `02_permuted_response_B.png`：普通单拍 1 张。
4. 再显示 `00_inset_asymmetric_locator.png`：普通单拍 1 张。
5. `03_pointgrid_G_5x5_31px.png`：普通单拍 1 张。
6. `04_pointgrid_B_5x5_31px.png`：普通单拍 1 张。

定位标记全部位于画面 10%--90% 的安全区域，避免 F11 全屏时再次截掉。两张响应图各含 56 个色块；8 个灰阶和 RGB/CMY 各重复 4 次，并且 A/B 图的同一位置绝不使用相同色块。精确位置与 RGB 值见 `pattern_manifest.json`。

G/B 大点阵使用 `31 x 31` 源像素方点，5 x 5 位置与原 calibration C 的 15 x 15 点阵相同。当前姿态用这两张替代原来的 G/B 小点阵，不要两种尺寸同时拍；拟合时会扣除已知的 31 x 31 方块支持。

## 泛用性姿态：22 张

再选择 4 个差异明显的拍摄姿态，每个姿态普通单拍以下 4 张，共 16 张：

1. 本目录的 `00_inset_asymmetric_locator.png`。
2. `../calibration_C/01_pointgrid_W_5x5_15px.png`。
3. `../calibration_C/05_field_point_edge.png`。
4. `../calibration_C/06-09` 中每个姿态使用一张不同文本图。

从上述 4 个姿态中选择几何畸变最大的 2 个，每个姿态再普通单拍以下 3 张，共 6 张：

- `../calibration_C/02_pointgrid_R_5x5_15px.png`
- `../calibration_C/03_pointgrid_G_5x5_15px.png`
- `../calibration_C/04_pointgrid_B_5x5_15px.png`

当前姿态 6 张，加泛用性姿态 22 张，共 28 张普通单拍。不需要使用定时连拍。

## 47 mm 自动域设置

- EV=-0.7。
- S、ISO、WB、F：自动。
- 长焦自适应：开启。
- 万物追焦：仅当它是进入或保持 47 mm 的必要条件时开启。
- HDR、AI 场景优化、夜景、美颜和滤镜关闭。
- 保存原始 JPEG 和 EXIF；不要混入 23/28/35 mm 手动主摄数据。

重新生成：

```bash
.venv/bin/python images/generate_calibration_v2_followup.py
```
