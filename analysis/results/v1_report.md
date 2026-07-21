# Smart-glasses effective degradation fit v1

## Scope

This report fits the selected readable reflection path in JPEG space. It does not separate OLED, lens, sensor, and ISP physics.

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

## 下一轮拍摄

1. 每个目标前后拍摄非对称定位点阵，且不移动相机或眼镜。
2. 锁定曝光、ISO、对焦、白平衡和焦段，在同一帧拍摄黑场、16 级灰阶和彩色色块。
3. 在 3 x 3 视场位置拍摄稀疏点与水平/垂直/对角斜边，估计空间变化 PSF/MTF。
4. 每个曝光设置拍摄 30 帧黑场和灰阶 burst，并保存 RAW/DNG 与 JPEG 配对。
5. 每个姿态拍摄至少 20 张独立随机文本图，每张保留 5--10 帧。
6. 覆盖设备、距离、yaw、pitch、环境照度和独立拍摄 session，并记录 pose/path ID。
