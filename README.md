# NeuroGuide — TI 无创深脑刺激仿真规划平台

NeuroGuide 是一款面向时域干涉（Temporal Interference，TI）无创深脑刺激的仿真规划与可视化工作站。它将 TI 电场仿真结果叠加在 2D 切面和 3D 体渲染上，支持模板空间（MNI152）与个体 T1 影像之间的配准，并内置面向 TI 的特定靶点刺激方案和一键仿真报告生成功能。

## 功能概览

- **2D/3D 四视图**：矢状面、冠状面、轴状面三视图 + 3D 体渲染实时联动
- **TI 电场叠加**：灰质电场、白质电场以可调色表/透明度叠加于解剖图像
- **标准高效模式**：内置 MNI152 模板 T1，选择疾病和靶点即可即刻查看对应的 TI 仿真结果
- **个体精准模式**：加载外部个体 T1，通过刚性→仿射→B样条三步配准将模板空间仿真结果映射到个体空间
- **TI 特定靶点方案**：内置约 30 个面向 TI 无创深脑刺激的特定靶点方案（GPi、STN、Vim、DLPFC、NAc 等），支持一键定位与方案查看
- **仿真报告生成**：自动生成 PDF 报告，包含患者信息、仿真参数、三视图切面和电极位置图
- **患者记录管理**：本地 JSON 数据库，支持新建、删除，可关联至仿真报告
- **单文件可执行**：PyInstaller 打包，开箱即用

## 快速开始

### 直接运行（打包版）

双击 `dist/NeuroGuide ver3.x/NeuroGuide ver3.x.exe` 即可启动，无需安装 Python 或任何依赖。

### 源码运行

#### 环境要求

- Python 3.10+
- Windows 10/11 (64-bit)

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 数据准备

程序运行需要以下数据目录与可执行文件同级：

| 目录 | 说明 |
|------|------|
| `mri/T1.nii` | MNI152 标准脑模板 |
| `sample_data/{疾病}/{靶点}/` | TI 特定靶点方案的灰/白质电场仿真结果（NIfTI 格式） |
| `assets/electrode_maps/` | 10-10 电极位置图（可选） |

#### 启动

```bash
python main.py
```

## 项目结构

```
code_revise/
├── main.py                              # 主程序入口
├── NeuroGuide_revise.spec               # PyInstaller 打包配置
├── requirements.txt
├── core/
│   ├── __init__.py
│   └── engine.py                        # 数据引擎（NIfTI 加载/叠加/配准）
├── ui/
│   ├── __init__.py
│   └── canvas.py                        # 画布组件（2D 切片 / 3D 体渲染）
├── tools/
│   ├── __init__.py
│   ├── register_template_to_individual.py  # 模板→个体配准
│   └── generate_pdf.py                    # PDF 报告生成
├── mri/                                 # 标准模板 T1.nii（需自行放入）
├── sample_data/                         # TI 特定靶点方案仿真数据（需自行放入）
├── assets/                              # 电极图等静态资源（可选）
└── dist/                                # 打包输出目录
```

## 打包说明

```bash
pyinstaller NeuroGuide_revise.spec --noconfirm
```

输出至 `dist/NeuroGuide ver3.x/`。

## 依赖

| 包名 | 用途 |
|------|------|
| numpy | 数值计算 |
| scipy | 科学计算、图像插值 |
| nibabel | NIfTI 医学影像读写 |
| SimpleITK | 医学图像配准 |
| PyQt6 | GUI 界面框架 |
| matplotlib | 2D 图像渲染 |
| pyvista | 3D 体渲染 |
| pyvistaqt | PyVista Qt 适配 |
| reportlab | PDF 报告生成 |
| Pillow | 图像处理 |

## dist分发
github上传存在2GB限制，因此分发版请至百度网盘下载：https://pan.baidu.com/s/18lIFF9CW8fRowpbFHrxxXQ?pwd=y6kt
