<div align="center">
  <img src="assets/branding/neuroguide_mark.png" alt="NeuroGuide Logo" width="160">
  <h1>NeuroGuide</h1>
  <p>从个体 MRI 到 TI / tES 电场仿真与方案评估</p>
  <p>Windows 桌面软件 · 当前分发版本 V4.5</p>
</div>

NeuroGuide 将影像导入、个体头模型构建、靶区选择、电极方案规划、电场显示和报告导出整合到同一工作区，帮助医生及研究人员查看、比较不同方案的仿真结果。

## 获取软件

**Windows 分发版：[夸克网盘下载](https://pan.quark.cn/s/7c774f15b867)**

**下载密码请联系 [xuxie@zju.edu.cn](mailto:xuxie@zju.edu.cn) 获取。**

完整解压后运行 `NeuroGuide-V4.5.exe`。请保留同目录的 `NeuroGuideWorker.exe`、`_internal` 及其他资源，不能只复制主 EXE。分发版包含 Python 运行环境、推理依赖及所需模型资源，无需自行安装 Python 或 MATLAB Runtime。

详细操作见 [V4.5 医生使用手册](docs/使用说明_V4.5_医生版.txt)，分发包内也提供 `使用说明.txt`。

## 功能

| 功能 | 操作与结果 |
| --- | --- |
| 个体影像导入 | 支持 T1 MRI 的 `.nii`、`.nii.gz`，也支持从 DICOM 文件夹选择序列后转换导入 |
| DICOM 序列确认 | 展示患者、检查日期、序列描述、模态、文件数等信息及样本预览，由医生确认 T1；只转换选中的序列 |
| 个体模型构建 | 完成组织处理、电极定位和导联场计算；检查并复用已有完整模型 |
| 靶区选择 | 在个体 MRI 上点选或输入 RAS 毫米坐标；也可准备 AAL3 图谱后搜索、选择一个或多个脑区 |
| TI 方案 | 自动搜索双电极对方案，或手动设置两组电极对及各通道电流后计算 |
| tES 方案 | 自动规划多电极方案，或手动输入逐电极带符号电流；检查电流平衡与设置的约束 |
| 结果显示 | 查看三个方向的 MRI 切片、2.5D 切面、电场叠加、靶区统计和电极示意；调整色表、透明度与显示范围 |
| 仿真报告 | 导出包含患者信息、靶区、电极、电流、统计和影像切片的 PDF |
| 患者记录 | 在本机记录基本信息，生成报告时选择关联记录；影像归属仍需人工核对 |
| 参考影像演示 | 分发版提供内置 MRI，便于熟悉完整操作流程 |

## 一次完整操作

1. 启动软件，点击 **新建刺激方案**。
2. 在 **1 影像** 选择个体 T1 MRI，或使用 **导入 DICOM…**。首次体验也可使用内置参考 MRI。
3. 浏览完整影像，核对患者、序列、覆盖范围和方向，再点击 **生成电场模型**。
4. 模型就绪后选择 **TI** 或 **tES**，在 **2 靶区** 中确定评价区域。
5. 点击 **自动规划并查看电场**，或在 **手动方案** 中输入方案并计算。
6. 检查电场分布、靶区统计及电极位置，再导出仿真报告。

### DICOM 如何避免选错序列

软件扫描文件夹及子文件夹，**不会默认选择序列，也不会根据名称替医生判定 T1**。医生选择序列、核对患者及检查信息并勾选 T1 确认框后，才能执行“转换并载入所选序列”。

更换序列需要重新确认；转换警告会再次提示；产生多份可用影像时需选择具体输出。原始 DICOM 保留，本地转换不需要上传影像。样本预览不代表完整体积，也不用于核对左右方向。转换完成后自动打开已确认的影像，**建模仍需医生检查后手动开始**。详情见 [DICOM 导入说明](docs/dicom_import.md)。

## 公开源码与 BrainLFNet

本仓库提供 NeuroGuide 的界面、流程编排、DICOM 导入、TI / tES 合成与优化、可视化、报告、测试及构建脚本。

**`BrainLFNet/` 暂不上传：相关论文尚未发表，该部分代码、模型权重及配套资源暂不在本仓库公开。** 后续公开安排以项目更新为准。

因此，**仅克隆本仓库不能运行完整的个体建模及推理流程，也不能直接构建与分发版等价的完整软件**。安装 Python 依赖不会补齐未公开的 BrainLFNet。希望直接使用完整软件，请获取上方分发版；有关模型资源的问题请联系 [xuxie@zju.edu.cn](mailto:xuxie@zju.edu.cn)。

仓库不包含 PPT、分发 EXE / ZIP、构建产物、患者记录、原始 MRI / DICOM、个体计算结果或模型权重。Logo、文本配置与文档作为项目资源保留。完整软件所需的影像模板、图谱和 DICOM 转换程序不作为源码上传，参见 [运行资源说明](docs/source_resources.md)。

## 源码结构

```text
main.py                 桌面程序入口
core/                   影像与任务流程、DICOM、TI / tES、缓存及运行时逻辑
ui/                     影像工作区、方案面板、DICOM 确认窗口和结果显示
tools/                  后台任务、报告、开发与受保护构建工具
tests/                  回归测试
assets/branding/        软件 Logo 与版本信息
docs/                   医生使用手册及源码资源说明
NeuroGuide_revise.spec   PyInstaller 打包配置
```

`BrainLFNet/` 是完整流程所需的外部目录，目前不随仓库提供。

## 开发环境

当前 Windows 分发版使用 Python 3.10 和 CUDA 12.8 对应的 PyTorch 运行库。源码开发建议先建立独立虚拟环境：

```powershell
git clone https://github.com/qaas-bci-lab/NeuroGuide.git
cd NeuroGuide
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-individual.txt
```

以上步骤只安装公开 Python 依赖。运行完整流程之前，还需要合法取得并配置未公开的 BrainLFNet 和 [运行资源](docs/source_resources.md)。资源齐全后可使用：

```powershell
.\.venv\Scripts\python.exe main.py
```

`requirements-cu128.txt` 用于匹配现有 CUDA 12.8 构建环境，并非所有显卡和开发环境的通用安装方案。部分测试依赖 BrainLFNet、图谱或影像资源；公开仓库不宣称在资源缺失时能够通过全部测试。

### 构建分发版

V4.5 使用 **Cython 编译项目模块 + PyInstaller 打包**。入口脚本为 `tools/build_protected.py`，构建依赖见 `requirements-build-protected.txt`。完整构建需要 Windows x64、匹配的 MSVC C++ 工具链、运行依赖及未公开资源。分发第三方组件时须保留对应许可。

Cython 原生编译提高常规解包和反编译成本，不等于不可逆向的加密；模型权重并未因此加密。

## 使用边界与数据

NeuroGuide 提供电场仿真与方案评估，不连接或控制刺激设备，不输出真实刺激。默认参数是软件初始设置，不是针对患者的治疗建议。电场结果需结合影像质量、模型适用性及既定诊疗／研究方案解释，不能仅凭电场分布确定疗效或实际刺激安全性。

TI 界面不设置刺激频率或时长；tES 当前采用直流或同频 0°/180° 相位交流的幅值模型。实际设备、电流、频率、时长和操作流程需另行核定。

患者记录、转换影像及仿真结果在本地保存，仍需按所在机构的数据管理要求处理。请勿将患者信息、原始影像、模型权重或计算结果提交到公开仓库。

## 联系

软件使用、下载密码与项目交流：**[xuxie@zju.edu.cn](mailto:xuxie@zju.edu.cn)**。
