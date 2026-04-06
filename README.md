# DDPM MNIST 项目

这是一个在 MNIST 上实现扩散模型生成的项目，目标是用轻量级 DDPM 复现从噪声逐步还原数字图像的生成过程，并记录模型迭代带来的效果变化。

## 项目定位

- 在 MNIST 上训练扩散模型
- 复现 forward diffusion 和 reverse denoising 流程
- 记录不同版本模型的生成质量变化
- 输出阶段性生成结果，观察训练演化过程

## 核心技术点

- forward diffusion
- reverse denoising
- noise prediction objective
- sinusoidal time embedding
- lightweight UNet

## 目录结构

```text
ddpm_mnist/
├── models/
│   └── unet.py            # UNet 模型
├── utils/
│   └── diffusion_utils.py # 扩散过程工具
├── outputs/               # 生成结果保存目录
├── train.py               # 训练入口
├── sample.py              # 采样入口
├── requirements.txt       # 依赖列表
└── README.md
```

## 训练流程

训练入口为：

```bash
python train.py
```

训练过程包括：

1. 加载 MNIST 数据集
2. 将图像归一化到 `[-1, 1]`
3. 采样随机时间步
4. 对图像加噪得到 `x_t`
5. 预测噪声并计算 MSE loss
6. 进行反向传播和参数更新
7. 定期保存训练状态和结果

## 结果记录

项目中保留了三个阶段的结果图，用于展示模型从噪声到数字结构逐步成形的过程：

- `outputs/version_1.png`
- `outputs/version_2.png`
- `outputs/version_3.png`

另外还保存了汇总样例：

- `outputs/generated_samples.png`

## 实验演化

该项目的实验演化已经比较清晰：

- 第一版训练后主要输出噪声
- 第二版降低学习率后，训练趋于稳定，但仍以笔画碎片为主
- 第三版升级 time embedding 和 UNet 后，开始生成可辨认数字

这条演化路径说明，扩散模型在小数据集上同样依赖网络结构和训练策略的稳定配合。

## 备注

- `sample.py` 用于后续采样和生成扩展
- `img.png` 保留为项目示意图
- `outputs/` 中的结果图用于阶段性对比，不属于训练参数文件

