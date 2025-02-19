
# 图像描述与图像生成项目

## 项目简介

本项目包含两个主要功能：

1. **图像描述**：通过预训练的 BERT 模型（`BertForImageCaptioning`）生成图像的描述文本。我们使用了 `Oscar` 库和 `VinVLFeatureExtractor` 进行图像特征提取，并通过训练模型生成准确的图像描述。
2. **图像生成**：结合图像描述和原始图像进行图像生成。通过预训练的 Stable Diffusion 模型（`ChineseLandscapeArt`）辅助生成新的图像，形成图像到图像的转换（Image-to-Image generation）。

## 项目结构

```
image_captioning_project/
├── datasets/               # 数据存储
│   ├── img.jpg             # 图像文件
├── weights/                # 预模型文件夹
│   ├── feature_extractor/  # 图像特征提取器——VinVL
│   └── image_caption/      # 图像描述——MengZi
│   └── nlp_tool/           # prompt优化过滤
│   └── image_painting/     # 图像生成——ChineseLandscapeArt
├── main.py                 # 项目主程序（调用图像描述+图生图）
├── generate_prompt.py      # 图像描述程序
├── requirements.txt        # 项目依赖文件
└── README.md               # 项目说明文档
```

## 安装依赖

确保你的环境已安装以下依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 图像描述生成

首先，使用 `generate_prompt.py` 脚本生成图像的描述。脚本会加载预训练的 BERT 模型，并提取图像特征来生成图像的自然语言描述。

```bash
python generate_prompt.py  "path/to/your/image.jpg"
```

该命令将会输出类似如下的图像描述：

```
"一只在草地上玩耍的狗，旁边有一颗树"
```

### 2. 图像生成

将图像描述与原始图像一起输入到图像生成模型中，生成的图像将融合图像的内容与其描述信息。

```bash
python main.py "path/to/your/image.jpg" 
```
该命令将会根据给定的描述生成新的图像，并保存为输出文件`output.png` 。

### 3. 训练模型（可选）

1. **训练图像描述模型**：[查看 Mengzi](models/image_caption/Mengzi-Oscar.md)
2. **训练图像生成模型**：[查看DeviantArt](https://civitai.com/models/120298/chinese-landscape-art)

### 4. 下载图像生成模型

[Chinese Landscape Art](https://civitai.com/models/120298/chinese-landscape-art)

---
