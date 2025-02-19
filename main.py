import warnings
import sys
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionImg2ImgPipeline
from generate_prompt import runner
from PIL import Image
# 禁用 UserWarning 和 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_img(img_path: str, size: tuple = (512, 512)) -> Image.Image:
    """
    加载并调整图像大小。
    :param img_path: 图像文件路径
    :param size: 图像的目标尺寸，默认为 512x512
    :return: 调整大小后的图像
    """
    # 打开图像文件
    image = Image.open(img_path)
    # 调整图像大小
    image = image.resize(size)
    return image


def initial_pipe(model_name: str = '') -> StableDiffusionImg2ImgPipeline:
    """
    初始化图像到图像（Img2Img）生成模型的管道。
    :param model_name: 预训练模型的路径或链接
    :return: 初始化的生成管道
    """
    # 使用预训练模型加载管道
    pipe = (StableDiffusionImg2ImgPipeline.
            from_single_file(pretrained_model_link_or_path=model_name,
                             config='models/sd_1_5'))
    # 将管道加载到GPU，可调整精度到torch.float32，但是生成会更慢
    pipe.to("cuda", dtype=torch.float32)
    return pipe


def generate_inpainted_img(pipe, prompt: str, original_img: Image.Image,
                           strength: float = 0.75) -> Image.Image:
    """
    使用给定的提示生成修复（inpainting）图像，将修复图像扩展到原始图像的外圈。
    :param pipe: 初始化的生成管道
    :param prompt: 用于生成图像的提示文本
    :param original_img: 原始图像，用于修复的基础图像
    :param strength: 控制生成图像与原始图像的相似度，值越大生成图像差异越小
    :return: 生成的修复图像，大小与原始图像一致，且修复部分扩展到外圈
    """
    # 计算扩展后的图像尺寸（比原始图像大）
    extended_size = (original_img.width + 512, original_img.height + 512)  # 增加外圈尺寸
    extended_img = original_img.resize(extended_size, Image.LANCZOS)  # 使用高质量的LANCZOS算法放大图像

    # 通过管道生成修复图像
    generated_image = pipe(prompt=prompt,
                           image=extended_img,  # 使用扩展后的图像
                           strength=strength,  # 强度
                           num_inference_steps=60,  # 推理步骤数，控制生成的细节
                           guidance_scale=7.5).images[0]  # 引导尺度，影响生成结果的创意程度

    return generated_image


def display(original_img: Image.Image, res_img: Image.Image):
    """
    显示原始图像和生成图像，并保存为文件。
    :param original_img: 原始图像
    :param res_img: 生成图像
    """

    # 保存结果图像到文件
    res_img.save('output.png')

    # 创建两个子图来显示原图和生成图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('original image')
    axes[0].axis('off')  # 关闭坐标轴

    # 显示生成图像
    axes[1].imshow(res_img)
    axes[1].set_title('generated image')
    axes[1].axis('off')  # 关闭坐标轴

    # 展示图像
    plt.show()


def main(img_path='datasets/img3.jpg'):
    """
    主函数：加载图像，生成修复图像，并显示结果。
    """
    # 使用生成的提示文本生成描述
    prompt = runner(img_path)
    print(f"生成的图像描述为: {prompt}")

    # 加载原始图像并调整大小
    original_img = load_img(img_path)

    # 初始化图像到图像生成管道
    pipe = initial_pipe("models/image_painting/ChineseLandscapeArt_v10.safetensors")

    # 生成修复图像
    gen_img = generate_inpainted_img(pipe, prompt, original_img, strength=0.65)

    # 显示并保存最终结果
    display(original_img, gen_img)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _path = sys.argv[1]
    else:
        _path = 'datasets/img3.jpg'
    print(f"输入的图像地址：{_path}")

    # 执行主函数
    main(_path)
