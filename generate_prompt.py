import warnings
# 禁用 UserWarning 和 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys

from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar.modeling.vinvl import VinVLFeatureExtractor
import cv2
import random
import torch
import yaml
import spacy


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70,
                 max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
                 is_train=False):

        """初始化函数，定义了训练和推理过程的相关配置
        Args:
            tokenizer: 用于文本处理的tokenizer。
            max_img_seq_length: 最大图像序列长度。
            max_seq_length: 最大文本序列长度。
            max_seq_a_length: 最大字幕序列长度。
            is_train: 是否为训练模式。
            mask_prob: 输入token的掩码概率。
            max_masked_tokens: 每个句子中最大掩码的token数量。
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1):
        """
        将图像和文本数据转换为适合模型的输入格式。
        Args:
            :param text_a: 主体文本。
            :param img_feat: 图像特征。
            :param text_b: 可选的第二段文本。
            :param sequence_b_segment_id: 第二段文本的segment ID。
            :param sequence_a_segment_id: 主体文本的segment ID。
            :param pad_token_segment_id: 填充token的segment ID。
            :param cls_token_segment_id: CLS token的segment ID。
        """

        # 训练模式下处理文本
        if self.is_train:
            # 对主体文本进行分词
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # 在推理阶段，使用掩码填充文本
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)

        # 截断文本长度，确保长度不超过最大限制
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        # 生成token序列，包含[CLS]和[SEP]标记
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)

        # 如果有第二段文本，合并并填充
        if text_b:
            # pad text_a以确保其长度固定，便于推理
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            # 对第二段文本进行分词
            tokens_b = self.tokenizer.tokenize(text_b)
            # 截断第二段文本，确保总长度不超过最大限制
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)

        # 如果是训练模式，执行掩码处理
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # 随机掩码文本中的部分词汇，忽略[CLS]标记
            candidate_masked_idx = list(range(1, seq_a_len))  # 仅对文本A进行掩码
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]

            # 根据一定的概率对掩码位置进行处理
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80%的几率替换为 ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10%的几率替换为一个随机词汇
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10%的几率保持不变
                    pass

            masked_pos[masked_idx] = 1
            # 对掩码词汇进行填充，确保掩码数量符合要求
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                                               (self.max_masked_tokens - num_masked))
            # 将掩码词汇转换为ID
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            # 如果不是训练模式，所有掩码位置都设置为1
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # 填充右侧以适应图像描述任务
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        # 将token转换为对应的ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 图像特征处理
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            # 如果图像特征长度超过最大限制，截取前max_img_seq_len个特征
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            # 如果图像特征长度不足，使用零填充
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # 准备注意力掩码
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for L-R:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        # 转换为tensor并返回
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # 返回处理后的数据字典
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': segment_ids,
            'img_feats': img_feat,
            'masked_pos': masked_pos
        }


def init_model(caption_checkpoint='models/image_caption/mengzi-oscar-base', extractor_cfg='models/feature_extractor/sgg_configs/bert_cfg.yaml'):
    # 图像特征提取器初始化
    feature_extractor = VinVLFeatureExtractor()
    with open(extractor_cfg, 'r') as fp:
        extractor_config = yaml.load(fp, Loader=yaml.BaseLoader)

    # 图像描述模型初始化
    caption_config = BertConfig.from_pretrained(caption_checkpoint)
    caption_config.output_hidden_states = False
    tokenizer = BertTokenizer.from_pretrained(caption_checkpoint)
    model = BertForImageCaptioning.from_pretrained(caption_checkpoint, config=caption_config)

    tokens = [tokenizer.cls_token, tokenizer.sep_token,
              tokenizer.pad_token, tokenizer.mask_token]
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = tokenizer.convert_tokens_to_ids(tokens)
    extractor_config.update({'bos_token_id': cls_token_id,
                             'pad_token_id': pad_token_id,
                             'eos_token_ids': [sep_token_id],
                             'mask_token_id': mask_token_id})

    return {'feature_extractor': feature_extractor,
            'caption_tensorizer': CaptionTensorizer(tokenizer=tokenizer),
            'caption_tokenizer': tokenizer,
            'caption_model': model,
            'extractor_params': extractor_config}


def image_caption(img_path, feature_extractor, caption_model, caption_tokenizer, caption_tensorizer, extractor_params):
    """
    feature_extractor 是用于提取图像特征的工具或模型。在这里，它是 VinVLFeatureExtractor 类的实例，专门用于从图像中提取视觉特征。
    具体来说，feature_extractor 的作用是对输入的图像进行处理，识别出其中的物体，并生成一组与图像内容相关的特征表示。
    image_features 是从 feature_extractor 提取出的图像特征。具体来说，这些特征可能是经过卷积神经网络（CNN）等模型处理后的高维向量，
    能够捕捉图像中的关键内容，如物体、场景结构等。这些特征将用于模型的进一步处理，比如生成描述图像内容的文本。
    od_labels 是与图像中的物体检测（Object Detection，简称 OD）相关的标签。这些标签指示了在图像中被检测到的各个物体的类别或类别名称。
    :param img_path: 图像文件路径
    :param feature_extractor: 图像特征提取器
    :param caption_model: 图像描述生成模型
    :param caption_tokenizer: 生成描述时使用的分词器
    :param caption_tensorizer: 用于将输入转换为模型格式的转换器
    :param extractor_params: 额外的特征提取参数
    :return: 生成的图像描述
    """

    # 读取并预处理图像：将图像从BGR转换为RGB格式
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # 使用特征提取器提取图像中的物体检测结果
    object_detections = feature_extractor([image])[0]

    # 获取图像特征和物体检测标签
    image_features, od_labels = object_detections["img_feats"], object_detections["od_labels"]

    # 将模型移动到GPU设备上
    device = torch.device('cuda')

    # 将caption模型移动到GPU并切换到评估模式
    model = caption_model.to(device)
    tokenizer = caption_tokenizer
    model.eval()

    # 使用caption_tensorizer将图像特征和物体检测标签转换为模型所需的输入格式
    inputs = caption_tensorizer.tensorize_example(text_a=None, img_feat=image_features, text_b=od_labels)

    # 对输入进行批处理维度扩展并将其传递到GPU
    inputs = {
        "input_ids": inputs['input_ids'].unsqueeze(0).to(device),  # Batch dim
        "attention_mask": inputs['attention_mask'].unsqueeze(0).to(device),
        "token_type_ids": inputs['token_type_ids'].unsqueeze(0).to(device),
        "img_feats": inputs['img_feats'].unsqueeze(0).to(device),
        "masked_pos": inputs['masked_pos'].unsqueeze(0).to(device),
    }

    # 更新输入字典，加入额外的特征提取参数
    inputs.update(extractor_params)

    # 禁用梯度计算，进行前向推理计算
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取生成的所有描述和相应的置信度
    all_caps = outputs[0]  # batch_size * num_keep_best * max_len
    all_confs = torch.exp(outputs[1])  # 对数转概率

    # 存储生成的描述和置信度
    res = []
    for caps, confs in zip(all_caps, all_confs):
        for cap, conf in zip(caps, confs):
            # 解码生成的描述
            cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
            # 将描述和置信度存储到结果列表中
            res.append({'caption': cap, 'conf': conf.item()})

    # 清理不再使用的对象
    del model, tokenizer, feature_extractor

    # 根据置信度对结果进行排序，选择最有信心的描述
    res = sorted(res, key=lambda x: x['conf'], reverse=True)[0]

    # 清理描述中的多余空格
    res = ''.join(res['caption'].split())

    # 后处理生成的描述，去除冗余部分
    res = post_process_prompt(res)

    # 返回最终的图像描述
    return res


def post_process_prompt(sentence):
    # 结果列表，存放去除人物描述后剩下的句子部分
    nlp = spacy.load('models/nlp_tool/zh_core_web_lg-3.8.0')

    # 人物相关词汇
    person_keywords = ["人", "男", "女"]

    # 处理每个句子
    doc = nlp(sentence)

    # 用一个列表存放非人物描述的部分
    non_human_parts = []

    # 判断每个词是否与人物描述相关
    for token in doc:
        # 如果该词是动词且其主语是人物
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':  # 找到动词
            subject = [child for child in token.lefts if child.dep_ == 'nsubj']  # 寻找主语
            if subject and subject[0].text in person_keywords:  # 如果主语是PERSON
                non_human_parts = []
                continue  # 跳过，认为这是与人的行为相关的描述

        # 如果不是与人类相关的描述，保留
        non_human_parts.append(token.text)

    return "".join(non_human_parts)


def runner(img_path='datasets/img1.png'):
    models = init_model()
    return image_caption(img_path, **models)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _path = sys.argv[1]
    else:
        _path = 'datasets/img1.png'
    print(f"输入的图像地址：{_path}")
    print(runner(_path))
