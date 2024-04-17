# -*- coding: utf-8 -*-
# 数据处理
# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

# POS是情感极性，great是情绪标签，wine list是方面词，drink style是方面类别
# 四元组[['Food', 'food quality', 'positive', 'excellent']]
# 四元组[方面词，方面类别，情感极性，意见词]

# 情感tag对应情感词
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}   # 没用上
# 情感tag对应情绪词
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}   # aste任务使用

# 情感词对应情绪词
# ASQP的目标构造由于每个情绪四方的方面类别ac和意见词o已经是自然语言形式，它们的投影函数只是保持原始格式： Pc (ac) = c和Po (o) = o
# 对于情感极性，投影如下：
# sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}   # tasd和asqp任务使用
sentword2opinion = {'love': 'love', 'joy': 'joy', 'approval': 'approval',  'neutral': 'ok',
                    'disappointed': 'disappointed', 'disgust': 'disgust', 'angry': 'angry'}  # 本文自己用的不是情感极性而是情绪极性

# 方面类别list：位置、价格、质量等等
"""aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']"""

aspect_cate_list = ['service',
                    'price',
                    'food',
                    'ambience',
                    'anecdotes/miscellaneous']


# 从文件路径按行读数据，每行数据都是：句子####情感四元组。返回list(句子),list(四元组)
#def read_line_examples_from_file(data_path, silence):
def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))  # service, angry
    #if silence:
     #   print(f"Total examples = {len(sents)}")
    return sents, labels


# aste任务：（方面词、意见词、情感极性）(本文三元组[方面词，意见词，情感极性])
# 对于旨在预测（a，o，p）三胞胎的ASTE任务，我们将其在所有情况下映射为一个隐式代词，
# 如“it”（Po (o) = it）。此外，它忽略了隐式方面项，即∈Vx。
# 然后我们总是在原始自然语言形式中使用术语： Pa (a) = a。
# 给出一个三重例子（中国菜，不错，POS），可以相应地构建一个目标句子“这很好，因为中国菜很好”。
def get_para_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri_sentences = []
        for tri in label:
            # a is an aspect term  ## a是方面词
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])

            # b is an opinion term ## b是情绪标签(意见词）
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])

            # c is the sentiment polarity  ## c是情感极性（通过映射转换为自然语言）
            c = senttag2opinion[tri[2]]           # 'POS' -> 'great'

            one_tri = f"It is {c} because {a} is {b}"
            ## one_tri = f"{a} is {c}"  ### {方面词a} is {情绪标签b}
            ## one_tri = f"it is {c}"  ### it is {情绪标签b}
            all_tri_sentences.append(one_tri)
        targets.append(' [SSEP] '.join(all_tri_sentences))
    return targets


# tasd任务：（方面类别，方面词，情绪极性）(本文数据集三元组[方面词，方面类别，情感极性])
# TASD任务预测了（c，a，p）三联体，其中所有的情绪元素都具有与ASQP问题中相同的条件。
# 由于它不涉及意见术语预测，我们只让Po (o) = Pp (p)使用一个人工构建的意见词作为意见表达式来描述释义中的情绪。
# 其他投影函数可以与ASQP任务中保持相同。
# 例如，它将（服务一般、服务员、NEG）三连音转换为目标句“服务一般很坏，因为服务员很坏”。
def get_para_tasd_targets(sents, labels):

    targets = []
    for label in labels:
        all_tri_sentences = []
        for triplet in label:
            at, ac, sp = triplet  # 方面词，方面词，情感极性

            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'

            if at == 'NULL':
                at = 'it'
            one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            ## one_tri = f"{at} is {man_ot}"  ### {方面词at} is {情绪标签man_ot}
            ## one_tri = f"{ac} is {man_ot}"  ### {方面类别ac} is {情绪标签man_ot}

            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return targets


# asqp任务：（方面类别、方面词、意见词、情感极性）四元组 (本文数据集四元组[方面词，方面类别，情感极性，意见词])
# 方面类别c属于类别集Vc
# 方面词a（如果没有明确提到目标，方面词也可以为空）和意见词o通常是句子x中的文本跨度
# 一个∈Vx∪{∅}和o∈Vx，其中Vx表示包含所有可能的连续跨度x的集合。
# 情感极性p属于情绪类别{POS、NEU、NEG}，分别表示积极、中性和消极情绪
def get_para_asqp_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            """
            at, ac, sp, ot = quad  # 方面词，方面词，情感极性，意见词
            man_ot = sentword2opinion[sp]  # 'positive' -> 'great'
            if at == 'NULL':  # for implicit aspect term
                at = 'it'
            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            """
            ac, ot = quad
            man_ot = sentword2opinion[ot]
            one_quad_sentence = f"{ac} is {man_ot}" ### {方面类别ac} is {情绪标签man_ot}
            ## one_quad_sentence = f"{at} is {man_ot}" ### {方面词at} is {情绪标签man_ot}

            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


# 根据任务将输入数据转换为相应的目标数据
def get_transformed_io(data_path, data_dir):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path,)  # 还差了一个silence参数

    # the input is just the raw sentence 输入是句子
    inputs = [s.copy() for s in sents]

    task = 'asqp'  # 默认asqp任务
    if task == 'aste':
        targets = get_para_aste_targets(sents, labels)
    elif task == 'tasd':
        targets = get_para_tasd_targets(sents, labels)
    elif task == 'asqp':
        targets = get_para_asqp_targets(sents, labels)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir)  # 根据task先转换相应数据

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
