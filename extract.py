# 从xml文件提取文本、方面词、方面类别、情感极性得到csv文件
import xml.etree.cElementTree as ET

import pandas as pd

#path = 'data/Restaurants_Train_v2.xml'
path = 'data/Restaurants_Test_Gold.xml'
tree = ET.parse(path)
root = tree.getroot()

# 获取aspectCategory
data = []

for sentence in root.findall('sentence'):
    text = sentence.find('text').text
    aspectCategories = sentence.find('aspectCategories')
    tmp1 = []
    for aspectCategory in aspectCategories.findall('aspectCategory'):
        tmp2 = []
        category = aspectCategory.get('category')
        polarity = aspectCategory.get('polarity')
        tmp2.append([category, polarity])
        tmp1.append(tmp2)
    data.append((text, tmp1))

df=pd.DataFrame(data,columns=['text', 'emotion'])
df.to_csv('data/text_cate_pola_1.csv')
df.head()

"""# 获取aspectTerms
data = []
for sentence in root.findall('.//aspectTerms/..'):
    text = sentence.find('text').text
    aspectTerms = sentence.find('aspectTerms')
    for aspectTerm in aspectTerms.findall('aspectTerm'):
        term = aspectTerm.get('term')
        polarity = aspectTerm.get('polarity')
        data.append((text, term, polarity))

df = pd.DataFrame(data, columns=['text', 'term', 'polarity'])
df = df[df['polarity'].isin(['love', 'joy', 'approval', 'neutral', 'disappointed', 'disgust', 'angry'])]
#df['polarity'] = df['polarity'].map(
 #   {'love': 7, 'joy': 6, 'approval': 5, 'neutral': 4, 'disappointed': 3, 'disgust': 2, 'angry': 1})

df.to_csv('data/text_term_pola.csv', sep=',', index=0)
df"""
