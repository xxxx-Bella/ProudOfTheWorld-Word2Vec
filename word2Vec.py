import jieba
import re
import jieba.posseg as poss
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 首先进行jieba分词，去除停用词；
# 然后通过正则表达式去除无关字符，构建词向量；
# 最后提取小说的所有人名并画图展示出来。

def load_txt(file_path):
    #读取数据
    file = open(file_path, encoding = 'utf-8')
    text = file.readlines()
    file.close()
    #将换行符等特殊字符替换掉
    text = text[1:]   # 第一行是小说的作者信息
    text = [re.sub('\u3000| |\n','',i) for i in text]
    return text


def TextProcessing(stopwords_path, text):
    #分词，去除停用词
    with open(stopwords_path,'r',encoding = 'utf-8') as f:
        stop_words = f.readlines()
    text_cut = [jieba.lcut(i) for i in text]
    stop_words = [re.sub(' |\n','',i) for i in stop_words]
    text_ = [[i for i in word if i not in stop_words] for word in text_cut]
    return text_


def train_model(text_):
    #训练模型，构建词向量
    model = Word2Vec(text_, size = 200, min_count = 5, window = 2, iter = 100)
    # size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间
    # min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5
    # window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）
    # iter是语料库上的迭代次数（epoch）
    print('Word2Vec Training Endding!')
    return model


def analyse_wordVector(model, name_list, sentence):
    for name in name_list:
        print('{}的词向量为:\n{}'.format(name, model[name]))
        print('与{}最相关的词:{}'.format(name, model.most_similar(name)))
        topn = 3 # 查看跟'令狐冲'相关性前三的词
        print('跟{}相关性前{}的词:\n{}'.format(name, topn, model.similar_by_word(name, topn=topn))) 
        print('跟{}关系相当于师妹跟林平之的关系的词:\n{}'.format(name, model.most_similar(['师妹','林平之'], [name], topn=topn)))
        print('跟{}关系相当于师妹跟圣姑的关系的词:\n{}'.format(name, model.most_similar(['师妹','圣姑'], [name], topn=topn)))
        #u"令狐冲 任盈盈 林平之 岳不群 东方不败"
    a,b = '令狐冲','师妹'
    print('集合{}中不同类的词语:{}'.format(name_list, model.wv.doesnt_match(u"令狐冲 任盈盈 林平之 岳不群 东方不败".split()))) # 选出集合中不同类的词语
    print('{}和{}之间的相关度:{}'.format(a, b, model.wv.similarity(a,b))) # 两个词语之间的相关度

    #分词后对词的属性进行分析
    sentence = poss.lcut(sentence)
    # cut()分词，返回一个生成器generator，可通过迭代的方法访问各个分词
    # lcut()返回的是list，list(jieba.cut())等价与jieba.lcut()
    print(sentence) # nr:人名 r:代词 v:动词
    print('测试句子中的人名有：', [list(i)[0] for i in sentence if list(i)[1] == 'nr']) # ['林平之']


def extractName(text, model):
    #提取这本小说里的所有人名（nr）
    temp = [poss.lcut(i) for i in text]
    people = [[i.word for i in x if i.flag =='nr'] for x in temp]
    temp2 = [' '.join(x) for x in people]
    people2 = list(set(' '.join(temp2).split()))
    data = []
    newpeo = []
    for i in people2:
        try:
            data.append(model.wv[i]) # wv = Word2VecKeyedVectors(size)
            newpeo.append(i)
        except KeyError:
            pass
    return data, newpeo


def Visualization(data, newpeo):
    # PCA降维后画图
    data = PCA(n_components=2).fit_transform(data)
    print('data_shape', data.shape) #(566, 2)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15,15), dpi=480)
    for i in range(len(data)):
        plt.scatter(data[i,0], data[i,1])
        plt.text(data[i,0], data[i,1], newpeo[i])
    plt.savefig('figure1.png')   # 保存到本地
    plt.show()


if __name__ == "__main__":
    file_path = r'笑傲江湖.txt'
    stopwords_path = r'中文停用词.txt'
    name = ['令狐冲', '师妹', '林平之', '岳不群', '东方不败']
    sentence = '林平之是谁'

    text = load_txt(file_path)
    text_ = TextProcessing(stopwords_path, text)
    model = train_model(text_)
    analyse_wordVector(model, name, sentence)
    #data, newpeo = extractName(text, model)
    #Visualization(data, newpeo)



