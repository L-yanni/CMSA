import os
import pickle
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import pipeline

# 使用Hugging Face transformers加载预训练的NER模型
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=0)  # device=0 使用GPU

# 加载预训练的Word2Vec模型
word2vec_model_path = 'GoogleNews-vectors-negative300.bin'  
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

# 提前加载停用词
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text, stop_words):
    """
    预处理文本，去除停用词和非字母字符
    """
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

def text_to_vector(text, model):
    """
    将文本转换为向量表示，基于词向量的平均值
    """
    tokens = preprocess(text, stop_words=stop_words)
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def vectorize_and_save_texts(corpus_df, save_path):
    """
    将文档向量化并保存到文件
    """
    print("正在向量化文档...") 
    vectors = []
    for text in tqdm(corpus_df['Origin'], desc="文档向量化"):  # 使用 'Origin' 列
        vector = text_to_vector(text, word2vec_model)  # 将文本转为向量
        vectors.append(vector)

    with open(save_path, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"文档向量化完成，已保存到 {save_path}")

def vectorize_and_save_claims(claims_df, save_path):
    """
    将声明向量化并保存到文件
    """
    print("正在向量化声明...")
    claim_vectors = {}
    for idx, row in tqdm(claims_df.iterrows(), desc="声明向量化"):
        claim_text = row['Claim']
        claim_vectors[row['claim_id']] = text_to_vector(claim_text, word2vec_model)

    with open(save_path, 'wb') as f:
        pickle.dump(claim_vectors, f)
    print(f"声明向量化完成，已保存到 {save_path}")

def vectorize_and_save_images(image_corpus_folder, save_path):
    """
    将图像描述向量化并保存到文件
    """
    print("正在向量化图像描述...")
    image_files = os.listdir(image_corpus_folder)
    image_descriptions = [os.path.splitext(f)[0].split('-')[-1] for f in image_files]
    
    vectors = []
    for desc in tqdm(image_descriptions, desc="图像描述向量化"):
        vector = text_to_vector(desc, word2vec_model)  # 将图像描述转为向量
        vectors.append(vector)
    
    # 保存图像向量
    with open(save_path, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"图像描述向量化完成，已保存到 {save_path}")

def preprocess_data(dataset_folder, splits=['train', 'val', 'test']):
    """
    对指定的数据集进行向量化并保存
    """
    # 对每个数据集进行处理
    for split in splits:
        claims_path = os.path.join(dataset_folder, split, 'Corpus2.csv')
        images_folder = os.path.join(dataset_folder, split, 'images')

        try:
            claims_df = pd.read_csv(claims_path)
            claims_df.columns = claims_df.columns.str.strip()  # 去除列名空格
        except Exception as e:
            print(f"加载 {claims_path} 时出错：{e}")
            continue

        print(f"正在处理 {split} 数据集...")

        # 向量化并保存声明、文档和图像描述
        corpus_vectors_path = os.path.join(dataset_folder, split, 'corpus_vectors.pkl')
        if not os.path.exists(corpus_vectors_path):  # 如果未处理过，则处理并保存
            vectorize_and_save_texts(claims_df, corpus_vectors_path)  # 使用 claims_df 的 'Origin' 列

        claim_vectors_path = os.path.join(dataset_folder, split, 'claim_vectors.pkl')
        if not os.path.exists(claim_vectors_path):  # 如果未处理过，则处理并保存
            vectorize_and_save_claims(claims_df, claim_vectors_path)

        image_vectors_path = os.path.join(dataset_folder, split, 'image_vectors.pkl')
        if not os.path.exists(image_vectors_path):  # 如果未处理过，则处理并保存
            vectorize_and_save_images(images_folder, image_vectors_path)

def main():
    dataset_folder = 'dataset'  
    
    # 调用预处理数据的函数
    preprocess_data(dataset_folder)

if __name__ == "__main__":
    main()
