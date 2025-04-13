import os 
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import argparse  # 导入 argparse 模块

# 加载预训练的Word2Vec模型
word2vec_model_path = 'GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

def load_vectors(file_path):
    """
    从文件加载向量
    """
    with open(file_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors

def extract_text_evidence(claim, claim_vector, corpus_vectors, corpus_df, top_n=2):
    """
    提取与声明相似的文本证据
    """
    # 计算文本之间的余弦相似度
    cosine_similarities = cosine_similarity([claim_vector], corpus_vectors)[0]  # 批量计算相似度
    
    # 获取相似度最高的前top_n条文本证据
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    text_evidences = corpus_df.iloc[top_indices]
    
    return text_evidences.reset_index(drop=True)

def extract_image_evidence(claims_batch, claims_vectors, image_vectors, image_files, top_n=5):
    """
    提取与声明相似的图像证据
    """
    all_image_evidences = []
    
    for i, claim in enumerate(claims_batch):
        claim_vector = claims_vectors[i]  # 使用预先计算的声明向量
        
        # 计算相似度
        cosine_similarities = cosine_similarity([claim_vector], image_vectors)[0]  # 批量计算相似度

        top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
        image_evidence_files = [image_files[j] for j in top_indices]
        
        all_image_evidences.append(image_evidence_files)
    
    return all_image_evidences

def retrieve_evidence(claims_df, split='train', top_n=5, batch_size=64):
    """
    提取证据并保存缓存
    """
    image_folder = os.path.join('dataset', split, 'images')
    cache_file = os.path.join('dataset', split, f'evidence_cache_{split}.pkl')

    # 加载缓存的证据数据
    if os.path.exists(cache_file):
        print(f"加载缓存的证据数据：{cache_file}")
        with open(cache_file, 'rb') as f:
            evidence_dict = pickle.load(f)
        return evidence_dict

    evidence_dict = {}

    # 加载预先向量化的声明向量、文本和图像向量
    claims_vectors_path = os.path.join('dataset', split, 'claim_vectors.pkl')
    corpus_vectors_path = os.path.join('dataset', split, 'corpus_vectors.pkl')
    image_vectors_path = os.path.join('dataset', split, 'image_vectors.pkl')

    try:
        claims_vectors = load_vectors(claims_vectors_path)
    except Exception as e:
        print(f"加载 {claims_vectors_path} 时出错：{e}")
        return evidence_dict

    try:
        corpus_vectors = load_vectors(corpus_vectors_path)
    except Exception as e:
        print(f"加载 {corpus_vectors_path} 时出错：{e}")
        return evidence_dict

    try:
        image_vectors = load_vectors(image_vectors_path)
    except Exception as e:
        print(f"加载 {image_vectors_path} 时出错：{e}")
        return evidence_dict

    # 获取图像文件列表
    try:
        image_files = os.listdir(image_folder)
    except Exception as e:
        print(f"加载图像文件夹 {image_folder} 时出错：{e}")
        return evidence_dict

    total_batches = len(claims_df) // batch_size + (1 if len(claims_df) % batch_size != 0 else 0)
    for idx in tqdm(range(0, len(claims_df), batch_size), total=total_batches, desc=f"{split}集 - 检索证据"):
        batch_claims = claims_df.iloc[idx:idx + batch_size]['Claim'].tolist()
        batch_claim_ids = claims_df.iloc[idx:idx + batch_size]['claim_id'].tolist()

        # 提取文本证据
        text_evidences_batch = []
        for i, claim in enumerate(batch_claims):
            claim_id = batch_claim_ids[i]
            
            # 通过 claim_id 获取 claim_vector
            if isinstance(claims_vectors, dict):
                claim_vector = claims_vectors.get(claim_id)
                if claim_vector is None:
                    print(f"警告：未找到 claim_id {claim_id} 的文本向量！")
                    text_evidences_batch.append([])
                    continue
            else:
                # 如果 claims_vectors 是数组，可以继续按索引访问
                if i >= len(claims_vectors):
                    print(f"警告：声明向量索引 {i} 超出范围！")
                    text_evidences_batch.append([])
                    continue
                claim_vector = claims_vectors[i]
            
            # 提取文本证据
            text_evidences = extract_text_evidence(claim, claim_vector, corpus_vectors, claims_df, top_n=top_n)
            
            # 确认使用正确的列名 'Origin'
            origin_column = 'Origin'
            if origin_column not in text_evidences.columns:
                print(f"错误：列 '{origin_column}' 不存在于 DataFrame 中。")
                text_evidences_batch.append([])
                continue
            
            text_evidences_batch.append(text_evidences[[origin_column]].values.tolist())  # 提取文本证据（文档内容）

        # 提取图像证据
        image_evidence_files_batch = []
        for i, claim in enumerate(batch_claims):
            claim_id = batch_claim_ids[i]

            # 通过 claim_id 获取图像向量
            if isinstance(claims_vectors, dict):
                claim_vector = claims_vectors.get(claim_id)
                if claim_vector is None:
                    print(f"警告：未找到 claim_id {claim_id} 的图像向量！")
                    image_evidence_files_batch.append([])
                    continue
            else:
                if i >= len(claims_vectors):
                    print(f"警告：声明向量索引 {i} 超出范围！")
                    image_evidence_files_batch.append([])
                    continue
                claim_vector = claims_vectors[i]

            image_evidence_files = extract_image_evidence([claim], [claim_vector], image_vectors, image_files, top_n=top_n)
            image_evidence_files_batch.append(image_evidence_files[0] if image_evidence_files else [])

        # 更新 evidence_dict
        for i, claim in enumerate(batch_claims):
            claim_id = batch_claim_ids[i]
            label = claims_df.iloc[idx + i].get('cleaned_truthfulness', 'UNKNOWN')

            evidence_dict[claim_id] = {
                'claim': claim,
                'text_evidences': text_evidences_batch[i],
                'image_evidences': image_evidence_files_batch[i],
                'label': label
            }

    # 将结果缓存到本地文件
    with open(cache_file, 'wb') as f:
        pickle.dump(evidence_dict, f)
    print(f"证据数据已缓存到：{cache_file}")

    return evidence_dict

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="检索文本和图像证据的脚本。")
    parser.add_argument('--top_n', type=int, default=15, help='要检索的最高相似度的证据数量。默认值为10。')
    parser.add_argument('--batch_size', type=int, default=128, help='每批处理的声明数量。默认值为64。')
    args = parser.parse_args()

    # 数据集文件夹路径
    dataset_folder = 'dataset'                                                                                                                                                        

    # 处理train, val, test三个数据集
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n正在处理 {split} 数据集...")

        # 加载数据
        claims_path = os.path.join(dataset_folder, split, 'Corpus2.csv')
        try:
            claims_df = pd.read_csv(claims_path)
            claims_df.columns = claims_df.columns.str.strip()  # 去除列名空格
            print(f"Columns in {claims_path}: {claims_df.columns.tolist()}")  # 调试信息
        except Exception as e:
            print(f"加载 {claims_path} 时出错：{e}")
            continue

        # 确保 'Origin' 列存在
        if 'Origin' not in claims_df.columns:
            print(f"错误：'Origin' 列不存在于 {claims_path} 中。请检查列名是否正确。")
            continue

        image_folder = os.path.join(dataset_folder, split, 'images')

        # 调用检索函数，获取证据
        evidence_dict = retrieve_evidence(
            claims_df, 
            split=split, 
            top_n=args.top_n,  # 使用命令行参数指定的 top_n
            batch_size=args.batch_size  # 使用命令行参数指定的 batch_size
        )

        # 保存最终的证据字典
        output_path = os.path.join(dataset_folder, split, 'evidence_dict.pkl')
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(evidence_dict, f)
            print(f"证据字典已保存到：{output_path}")
        except Exception as e:
            print(f"保存证据字典到 {output_path} 时出错：{e}")

if __name__ == "__main__":
    main()
