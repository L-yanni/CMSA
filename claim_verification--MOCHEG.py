import os
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def load_data(features_file):
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    return data

def save_features(features_data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(features_data, f)
    print(f"特征已保存到 {save_path}")

def map_label(label_str):
    label_mapping = {'refuted': 0, 'supported': 1, 'NEI': 2}
    return label_mapping.get(label_str, -1)  # 返回-1表示未知标签

def ensure_string_format(text):
    """确保输入为字符串或字符串列表，若不是则转换为字符串"""
    if isinstance(text, str):
        return text
    elif isinstance(text, list):
        return " ".join(map(str, text))  # 将列表拼接为单一字符串
    else:
        return str(text)  # 转换其他类型为字符串

def process_data(data, image_folder, clip_processor, clip_model, device):
    features_data = {}
    
    for evidence_id, evidence in tqdm(data.items(), desc="处理数据"):
        claim = ensure_string_format(evidence['claim'])
        label_str = evidence['label']
        label = map_label(label_str)
        
        if label == -1:
            print(f"未知标签: {label_str}，样本ID: {evidence_id}，跳过此样本。")
            continue  # 跳过未知标签的样本
        
        # 提取声明特征，启用截断
        claim_inputs = clip_processor(
            text=[claim],
            return_tensors="pt",
            truncation=True,
            max_length=77  # 设置最大长度为77
        ).to(device)
        with torch.no_grad():
            claim_features = clip_model.get_text_features(**claim_inputs).detach().cpu().numpy()
        
        # 提取文本证据特征，启用截断
        text_evidence_features = []
        for text in evidence.get('text_evidences', []):
            text = ensure_string_format(text)
            text_inputs = clip_processor(
                text=[text],
                return_tensors="pt",
                truncation=True,
                max_length=77  # 设置最大长度为77
            ).to(device)
            with torch.no_grad():
                text_feat = clip_model.get_text_features(**text_inputs).detach().cpu().numpy()
            text_evidence_features.append(text_feat.squeeze(0))
        
        # 提取图像证据特征
        image_features = []
        for img_path in evidence.get('image_evidences', []):
            img_full_path = os.path.join(image_folder, img_path)
            if not os.path.exists(img_full_path):
                print(f"图像文件不存在: {img_full_path}，跳过此图像。")
                continue
            image = Image.open(img_full_path).convert("RGB")
            image_inputs = clip_processor(
                images=image,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                img_feat = clip_model.get_image_features(**image_inputs).detach().cpu().numpy()
            image_features.append(img_feat.squeeze(0))
        
        # 检查特征是否完整
        if len(text_evidence_features) == 0 and len(image_features) == 0:
            print(f"样本ID: {evidence_id} 没有有效的文本或图像证据，跳过此样本。")
            continue
        
        # 存储特征和标签
        all_features = {
            'claim': claim,
            'claim_text_features': claim_features.squeeze(0),
            'text_evidence_features': text_evidence_features,
            'image_features': image_features,
            'label': label
        }
        features_data[evidence_id] = all_features
    
    return features_data

def main():
    # 文件路径
    train_file = 'dataset/train/10--evidence_cache_train.pkl'
    val_file = 'dataset/val/10--evidence_cache_val.pkl'
    test_file = 'dataset/test/10--evidence_cache_test.pkl'
    
    # 保存路径
    train_save_path = 'dataset/train/10-CLIP-alignment_features.pkl'
    val_save_path = 'dataset/val/10-CLIP-alignment_features.pkl'
    test_save_path = 'dataset/test/10-CLIP-alignment_features.pkl'
    
    # 图像文件夹路径
    image_folder_train = 'dataset/train/images'
    image_folder_val = 'dataset/val/images'
    image_folder_test = 'dataset/test/images'
    
    # 加载CLIP模型和处理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 处理并保存训练集
    train_data = load_data(train_file)
    train_features = process_data(train_data, image_folder_train, clip_processor, clip_model, device)
    save_features(train_features, train_save_path)
    
    # 处理并保存验证集
    val_data = load_data(val_file)
    val_features = process_data(val_data, image_folder_val, clip_processor, clip_model, device)
    save_features(val_features, val_save_path)
    
    # 处理并保存测试集
    test_data = load_data(test_file)
    test_features = process_data(test_data, image_folder_test, clip_processor, clip_model, device)
    save_features(test_features, test_save_path)

if __name__ == "__main__":
    main()
