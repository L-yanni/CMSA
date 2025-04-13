import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

# ===================== 数据集定义 =====================
class EvidenceDataset(Dataset):
    def __init__(self, features_file):
        with open(features_file, 'rb') as f:
            self.features_data = pickle.load(f)
        self.evidence_ids = list(self.features_data.keys())

    def __len__(self):
        return len(self.evidence_ids)

    def __getitem__(self, idx):
        evidence_id = self.evidence_ids[idx]
        data = self.features_data[evidence_id]
        
        # 获取特征
        claim_features = torch.tensor(data['claim_text_features'], dtype=torch.float32)
        
        # 文本证据特征：列表转为张量 [num_text_evidences, feature_dim]
        if len(data['text_evidence_features']) > 0:
            text_evidence_features = torch.tensor(np.vstack(data['text_evidence_features']), dtype=torch.float32)
        else:
            # 如果没有文本证据，创建一个全零的张量
            text_evidence_features = torch.zeros((1, 512), dtype=torch.float32)
        
        # 图像证据特征：列表转为张量 [num_image_evidences, feature_dim]
        if 'image_features' in data and len(data['image_features']) > 0:
            image_features = torch.tensor(np.vstack(data['image_features']), dtype=torch.float32)
        else:
            # 如果没有图像证据，创建一个全零的张量
            image_features = torch.zeros((1, 512), dtype=torch.float32)
        
        # 标签
        label = torch.tensor(data['label'], dtype=torch.long)
        
        # 修正返回值中的变量名
        return claim_features, text_evidence_features, image_features, label

# ===================== Cross-Attention 层定义 =====================
class CrossAttentionLayer(nn.Module):
    def __init__(self, feature_dim, attention_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=attention_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm_ff = nn.LayerNorm(feature_dim)

    def forward(self, query, key, value):
        # Cross-Attention
        attn_output, attn_weights = self.cross_attention(query, key, value)
        attn_output = self.dropout(attn_output)
        query = self.layer_norm(attn_output + query)  # 残差连接 + LayerNorm

        # Feed-Forward Network
        ff_output = self.feed_forward(query)
        ff_output = self.dropout(ff_output)
        output = self.layer_norm_ff(ff_output + query)  # 残差连接 + LayerNorm

        return output

# ===================== 模型定义 =====================
class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, attention_heads=4, num_classes=3, dropout=0.3):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim

        # 定义一个 Cross-Attention 层用于所有证据（文本 + 图像）
        self.cross_attn = CrossAttentionLayer(feature_dim, attention_heads, dropout)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, claim, text_evidences, image_evidences):
        """
        参数：
            claim: [batch_size, feature_dim]
            text_evidences: [batch_size, num_text_evidences, feature_dim]
            image_evidences: [batch_size, num_image_evidences, feature_dim]
        返回：
            logits: [batch_size, num_classes]
        """
        # 合并文本和图像证据
        combined_evidences = torch.cat((text_evidences, image_evidences), dim=1)  # [batch_size, num_text + num_image, feature_dim]

        # 将 claim 转换为 [batch_size, 1, feature_dim] 以适应 MultiheadAttention
        claim = claim.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Cross-Attention: Query = claim, Key & Value = combined_evidences
        attended = self.cross_attn(claim, combined_evidences, combined_evidences)  # [batch_size, 1, feature_dim]
        attended = attended.squeeze(1)  # [batch_size, feature_dim]

        # 分类
        logits = self.classifier(attended)  # [batch_size, num_classes]

        return logits

# ===================== 加载配置文件 =====================
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# ===================== 训练参数设置 =====================
# 超参数
EPOCHS = config['training']['epochs']
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
ATTENTION_HEADS = config['training']['attention_heads']
NUM_CLASSES = config['training']['num_classes']
FEATURE_DIM = config['training']['feature_dim']
DROPOUT = config['training']['dropout']
EARLY_STOPPING_PATIENCE = config['training']['early_stopping_patience']

# 文件路径
TRAIN_FEATURES = config['paths']['train_features']
VAL_FEATURES = config['paths']['val_features']
TEST_FEATURES = config['paths']['test_features']  # 新增测试集路径
MODEL_SAVE_PATH = config['paths']['model_save_path']

# 设备配置
DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ===================== 数据加载 =====================
val_dataset = EvidenceDataset(VAL_FEATURES)
test_dataset = EvidenceDataset(TEST_FEATURES)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True
)

# ===================== 模型加载 =====================
model = CrossModalAttention(
    feature_dim=FEATURE_DIM,
    attention_heads=ATTENTION_HEADS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)

# 加载模型参数
if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最优模型参数从 {MODEL_SAVE_PATH}")
else:
    print(f"模型文件 {MODEL_SAVE_PATH} 不存在，无法进行评估。")
    exit()

# ===================== 计算类别权重 =====================
# 计算类别权重与训练时一致
with open(TRAIN_FEATURES, 'rb') as f:
    train_data = pickle.load(f)
train_labels = [sample['label'] for sample in train_data.values()]
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)
print(f"类别权重: {class_weights}")

# ===================== 损失函数定义 =====================
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ===================== 评估函数定义 =====================
def evaluate(model, dataloader, criterion, device, class_names, save_confusion_matrix=False, save_path=None):
    """
    评估模型在指定数据集上的性能，并生成分类报告和混淆矩阵。

    参数：
        model (nn.Module): 训练好的模型。
        dataloader (DataLoader): 要评估的数据加载器。
        criterion (nn.Module): 损失函数。
        device (torch.device): 设备（CPU 或 GPU）。
        class_names (list): 类别名称列表。
        save_confusion_matrix (bool, optional): 是否保存混淆矩阵图像。
        save_path (str, optional): 混淆矩阵图像的保存路径。

    返回：
        tuple: 包含损失、准确率、加权 Precision、加权 Recall、计算出的 F1 分数的元组。
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for claim, text_evidence_features, image_evidences, labels in tqdm(dataloader, desc="评估"):
            claim = claim.to(device)
            text_evidence_features = text_evidence_features.to(device)
            image_evidences = image_evidences.to(device)
            labels = labels.to(device)

            outputs = model(claim, text_evidence_features, image_evidences)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * claim.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 生成分类报告（按类别）
    num_classes = len(class_names)
    precision_per_class = []
    recall_per_class = []
    support_per_class = []

    for i in range(num_classes):
        TP = cm[i][i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        support = cm[i,:].sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        support_per_class.append(support)

    # 计算加权平均 Precision 和 Recall
    total_support = sum(support_per_class)
    weighted_precision = sum(p * s for p, s in zip(precision_per_class, support_per_class)) / total_support
    weighted_recall = sum(r * s for r, s in zip(recall_per_class, support_per_class)) / total_support

    # 使用标准公式计算 F1
    if (weighted_precision + weighted_recall) > 0:
        f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
    else:
        f1 = 0.0

    # 生成分类报告字符串
    classification_report_str = "分类报告:\n"
    classification_report_str += "{:<15} {:<10} {:<10} {:<10}\n".format('类别', 'Precision', 'Recall', 'Support')
    for i, class_name in enumerate(class_names):
        classification_report_str += "{:<15} {:<10.4f} {:<10.4f} {:<10}\n".format(
            class_name,
            precision_per_class[i],
            recall_per_class[i],
            support_per_class[i]
        )
    classification_report_str += "\n"
    classification_report_str += "加权 Precision: {:.4f} | 加权 Recall: {:.4f} | F1 Score: {:.4f}\n".format(
        weighted_precision, weighted_recall, f1
    )

    # 打印分类报告
    print(classification_report_str)

    # 保存分类报告到文件
    if save_path:
        with open(save_path, 'a') as f:
            f.write(classification_report_str + "\n")

    # 生成并显示混淆矩阵
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    if save_confusion_matrix and save_path:
        # 修改保存路径以避免覆盖分类报告
        cm_save_path = os.path.splitext(save_path)[0] + '_confusion_matrix.png'
        plt.savefig(cm_save_path)
        print(f"混淆矩阵已保存到 {cm_save_path}")
    plt.show()

    return epoch_loss, epoch_acc, weighted_precision, weighted_recall, f1

# ===================== 主评估过程 =====================
def main():
    # 定义类别名称
    class_names = ['REFUTES', 'SUPPORTS', 'NEI']

    # 创建一个文本文件来保存评估结果
    results_file = 'evaluation_results.txt'
    if os.path.exists(results_file):
        os.remove(results_file)  # 删除旧的结果文件

    with open(results_file, 'w') as f:
        # 评估验证集
        print("\n评估验证集...")
        print("评估验证集...", file=f)
        val_loss, val_acc, val_weighted_precision, val_weighted_recall, val_f1 = evaluate(
            model, val_loader, criterion, DEVICE, class_names,
            save_confusion_matrix=True, save_path='confusion_matrix_val.txt'
        )
        print(f"验证 Loss: {val_loss:.4f} | 验证 Accuracy: {val_acc:.4f}")
        print(f"验证 加权 Precision: {val_weighted_precision:.4f} | Recall: {val_weighted_recall:.4f} | F1 Score: {val_f1:.4f}")
        print(f"验证 Loss: {val_loss:.4f} | 验证 Accuracy: {val_acc:.4f}", file=f)
        print(f"验证 加权 Precision: {val_weighted_precision:.4f} | Recall: {val_weighted_recall:.4f} | F1 Score: {val_f1:.4f}", file=f)
        
        # 评估测试集
        print("\n评估测试集...")
        print("评估测试集...", file=f)
        test_loss, test_acc, test_weighted_precision, test_weighted_recall, test_f1 = evaluate(
            model, test_loader, criterion, DEVICE, class_names,
            save_confusion_matrix=True, save_path='confusion_matrix_test.txt'
        )
        print(f"测试 Loss: {test_loss:.4f} | 测试 Accuracy: {test_acc:.4f}")
        print(f"测试 加权 Precision: {test_weighted_precision:.4f} | Recall: {test_weighted_recall:.4f} | F1 Score: {test_f1:.4f}")
        print(f"测试 Loss: {test_loss:.4f} | 测试 Accuracy: {test_acc:.4f}", file=f)
        print(f"测试 加权 Precision: {test_weighted_precision:.4f} | Recall: {test_weighted_recall:.4f} | F1 Score: {test_f1:.4f}", file=f)

    print(f"\n评估结果已保存到 {results_file}")
    print(f"混淆矩阵已保存到 confusion_matrix_val_confusion_matrix.png 和 confusion_matrix_test_confusion_matrix.png")

if __name__ == "__main__":
    main()
