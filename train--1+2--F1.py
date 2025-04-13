import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
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

# ===================== 训练参数设置 =====================
# 加载配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
MODEL_SAVE_PATH = config['paths']['model_save_path']

# 设备配置
DEVICE = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ===================== 数据加载 =====================
train_dataset = EvidenceDataset(TRAIN_FEATURES)
val_dataset = EvidenceDataset(VAL_FEATURES)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True
)

# ===================== 计算类别权重（如果需要） =====================
# 检查训练集类别分布
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

# ===================== 模型、损失函数和优化器定义 =====================
model = CrossModalAttention(
    feature_dim=FEATURE_DIM,
    attention_heads=ATTENTION_HEADS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)

# 使用类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 使用 AdamW 优化器，增加权重衰减
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4  # 增加权重衰减
)

# 学习率调度器，使用 Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# ===================== 训练和验证函数定义 =====================
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for claim, text_evidences, image_evidences, labels in tqdm(dataloader, desc="训练"):
        claim = claim.to(device)
        text_evidences = text_evidences.to(device)
        image_evidences = image_evidences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(claim, text_evidences, image_evidences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        running_loss += loss.item() * claim.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for claim, text_evidences, image_evidences, labels in tqdm(dataloader, desc="验证"):
            claim = claim.to(device)
            text_evidences = text_evidences.to(device)
            image_evidences = image_evidences.to(device)
            labels = labels.to(device)
            
            outputs = model(claim, text_evidences, image_evidences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * claim.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='weighted'
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1

# ===================== 评估函数定义 =====================
def evaluate_loaded_model(model, dataloader, criterion, device):
    """
    评估已加载模型在验证集上的准确率，用于初始化 best_val_acc。
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for claim, text_evidences, image_evidences, labels in tqdm(dataloader, desc="评估已加载模型"):
            claim = claim.to(device)
            text_evidences = text_evidences.to(device)
            image_evidences = image_evidences.to(device)
            labels = labels.to(device)
            
            outputs = model(claim, text_evidences, image_evidences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * claim.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    print(f"已加载模型在验证集上的表现 -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
    return epoch_acc

# ===================== 初始化 TensorBoard =====================
writer = SummaryWriter(log_dir='runs/cross_modal_attention')

# ===================== 训练过程 =====================
best_val_f1 = 0.0  # 初始化最佳验证 F1 值
early_stopping_patience = EARLY_STOPPING_PATIENCE  # 早停容忍轮数
epochs_no_improve = 0

# 检查是否存在已保存的模型
if os.path.exists(MODEL_SAVE_PATH):
    print(f"检测到已保存的模型文件 '{MODEL_SAVE_PATH}'，正在加载...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    print(f"已加载模型的验证 F1 值为: {best_val_f1:.4f}")
else:
    print(f"未检测到模型文件 '{MODEL_SAVE_PATH}'，将从头开始训练。")

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 10)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"训练 Loss: {train_loss:.4f} | 训练 Accuracy: {train_acc:.4f}")
    
    val_loss, val_acc, val_precision, val_recall, val_f1 = eval_epoch(model, val_loader, criterion, DEVICE)
    print(f"验证 Loss: {val_loss:.4f} | 验证 Accuracy: {val_acc:.4f}")
    print(f"验证 Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1 Score: {val_f1:.4f}")
    
    # 更新学习率调度器
    scheduler.step(val_acc)
    
    # 记录指标到 TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_acc, epoch)
    writer.add_scalar('F1_Score/Validation', val_f1, epoch)
    
    # 保存验证集上 F1 值最优的模型
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1
        }, MODEL_SAVE_PATH)
        epochs_no_improve = 0
        print(f"最优模型已保存到 {MODEL_SAVE_PATH}，最佳 F1 值: {best_val_f1:.4f}")
    else:
        epochs_no_improve += 1
        print(f"验证 F1 值未提升，连续未提升轮数: {epochs_no_improve}")
        if epochs_no_improve >= early_stopping_patience:
            print("早停触发，停止训练。")
            break

# 关闭 TensorBoard 写入器
writer.close()

print(f"\n训练完成。最优验证 F1 值: {best_val_f1:.4f}")
