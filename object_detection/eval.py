import torch
import torchvision
from torch.utils.data import DataLoader,Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time

def evaluate_model(model, dataloader, iou_threshold=0.5):
    """
    评估 Faster R-CNN 模型在 Cityscapes 数据集上的表现，计算 AP, mAP, Precision 和 F1 Score。

    Args:
        model: 经过训练的 Faster R-CNN 模型
        dataloader: 测试数据集的 DataLoader
        iou_threshold: 用于计算 Precision 的 IoU 阈值
    
    Returns:
        results: 包含 mAP、Precision 和 F1 Score 的字典
    """
    model.eval()
    aps, f1_scores = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for imgs, gts in tqdm(dataloader):
            imgs = list(img.to(device) for img in imgs)
            gt_boxes = [gt['boxes'].to(device) for gt in gts]
            
            preds = model(imgs)  # 推理

            for pred, gt_box in zip(preds, gt_boxes):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                
                # 计算 IoU 并过滤预测框
                iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_box)
                tp = (iou_matrix >= iou_threshold).any(dim=1).sum().item()  # 真正例
                fp = len(pred_boxes) - tp                                   # 假正例

                # 计算 Precision 和 F1 Score
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * precision / (1 + precision) if precision > 0 else 0
                f1_scores.append(f1)
                
                # 计算平均精度（AP）
                ap = precision
                aps.append(ap)

    # 计算 mAP
    mAP = np.mean(aps)
    avg_f1_score = np.mean(f1_scores)

    results = {
        "mAP": mAP,
        "F1 Score": avg_f1_score
    }
    return results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 调用评估函数

from label_mapping import name2label
filterd_label=list(name2label.keys())
# 加载 Faster R-CNN结构
model = fasterrcnn_resnet50_fpn()
#修改分类头
num_classes = len(filterd_label) + 1  # 加1是因为要包括背景类，在pytorch官方定义的fasterrcnn 模型中，0是背景类别
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model = model.to(device)
model.load_state_dict(torch.load("./fasterrcnn_cityscapes_new_200.pth"))



from dataset import CityscapesDataset
# test_dataset = CityscapesDataset("./data/cityscapes",split="val")
test_dataset = CityscapesDataset("~/autodl-tmp/dataset/data/cityscapes",split="val")
# todo 只取前10条数据
test_dataset = Subset(test_dataset, list(range(10)))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4,collate_fn=lambda x: tuple(zip(*x)))


results = evaluate_model(model, test_loader)
print("Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
