# -*- coding: utf-8 -*-

import json
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
# 从自定义模块导入特定组件
from cn_clip.clip.model import convert_weights, CLIP
from utils.convert_precision import convert_models_to_fp32,preprocess_text
from utils.dataload import tokenize
from utils.load_model import load_model



import pandas as pd
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_feat_path = Path(__file__).parent / "databases/img_feat.jsonl"

    image_ids = []
    score_tuples = []
    base64_tuples = []
    image_feats = []

    idx = 0
    top_k = 10
    batch_size = 65536

    query = "洗衣柜"

    model = load_model()
    with torch.no_grad():
        query = tokenize(preprocess_text(query))
        texts = query.cuda(device, non_blocking=True)
        text_features = model(None, texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print(type(text_features))
    with open(image_feat_path, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    while idx < len(image_ids):
        img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + batch_size, len(image_ids))]).cuda() # [batch_size, feature_dim]
        batch_scores = text_features @ img_feats_tensor.t() # [1, batch_size]
        for image_id, score in zip(image_ids[idx : min(idx + batch_size, len(image_ids))], batch_scores.squeeze(0).tolist()):
            score_tuples.append((image_id, score))
        idx += batch_size
        top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:top_k]
    outout_ids = [result[0] for result in top_k_predictions]
    print(outout_ids)
    print("-------------------------")
    tsv_file_path = '/home/cv/train/text2images/databases/valid_imgs.tsv'
    df = pd.read_csv(tsv_file_path, sep='\t', header=None, names=['image_id', 'base64'])
    # 遍历每一行，查找对应的Base64编码
    for index, row in df.iterrows():
        image_id = row['image_id']
        base64_data = row['base64']

        # 如果图片ID在需要查找的列表中
        if image_id in outout_ids:
            base64_tuples.append((base64_data))


