# -*- coding: utf-8 -*-
from pathlib import Path
import json
import torch
from tqdm import tqdm
from utils.dataload import get_eval_img_dataset
from utils.load_model import load_model
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_feat_output_path = Path(__file__).parent / "databases/img_feat.jsonl"
    data_path = Path(__file__).parent / "data"
    print(data_path)
    batch_size = 512
    model = load_model()
    img_data = get_eval_img_dataset(data_path,batch_size)
    write_cnt = 0
    with open(image_feat_output_path, "w") as fout:
        dataloader = img_data.dataloader
        with torch.no_grad():
            for batch in tqdm(dataloader):
                image_ids, images = batch
                images = images.cuda(device, non_blocking=True)
                image_features = model(images, None)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                    fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                    write_cnt += 1
    print("完成！")
