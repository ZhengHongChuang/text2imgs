import json
from pathlib import Path
import torch
# 从自定义模块导入特定组件
from cn_clip.clip.model import convert_weights, CLIP
from utils.convert_precision import convert_models_to_fp32
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resume = Path(__file__).parent.parent / "weights/epoch7.pt"
    vision_model_config_file = Path(__file__).parent.parent / "configs/vision_model.json"
    text_model_config_file = Path(__file__).parent.parent / "configs/text_model.json"
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v
    model = CLIP(**model_info)
    convert_weights(model)    
    convert_models_to_fp32(model)
    model.to(device)
    checkpoint = torch.load(resume, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    model.eval
    return model