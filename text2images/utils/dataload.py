import os
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import lmdb
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from cn_clip.clip import _tokenizer


def _convert_to_rgb(image):
    return image.convert('RGB')

class EvalImgDataset(Dataset):
    def __init__(self, lmdb_imgs, resolution=224):

        assert os.path.isdir(lmdb_imgs), "The image LMDB directory {} not exists!".format(lmdb_imgs)


        self.env_imgs = lmdb.open(str(lmdb_imgs), readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)
        self.cursor_imgs = self.txn_imgs.cursor()
        self.iter_imgs = iter(self.cursor_imgs)
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        img_id, image_b64 = next(self.iter_imgs)
        if img_id == b"num_images":
            img_id, image_b64 = next(self.iter_imgs)

        img_id = img_id.tobytes()
        image_b64 = image_b64.tobytes()

        img_id = int(img_id.decode(encoding="utf8", errors="ignore"))
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
        image = self.transform(image)

        return img_id, image


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def fetch_resolution():
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent/ "configs/vision_model.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


def get_eval_img_dataset(image_data,batch_size):
    lmdb_imgs = image_data
    dataset = EvalImgDataset(
        lmdb_imgs, resolution=fetch_resolution())
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def tokenize(query, context_length = 52):
    all_tokens = []
    all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(query))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


