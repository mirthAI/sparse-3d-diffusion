import time
import random
from tqdm import tqdm
import torch

import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def load_data(
    data_dir,
    batch_size,
    patch_size,
    num_workers,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    full_dose_files = all_files[0::2]
    low_dose_files = all_files[1::2]

    if len(low_dose_files) != len(full_dose_files):
        raise ValueError("The number of low dose and full dose files should be equal！")

    for low_dose, full_dose in zip(low_dose_files, full_dose_files):

        low_dose_number = low_dose.split('/')[-1].split('_')[0]
        full_dose_number = full_dose.split('/')[-1].split('_')[0]

        if low_dose_number != full_dose_number:
            raise ValueError(f"Low dose and full dose files mismatch: {low_dose} and {full_dose}")

    dataset = Patch_ImageDataset(
        low_dose_files,
        full_dose_files,
        patch_size,
    )
    print('There are {} samples in training set'.format(len(dataset)))

    dist_sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=dist_sampler, num_workers=num_workers, drop_last=True, pin_memory=True)

    while True:
        dist_sampler.set_epoch(random.randint(0, 2**31 - 1))
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


# normalize to (-1,1)
class Patch_ImageDataset(Dataset):
    def __init__(
        self,
        low_paths,
        full_paths,
        patch_size,
    ): # creating a dataset of 3d voxel patches by reading and randomly cropping 3d patches from an image for reference
        super().__init__()

        assert len(full_paths) == len(low_paths)
        self.patch_size = patch_size
        self.stride = [s // 2 for s in self.patch_size]
        self.low_images = []
        self.full_images = []
        
        print(f"Loading {len(low_paths)} image pairs ...")
        for i, (low_path, full_path) in enumerate(tqdm(zip(low_paths, full_paths), total=len(low_paths))):
            low_img = np.load(low_path, allow_pickle=False)
            full_img = np.load(full_path, allow_pickle=False)
            assert low_img.shape == full_img.shape, f"Shape mismatch at index {i}: {low_img.shape} vs {full_img.shape}"
            
            self.low_images.append(low_img)
            self.full_images.append(full_img)
        print("All images are loaded into memory successfully!")

        def compute_starts(dim_size, patch, stride):
            starts = list(range(0, max(dim_size - patch + 1, 1), stride))
            if starts[-1] + patch < dim_size:
                starts.append(dim_size - patch)
            return starts

        self.index_list = []
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        for img_idx, img in enumerate(self.low_images):
            C, H, W = img.shape
            d_starts = compute_starts(C, pd, sd)
            h_starts = compute_starts(H, ph, sh)
            w_starts = compute_starts(W, pw, sw)

            for c in d_starts:
                for h in h_starts:
                    for w in w_starts:
                        self.index_list.append((img_idx, c, h, w))
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        img_idx, c, h, w = self.index_list[idx]
        pc, ph, pw = self.patch_size
        low_img = self.low_images[img_idx]
        full_img = self.full_images[img_idx]

        low_patch = low_img[c:min(c+pc, low_img.shape[0]),
                           h:min(h+ph, low_img.shape[1]),
                           w:min(w+pw, low_img.shape[2])]
        full_patch = full_img[c:min(c+pc, full_img.shape[0]),
                             h:min(h+ph, full_img.shape[1]),
                             w:min(w+pw, full_img.shape[2])]

        low_img_patch = np.full((pc, ph, pw), -1, dtype=low_patch.dtype)
        full_img_patch = np.full((pc, ph, pw), -1, dtype=full_patch.dtype)

        low_img_patch[:low_patch.shape[0], :low_patch.shape[1], :low_patch.shape[2]] = low_patch
        full_img_patch[:full_patch.shape[0], :full_patch.shape[1], :full_patch.shape[2]] = full_patch

        assert low_img_patch.shape == full_img_patch.shape
        data = {'low_res': torch.from_numpy(low_img_patch.copy()).float(), 'full_res': torch.from_numpy(full_img_patch.copy()).float(), 'idx': idx}

        return data