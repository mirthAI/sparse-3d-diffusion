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

    return dataset


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
        self.stride = [s // 1 for s in self.patch_size]
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
        
    def __len__(self):
        return len(self.full_images)

    def __getitem__(self, idx):
        pc, ph, pw = self.patch_size
        sc, sh, sw = self.stride
        low_patch_list = []
        full_patch_list = []

        low_img = self.low_images[idx]
        full_img = self.full_images[idx]

        pad_c = max(0, pc - low_img.shape[0])
        pad_h = max(0, ph - low_img.shape[1])
        pad_w = max(0, pw - low_img.shape[2])

        if pad_c or pad_h or pad_w:
            low_img = np.pad(low_img, ((0, pad_c), (0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
            full_img = np.pad(full_img, ((0, pad_c), (0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)

        def compute_starts(dim_size, patch, stride):
            starts = list(range(0, dim_size - patch + 1, stride))
            if starts[-1] + patch < dim_size:
                starts.append(dim_size - patch)
            return starts
        
        C, H, W = low_img.shape
        d_starts = compute_starts(C, pc, sc)
        h_starts = compute_starts(H, ph, sh)
        w_starts = compute_starts(W, pw, sw)

        for c in d_starts:
            for h in h_starts:
                for w in w_starts:
                    low_patch_list.append(low_img[c:c+pc, h:h+ph, w:w+pw])
                    full_patch_list.append(full_img[c:c+pc, h:h+ph, w:w+pw])

        low_patches = np.stack(low_patch_list, axis=0)
        full_patches = np.stack(full_patch_list, axis=0)
        assert low_patches.shape == full_patches.shape
        data = {'low_res': torch.from_numpy(low_patches.copy()).float(), 'full_res': torch.from_numpy(full_patches.copy()).float(), 'shape': low_img.shape}

        return data