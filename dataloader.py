import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tokenizer import Tokenizer

class TextDataset(Dataset):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.total_lines = 0
        self.h5_file = h5py.File(filepath, 'r')
        self.dataset = self.h5_file['text_data']

        print(f"Dataset contains {self.dataset.shape[0]} Entries!")

    def __exit__(self, *args):
        self.h5_file.close()

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        data = self.dataset[index].decode('utf-8')
        return data

def collate_fn_wrapper(tokenizer: Tokenizer):
    def collate_fn(text_data: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized, tokenized_pad_masks = tokenizer.encode_batch(text=text_data)
        return tokenized, tokenized_pad_masks
    
    return collate_fn

def get_data_loader(dataset_path: str, batch_size: int, tokenizer: Tokenizer) -> DataLoader:
    dataset = TextDataset(dataset_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_wrapper(tokenizer))
    return data_loader