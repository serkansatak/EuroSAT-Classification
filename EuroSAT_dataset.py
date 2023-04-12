from torchvision import datasets
import glob, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import NUM_CLASSES, DATASET_PATH, LABELS
import cv2
import torch
from tqdm import tqdm


class EuroSAT(Dataset):
    def __init__(self, is_train):
        self.imgs_path = os.path.join(DATASET_PATH, "train") if is_train else os.path.join(DATASET_PATH, "test")
        self.data = []
        self.data_nums = {}
        
        for class_name in LABELS:
            class_path = os.path.join(self.imgs_path, class_name)
            self.data_nums[class_name] = 0
            for img_path in glob.glob(os.path.join(class_path + "/*.jpg")):
                self.data.append([img_path, class_name])
                self.data_nums[class_name] += 1
        
        self.class_map = dict((j,idx) for idx,j in enumerate(self.data_nums.keys()))
        self.img_dims = (64,64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dims)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img).permute(2,0,1).float()
        #label = np.zeros((10,),dtype=float)
        #label[class_id] = 1
        #class_id = torch.from_numpy(label).float()
        return img_tensor, class_id
    
    
if __name__ == "__main__":
    train_dataset = EuroSAT(is_train=True)
    print(f"\nTrain Dataset Info: \n{train_dataset.data_nums}\n")
    
    test_dataset = EuroSAT(is_train=False)
    print(f"\nTest Dataset Info: \n{test_dataset.data_nums}\n")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    for sample in tqdm(train_loader):
        pass
        #print(f"Train Batch of images has shape: {sample[0].shape}") #imgs.shape
        #print(f"Train Batch of labels has shape: {sample[1].shape}") #labels.shape

    for imgs, labels in test_loader:
        pass
        #print(f"Test Batch of images has shape: {imgs.shape}")
        #print(f"Test Batch of labels has shape: {labels.shape}")