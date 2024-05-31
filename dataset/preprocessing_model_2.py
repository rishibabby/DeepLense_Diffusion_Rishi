import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path)
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)
        if self.transform:
            data = self.transform(data)

        return data

class CustomDataset_Conditional(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.folder_path = folder_path
        self.class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        #print(self.class_folders)
        # Initialize LabelEncoder
        self.label_encoder = LabelEncoder()
        
        # Fit and transform class labels to numerical labels
        self.labels = self.label_encoder.fit_transform(self.class_folders)
        self.transform = transforms
        self.data = []
        
        for class_folder in self.class_folders:
            class_path = os.path.join(folder_path, class_folder)
            file_list = [f for f in os.listdir(class_path) if f.endswith('.npy')]

            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                data_point = np.load(file_path, allow_pickle=True)
                if file_name.startswith('axion'):
                    data_point = data_point[0]
                self.data.append((data_point, class_folder))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point, class_name = self.data[idx]
        label = self.label_encoder.transform([class_name])[0]
        
        #data_point = (data_point - np.mean(data_point, axis=(1,2)))/(np.std(data_point, axis=(1,2)))
        
        # Convert NumPy array to PyTorch tensor
        data_point = torch.from_numpy(data_point).float()
        data_point = data_point.unsqueeze(0)

        if self.transform:
            data_point = self.transform(data_point)

        # # Convert label to one-hot vector
        # one_hot_label = F.one_hot(torch.tensor(label), num_classes=len(self.labels))

        
        return data_point, label

if __name__ == '__main__':
    print('hi')
    dataset =  CustomDataset_Conditional('../../Data/Model_II/')
    print(dataset[0].shape)