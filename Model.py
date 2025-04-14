import os  
import torch  
import torch.nn as nn  
import torchvision.transforms as transforms  
import torchvision.datasets as datasets  
import torch.optim as optim  
import matplotlib.pyplot as plt  
import psutil  
import GPUtil  
from torch.utils.data import DataLoader, Subset, Dataset  
import random  
import numpy as np  
from PIL import ImageOps, Image  
import torch.nn.functional as F  
import pandas as pd  
from lifelines.utils import concordance_index  
import re  
from sklearn.metrics import roc_auc_score, roc_curve, auc  

class PrintShape(nn.Module):  
    def __init__(self, message="Shape"):   
        super(PrintShape, self).__init__()  
        self.message = message  

    def forward(self, x):  
        print(f"{self.message}: {x.shape}")  
        return x  
    
def get_gpu_memory_map():  
    gpu_memory_map = GPUtil.getGPUs()[0].memoryUsed  
    return gpu_memory_map  

class CoxPHLoss(nn.Module):  
    def __init__(self, l2_reg=1e-4):  
        super(CoxPHLoss, self).__init__()  
        self.l2_reg = l2_reg  
    
    def forward(self, risk_scores, survival_time, event):  
        risk_scores = risk_scores.view(-1)  
        
        if risk_scores.shape[0] <= 1 or torch.sum(event) == 0:  
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)  
 
        _, indices = torch.sort(survival_time, descending=True)  
        risk_scores = risk_scores[indices]  
        event = event[indices]  

        max_risk = torch.max(risk_scores)  
        risk_scores_shift = risk_scores - max_risk  
        exp_risk = torch.exp(risk_scores_shift)  
      
        cum_exp_risk = torch.cumsum(exp_risk, dim=0)  
        log_risk = torch.log(cum_exp_risk) + max_risk  
        
        event_indices = (event == 1).nonzero().squeeze()  
        if event_indices.dim() == 0:   
            event_indices = event_indices.unsqueeze(0)  
            
        neg_likelihood = -torch.sum(risk_scores_shift[event_indices] - log_risk[event_indices])  

        reg_term = self.l2_reg * torch.sum(risk_scores**2) / 2  
        
        num_events = len(event_indices)  
        neg_log_likelihood = neg_likelihood / num_events + reg_term  
        
        return neg_log_likelihood  

class RiskScoreHead(nn.Module):  
    def __init__(self, in_features):  
        super(RiskScoreHead, self).__init__()  
        self.fc1 = nn.Linear(in_features, 256)  
        self.bn1 = nn.BatchNorm1d(256)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(256, 128)  
        self.bn2 = nn.BatchNorm1d(128)  
        self.relu2 = nn.ReLU()  
        self.risk = nn.Linear(128, 1)  
    
    def forward(self, x):  
        if x.size(0) == 1:  
            x = self.fc1(x)  
            x = self.relu1(x)  
        else:  
            x = self.fc1(x)  
            x = self.relu1(x)  
            
        if x.size(0) == 1:  
            x = self.fc2(x)  
            x = self.relu2(x)  
        else:  
            x = self.fc2(x)  
            x = self.relu2(x)  
            
        risk_output = self.risk(x)  
        return risk_output  

class SurvivalImageDataset(Dataset):  
    def __init__(self, image_dataset, survival_data, transform=None):  
        self.image_dataset = image_dataset  
        self.survival_data = survival_data  
        self.transform = transform  
        self.valid_indices = []  
        self.valid_samples = []  
        self.patient_image_map = {}   
        
        for i, (path, _) in enumerate(image_dataset.samples):   
            match = re.search(r'\d+', os.path.basename(path))  
            if match:  
                img_id = match.group().strip()  
                img_id = str(img_id)   
                
                if img_id in survival_data:  
                    self.valid_indices.append(i)  
                    self.valid_samples.append(path)  
                     
                    if img_id not in self.patient_image_map:  
                        self.patient_image_map[img_id] = []  
                    self.patient_image_map[img_id].append(len(self.valid_indices) - 1)  

        all_times = []  
        for img_id in survival_data:  
            all_times.append(survival_data[img_id]['time'])  
          
        self.time_mean = np.mean(all_times)  
        self.time_std = np.std(all_times)  
        if self.time_std == 0:   
            self.time_std = 1.0  
        
        print(f"生存时间标准化: 均值={self.time_mean:.2f}, 标准差={self.time_std:.2f}") 
        
    def __len__(self):  
        return len(self.valid_indices)  
    
    def __getitem__(self, idx):  
        img_idx = self.valid_indices[idx]  
        img_path, _ = self.image_dataset.samples[img_idx]  
        image = self.image_dataset.loader(img_path)  
        
        if self.transform:  
            image = self.transform(image)  
        
        match = re.search(r'\d+', os.path.basename(img_path))  
        if match:  
            img_id = match.group().strip()  
            img_id = str(img_id)  
            time = torch.tensor(self.survival_data[img_id]['time'], dtype=torch.float32)  
            event = torch.tensor(self.survival_data[img_id]['event'], dtype=torch.float32)  
        else:  
            raise ValueError(f"No numeric ID found in filename: {img_path}")  
        
        time_value = self.survival_data[img_id]['time']
        normalized_time = (time_value - self.time_mean) / self.time_std  
        time = torch.tensor(normalized_time, dtype=torch.float32)  
        event = torch.tensor(self.survival_data[img_id]['event'], dtype=torch.float32)  
       
        return image, time, event, img_id 
      
def calculate_c_index(risk_scores, survival_times, events):  
    if len(risk_scores) <= 1 or events.sum() == 0:  
        return 0.0  
    
    risk_scores = risk_scores.detach().cpu().numpy()  
    survival_times = survival_times.detach().cpu().numpy()  
    events = events.detach().cpu().numpy()  
    
    c_index = concordance_index(survival_times, -risk_scores, events)  
    return c_index  

def validate(model, loader, cox_criterion, device):  
    model.eval()  
    total_cox_loss = 0.0  
    
    all_risk_scores = []  
    all_survival_times = []  
    all_events = []  
    
    with torch.no_grad():  
        for data in loader:  
            images, times, events, _ = data  
            images = images.to(device)  
            times, events = times.to(device), events.to(device)  

            risk_outputs = model(images)  
            risk_scores = risk_outputs.squeeze()  

            if risk_scores.dim() == 0:  
                risk_scores = risk_scores.unsqueeze(0)  

            if len(risk_scores) > 1 and events.sum() > 0:  
                cox_loss = cox_criterion(risk_scores, times, events)  
                total_cox_loss += cox_loss.item()  
            
            all_risk_scores.append(risk_scores.cpu())  
            all_survival_times.append(times.cpu())  
            all_events.append(events.cpu())  

    num_batches = len(loader)  
    avg_cox_loss = total_cox_loss / num_batches if num_batches > 0 else 0.0  
  
    c_index = 0.0  
    if all_risk_scores:  
        all_risk_scores = torch.cat(all_risk_scores)  
        all_survival_times = torch.cat(all_survival_times)  
        all_events = torch.cat(all_events)  
        
        if len(all_risk_scores) > 1 and all_events.sum() > 0:  
            c_index = calculate_c_index(all_risk_scores, all_survival_times, all_events)  
    
    return avg_cox_loss, c_index  

seed = 42  
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
np.random.seed(seed)  
random.seed(seed)  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  
train_data_path =
val_data_paths = {  
}  
clinical_data_path = r""  

batch_size = 64  
learning_rate = 0.003  
num_epochs = 200  
iter_validation = 100  
evaluate_train_size = 0.01  
evaluate_val_size = 1.0 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

clinical_df = pd.read_csv(clinical_data_path)  

clinical_df['ID'] = clinical_df['ID'].astype(str).str.strip()  

survival_data = {}  
for _, row in clinical_df.iterrows():  
    survival_data[row['ID']] = {  
        'time': row['TIME'],  
        'event': row['label']  
    }  
  
from torchvision.models import swin_v2_t, Swin_V2_T_Weights  
model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)  
model.head = RiskScoreHead(model.head.in_features)   
model = model.to(device)  

params = []  
  
feature_params = [p for name, p in model.named_parameters() if 'head' not in name]  
params.append({  
    'params': feature_params,  
    'lr': learning_rate * 1  
})  

params.append({  
    'params': model.head.parameters(),  
    'lr': learning_rate  
})  

optimizer = optim.AdamW(params, weight_decay=0.00001)  

train_transform = transforms.Compose([  
    transforms.ToTensor(),  
])  

val_transform = transforms.Compose([  
    transforms.ToTensor(),  
])  

train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)  
train_dataset2 = datasets.ImageFolder(train_data_path, transform=val_transform)  
  
train_survival_dataset = SurvivalImageDataset(train_dataset, survival_data, train_transform)   
  
train_size = int(evaluate_train_size * len(train_survival_dataset))  
train_indices = random.sample(range(len(train_survival_dataset)), train_size) if train_size < len(train_survival_dataset) else list(range(len(train_survival_dataset)))  
train_survival_subset = Subset(train_survival_dataset, train_indices)  

dataloader_train = DataLoader(train_survival_dataset, batch_size=batch_size, shuffle=True)  
dataloader_train_subset = DataLoader(train_survival_subset, batch_size=batch_size, shuffle=False)  

val_survival_datasets = {}  
val_survival_loaders = {}  
val_survival_subset_loaders = {}  

for name, path in val_data_paths.items():  
    val_dataset = datasets.ImageFolder(path, transform=val_transform)  
    val_survival_dataset = SurvivalImageDataset(val_dataset, survival_data, val_transform)  
    val_survival_datasets[name] = val_survival_dataset  
    val_survival_loaders[name] = DataLoader(val_survival_dataset, batch_size=batch_size, shuffle=False)  
    
    val_size = int(evaluate_val_size * len(val_survival_dataset))  
    if val_size > 0:  
        if val_size < len(val_survival_dataset):  
            val_indices = random.sample(range(len(val_survival_dataset)), val_size)  
            val_subset = Subset(val_survival_dataset, val_indices)  
        else:  
            val_subset = val_survival_dataset  
            
        val_survival_subset_loaders[name] = DataLoader(val_subset, batch_size=batch_size, shuffle=False)  
        print(f" {len(val_dataset)} {len(val_survival_dataset)} {val_size}")  


cox_criterion = CoxPHLoss()  

 
iter_numbers = []  
train_cox_losses = []  
train_c_indices = []  

val_cox_losses = {name: [] for name in val_data_paths}  
val_c_indices = {name: [] for name in val_data_paths}  

global_iteration = 0  
for epoch in range(num_epochs):  
    model.train()  
    running_cox_loss = 0.0  
    rem_loss = 0.0
    
    batch_risk_scores = []  
    batch_survival_times = []  
    batch_events = []  
    
    for i, (images, times, events, _) in enumerate(dataloader_train):  
        images = images.to(device)  
        times, events = times.to(device), events.to(device)  
        
        optimizer.zero_grad()  
        risk_outputs = model(images)  
        risk_scores = risk_outputs.squeeze()  

        if risk_scores.dim() == 0:  
            risk_scores = risk_scores.unsqueeze(0)  
        
        if len(risk_scores) > 1 and events.sum() > 0:  
            cox_loss = cox_criterion(risk_scores, times, events)  
            cox_loss.backward()  
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()  
            running_cox_loss += cox_loss.item()  
            rem_loss += cox_loss.item()  
        
        batch_risk_scores.append(risk_scores.detach())  
        batch_survival_times.append(times.detach())  
        batch_events.append(events.detach())  
        
        global_iteration += 1  
        
        gpu_memory = get_gpu_memory_map()  
        print(f"Epoch [{epoch + 1}/{num_epochs}], Iter [{global_iteration}], "  
              f"Cox Loss: {cox_loss.item() if 'cox_loss' in locals() else 0.0:.4f}, "  
              f"GPU Memory: {gpu_memory} MB")  
        if global_iteration % iter_validation == 0:    
            if batch_risk_scores:  
                all_risk_scores = torch.cat(batch_risk_scores)  
                all_survival_times = torch.cat(batch_survival_times)  
                all_events = torch.cat(batch_events)  
                train_c_index = calculate_c_index(all_risk_scores, all_survival_times, all_events)  
            else:  
                train_c_index = 0.0  
            
            train_cox_loss, train_subset_c_index = validate(  
                model, dataloader_train_subset, cox_criterion, device  
            )  
            
            iter_numbers.append(global_iteration)  
            train_cox_losses.append(rem_loss / iter_validation)
            
            rem_loss = 0
            
            train_c_indices.append(train_c_index)  
            
            for name, dataset in val_survival_datasets.items():  
                val_cox_loss, val_c_index = validate(  
                    model, dataset, cox_criterion, device  
                )  
            
                val_cox_losses[name].append(val_cox_loss)  
                val_c_indices[name].append(val_c_index)  
            
            plt.figure(figsize=(10, 10))  
            
            plt.subplot(2, 1, 1)  
            plt.plot(iter_numbers, train_cox_losses, 'b-', label='Train Cox Loss')  
            plt.xlabel('Iterations')  
            plt.ylabel('Loss')  
            plt.title('Cox Proportional Hazard Loss')  
            plt.legend()  
            plt.grid(True)  
            
            plt.subplot(2, 1, 2)  
            plt.plot(iter_numbers, train_c_indices, 'b-', label='Train C-index')  
            for name in val_c_indices:  
                if val_c_indices[name]:
                    plt.plot(iter_numbers, val_c_indices[name], label=f'{name} C-index')  
            plt.xlabel('Iterations')  
            plt.ylabel('C-index')  
            plt.title('Concordance Index')  
            plt.legend()  
            plt.grid(True)  
            
            plt.tight_layout()  
            plt.show()  
            plt.close()  
            
            batch_risk_scores = []  
            batch_survival_times = []  
            batch_events = []  
            
            training_history = {  
                'iterations': iter_numbers,  
                'train': {  
                    'cox_loss': train_cox_losses,  
                    'c_index': train_c_indices  
                }  
            }  
            
            for name in val_data_paths:  
                if name in val_cox_losses:  
                    training_history[name] = {  
                        'cox_loss': val_cox_losses[name],  
                        'c_index': val_c_indices[name]  
                    }  
            

final_metrics = {}  

train_cox_loss, train_c_index = validate(  
    model, dataloader_train_subset, cox_criterion, device  
)  
final_metrics['Train'] = {  
    'cox_loss': train_cox_loss,  
    'c_index': train_c_index  
}  

for name, dataset in val_survival_datasets.items():  
    val_cox_loss, val_c_index = validate(  
        model, dataset, cox_criterion, device  
    )  
    final_metrics[name] = {  
        'cox_loss': val_cox_loss,  
        'c_index': val_c_index  
    }  

for dataset_name, metrics in final_metrics.items():  
    print(f"\n{dataset_name}:")  
    print(f"  Cox损失: {metrics['cox_loss']:.4f}")  
    print(f"  C-index: {metrics['c_index']:.4f}")  

torch.save({  
    'model_state_dict': model.state_dict(),  
    'optimizer_state_dict': optimizer.state_dict(),  
    'final_metrics': final_metrics,  
}, 'final_cox_model.pth')  
