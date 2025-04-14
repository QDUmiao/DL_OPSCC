import os  
import re  
import torch  
import torch.nn as nn  
import numpy as np  
import pandas as pd  
import cv2  
from PIL import Image  
import shap  
from torchvision.models import swin_v2_t  
import torchvision.transforms as transforms  
from torch.utils.data import Dataset, DataLoader  
import matplotlib.pyplot as plt  

class DeepSurvHead(nn.Module):  
    def __init__(self, in_features, dropout=0.3):  
        super(DeepSurvHead, self).__init__()  
        self.fc1 = nn.Linear(in_features, 128)  
        self.act1 = nn.LeakyReLU(0.1)  
        self.drop1 = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(128, 64)  
        self.act2 = nn.LeakyReLU(0.1)  
        self.drop2 = nn.Dropout(dropout)  
        self.risk = nn.Linear(64, 1)  
        self._init_weights()  

    def _init_weights(self):  
        for m in self.modules():  
            if isinstance(m, nn.Linear):  
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')  
                if m.bias is not None:  
                    nn.init.zeros_(m.bias)  

    def forward(self, x):  
        x = self.drop1(self.act1(self.fc1(x)))  
        x = self.drop2(self.act2(self.fc2(x)))  
        risk_output = self.risk(x)  
        return risk_output  

class MedicalImageDataset(Dataset):  
    def __init__(self, img_paths, clinical_data=None, transform=None):  
        self.img_paths = img_paths  
        self.clinical_data = clinical_data  
        self.transform = transform  
        
        self.clinical_id_map = {}  
        if clinical_data:  
            for original_id in clinical_data.keys():  
                self.clinical_id_map[original_id] = original_id  
                clean_id = original_id.replace('.0', '') if isinstance(original_id, str) else str(original_id).replace('.0', '')  
                self.clinical_id_map[clean_id] = original_id  
                numeric_id = re.sub(r'\D', '', str(original_id))  
                if numeric_id:  
                    self.clinical_id_map[numeric_id] = original_id  

    def __len__(self):  
        return len(self.img_paths)  

    def __getitem__(self, idx):  
        img_path = self.img_paths[idx]  
        base_name = os.path.basename(img_path)  
        file_name = os.path.splitext(base_name)[0]  
        id1 = file_name  
        match = re.search(r'\d+', base_name)  
        id2 = match.group().strip() if match else ""  
        id3 = re.sub(r'\D', '', file_name)  
        potential_ids = [id1, id2, id3]  
        try:  
            image = Image.open(img_path).convert('RGB')  
            if self.transform:  
                image = self.transform(image)  
        except Exception as e:  
            print(f"Failed to read image {img_path}: {e}")  
            image = torch.zeros(3, 224, 224)  

        matched_id = None  
        clinical_data_entry = None  
        if self.clinical_data:  
            for pid in potential_ids:  
                if pid in self.clinical_data:  
                    matched_id = pid  
                    clinical_data_entry = self.clinical_data[pid]  
                    break  
                if pid in self.clinical_id_map:  
                    original_id = self.clinical_id_map[pid]  
                    matched_id = pid  
                    clinical_data_entry = self.clinical_data[original_id]  
                    break  

        if clinical_data_entry:  
            event = clinical_data_entry['event']  
            time = clinical_data_entry['time']  
            patient_id = matched_id  
            return image, patient_id, event, time, img_path  

        patient_id = id2 if id2 else id1  
        return image, patient_id, -1, -1, img_path  

def load_clinical_data(file_path):  
    try:  
        df = pd.read_csv(file_path)  
        print(f"Clinical data columns: {df.columns.tolist()}")  
        id_col = 'ID'  
        event_col = 'label'  
        time_col = 'TIME'  
        print(f"Using columns: ID={id_col}, Event={event_col}, Time={time_col}")  
        clinical_data = {}  
        for _, row in df.iterrows():  
            patient_id = str(row[id_col])  
            clinical_data[patient_id] = {  
                'event': int(row[event_col]),  
                'time': float(row[time_col])  
            }  
        return clinical_data  
    except Exception as e:  
        print(f"Error loading clinical data: {e}")  
        import traceback  
        traceback.print_exc()  
        return {}  

def nhwc_to_nchw(x):  
    if x.dim() == 4:  
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)  
    elif x.dim() == 3:  
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)  
    return x  

def process_image(image_path):  
    img = cv2.imread(image_path)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (224, 224))  
    img = np.float32(img)  
    return img  

def generate_shap_explanation(model, image, device):  
    mean = [0.485, 0.456, 0.406]  
    std = [0.229, 0.224, 0.225]  

    def predict(img):  
        img = torch.Tensor(img)  
        img = nhwc_to_nchw(img)  
        img = img.to(device)  
        img = img * (1/255)  
        risk_score = model(img)  
        probs = torch.cat([risk_score, -risk_score], dim=1)  
        return probs  

    masker = shap.maskers.Image("blur(64,64)", image.shape)  
    explainer = shap.Explainer(predict, masker, output_names=['Class 0', 'Class 1'])  
    shap_values = explainer(  
        image[None,...],  
        max_evals=3000,  
        batch_size=50,  
        outputs=shap.Explanation.argsort.flip[:2]  
    )  
    return shap_values  

def custom_image_plot(shap_values, pixel_values, labels=None, true_labels=None, width=20, aspect=0.2,   
                      hspace=0.2, labelpad=None, show=True, background_alpha=0):  
    import matplotlib.pyplot as plt  
    import numpy as np  
    import shap  
    from shap.plots import colors  
    
    cmap = colors.red_transparent_blue  

    if str(type(shap_values)).endswith("Explanation'>"):  
        shap_exp = shap_values  
        if len(shap_exp.output_dims) == 1:  
            shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]  
        elif len(shap_exp.output_dims) == 0:  
            shap_values = shap_exp.values  
        else:  
            raise Exception("Number of outputs needs to have support added!")  
        if pixel_values is None:  
            pixel_values = shap_exp.data  
        if labels is None:  
            labels = shap_exp.output_names  

    if not isinstance(shap_values, list):  
        shap_values = [shap_values]  

    if len(shap_values) < 2:  
        print("Warning: Less than 2 SHAP value arrays, copying the first value as the second value")  
        shap_values = [shap_values[0], shap_values[0]]  

    if len(shap_values[0].shape) == 3:  
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]  
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)  

    if labels is not None:  
        if isinstance(labels, list):  
            labels = np.array(labels).reshape(1, -1)  

    label_kwargs = {} if labelpad is None else {'pad': labelpad}  

    x = pixel_values  
    ncols = len(shap_values) + 1  
    fig_size = np.array([3 * ncols, 2.5 * (x.shape[0] + 1)])  
    if fig_size[0] > width:  
        fig_size *= width / fig_size[0]  
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=ncols, figsize=fig_size)  
    if len(axes.shape) == 1:  
        axes = axes.reshape(1, axes.size)  

    for row in range(x.shape[0]):  
        x_curr = x[row].copy()  
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:  
            x_curr = x_curr.reshape(x_curr.shape[:2])  
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:  
            x_curr_gray = 0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2]  
            x_curr_disp = x_curr  
        else:  
            x_curr_gray = x_curr  
            x_curr_disp = x_curr  

        axes[row, 0].imshow(x_curr_disp / 255)  
        if true_labels:  
            axes[row, 0].set_title(true_labels[row], **label_kwargs)  
        axes[row, 0].axis('off')  

        if len(shap_values[0][row].shape) == 2:  
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()  
        else:  
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()  
        max_val = np.nanpercentile(abs_vals, 99.9)  

        for i in range(len(shap_values)):  
            if labels is not None:  
                axes[row, i + 1].set_title(labels[row, i], **label_kwargs)  
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)  

            axes[row, i + 1].imshow(x_curr_gray / 255, cmap=plt.get_cmap('gray'),  
                                     alpha=background_alpha,  
                                     extent=(-1, sv.shape[1], sv.shape[0], -1))  
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)  
            axes[row, i + 1].axis('off')  

    fig.subplots_adjust(hspace=hspace, bottom=0.25)  
    
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value",  
                     orientation="horizontal", aspect=fig_size[0] / aspect,  
                     format='%.6f')  
    
    cb.outline.set_visible(False)  
    
    cb.locator = plt.MaxNLocator(nbins=5)  
    cb.update_ticks()  
    
    cb.ax.tick_params(labelsize=8)  

    if show:  
        plt.show()  

    return fig, axes  

def save_shap_visualization(shap_values, original_image, save_path):  
    import matplotlib.pyplot as plt  
    
    if str(type(shap_values)).endswith("Explanation'>"):  
        if shap_values.values.shape[-1] >= 2:  
            values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]  
        else:  
            values = [shap_values.values[0], shap_values.values[0]]  
            
        shap_values.values = values  

    plt.figure(figsize=(16, 7))  
    
    custom_image_plot(  
        shap_values=shap_values.values,  
        pixel_values=original_image,  
        labels=['Class 0', 'Class 1'],  
        show=False,  
        background_alpha=0  
    )  

    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)  
    plt.close()  

    plt.figure(figsize=(16, 7))  
    custom_image_plot(  
        shap_values=shap_values.values,  
        pixel_values=original_image,  
        labels=['Class 0', 'Class 1'],  
        show=False,  
        background_alpha=0.3  
    )  

    save_path2 = save_path.replace(".jpg", "2.jpg")  
    plt.savefig(save_path2, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)  
    plt.close()  

def save_explanations():  
    try:  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        model_path = r""  
        base_model = swin_v2_t(weights=None)  
        model = base_model  
        model.head = DeepSurvHead(base_model.head.in_features)  
        checkpoint = torch.load(model_path, map_location=device)  
        if 'model_state_dict' in checkpoint:  
            model.load_state_dict(checkpoint['model_state_dict'])  
        else:  
            model.load_state_dict(checkpoint)  
        model = model.to(device)  
        model.eval()  
        print("Model loaded successfully")  

        datasets = ["HPV", "HPV0", "HPV1", "Test", "Train", "Val"]  
        base_dir = r"E:\JTZ\processed_images_new"  
        shap_results_dir = r""  
        os.makedirs(shap_results_dir, exist_ok=True)  

        for dataset_name in datasets:  
            print(f"\n{'-'*50}")  
            print(f"Processing dataset: {dataset_name}")  
            print(f"{'-'*50}")  
            dataset_dir = os.path.join(base_dir, dataset_name)  
            if not os.path.exists(dataset_dir):  
                print(f"Warning: Directory does not exist {dataset_dir}")  
                continue  

            img_paths = []  
            for root, _, files in os.walk(dataset_dir):  
                for file in files:  
                    if file.endswith(('.jpg', '.jpeg', '.png')):  
                        img_paths.append(os.path.join(root, file))  
            if not img_paths:  
                print(f"Warning: No images found in directory {dataset_dir}")  
                continue  
            print(f"Found {len(img_paths)} images")  

            dataset_shap_dir = os.path.join(shap_results_dir, dataset_name)  
            os.makedirs(dataset_shap_dir, exist_ok=True)  
            data_transform = transforms.Compose([  
                transforms.Resize((224, 224)),  
                transforms.ToTensor(),  
            ])  
            clinical_data = load_clinical_data(r"E:\JTZ\OS临床补全_更新.csv")  
            dataset = MedicalImageDataset(img_paths, clinical_data, transform=data_transform)  

            print("\nStarting SHAP visualization generation...")  

            max_images = min(20, len(dataset))  
            for i in range(max_images):  
                image, patient_id, event, time, img_path = dataset[i]  
                try:  
                    print(f"Processing image {i+1}/{max_images}: {os.path.basename(img_path)}")  
                    original_image = process_image(img_path)  
                    with torch.no_grad():  
                        input_tensor = image.unsqueeze(0).to(device)  
                        risk_score = model(input_tensor).item()  
                    shap_values = generate_shap_explanation(model, original_image, device)  
                    base_name = os.path.basename(img_path)  
                    save_path = os.path.join(dataset_shap_dir, f"shap_{base_name.split('.')[0]}.jpg")  
                    save_shap_visualization(shap_values, original_image, save_path)  
                    print(f"  SHAP visualization saved to: {save_path}")  
                except Exception as e:  
                    print(f"  Error processing image: {e}")  
                    import traceback  
                    traceback.print_exc()  
            print(f"Completed SHAP visualizations for dataset {dataset_name}")  
    except Exception as e:  
        print(f"Global error: {e}")  
        import traceback  
        traceback.print_exc()  

if __name__ == "__main__":  
    save_explanations()  
