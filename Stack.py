import os  
import cv2  
import numpy as np  
from pathlib import Path  
import shutil  
from tqdm import tqdm  
import matplotlib.pyplot as plt  

def process_patient_data(patient_dir, output_class_dir, patient_name, dataset_name):  
    rejpg_dir = os.path.join(patient_dir, "rejpg")  
    remask_dir = os.path.join(patient_dir, "remask")  
    
    if not os.path.exists(rejpg_dir) or not os.path.exists(remask_dir):  
        return 0  
    MASK_AREA_THRESHOLD = 0.30  
    
    try:  
        mask_files = sorted([f for f in os.listdir(remask_dir) if f.endswith('.jpg')],  
                          key=lambda x: int(os.path.splitext(x)[0]))  
        
        if not mask_files:  
            return 0  
        
        selected_layers = []  
        filtered_layers = [] 
        
        for mask_file in mask_files:  
            mask_path = os.path.join(remask_dir, mask_file)  
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
            
            if mask is None:  
                continue  
                
            total_pixels = mask.shape[0] * mask.shape[1]  
            white_pixels = np.count_nonzero(mask)  
            white_ratio = white_pixels / total_pixels  
            
            layer_index = int(os.path.splitext(mask_file)[0])  
            
            if white_ratio > 0 and white_ratio <= MASK_AREA_THRESHOLD:  
                selected_layers.append(layer_index)  
            elif white_ratio > MASK_AREA_THRESHOLD:  
                filtered_layers.append((layer_index, white_ratio))  
        
        ct_files = [f for f in os.listdir(rejpg_dir) if f.endswith('.jpg')]  
        ct_indices = [int(os.path.splitext(f)[0]) for f in ct_files]  
        ct_indices.sort()  
        
        processed_count = 0  
        for layer in selected_layers:  
            if layer == min(ct_indices) or layer == max(ct_indices):  
                continue  
            
            if layer not in ct_indices:  
                continue  
            
            prev_layer = layer - 1  
            next_layer = layer + 1  
            
            if prev_layer not in ct_indices or next_layer not in ct_indices:  
                continue  
            
            current_img = cv2.imread(os.path.join(rejpg_dir, f"{layer}.jpg"), cv2.IMREAD_GRAYSCALE)  
            current_mask = cv2.imread(os.path.join(remask_dir, f"{layer}.jpg"), cv2.IMREAD_GRAYSCALE)  
            prev_img = cv2.imread(os.path.join(rejpg_dir, f"{prev_layer}.jpg"), cv2.IMREAD_GRAYSCALE)  
            next_img = cv2.imread(os.path.join(rejpg_dir, f"{next_layer}.jpg"), cv2.IMREAD_GRAYSCALE)  
            
            if current_img is None or current_mask is None or prev_img is None or next_img is None:  
                continue  
            y_indices, x_indices = np.where(current_mask > 0)  
            x_min, x_max = np.min(x_indices), np.max(x_indices)  
            y_min, y_max = np.min(y_indices), np.max(y_indices)  
            
            current_cropped = current_img[y_min:y_max+1, x_min:x_max+1]  
            prev_cropped = prev_img[y_min:y_max+1, x_min:x_max+1]  
            next_cropped = next_img[y_min:y_max+1, x_min:x_max+1]  
            
            current_resized = cv2.resize(current_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)  
            prev_resized = cv2.resize(prev_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)  
            next_resized = cv2.resize(next_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)  
          
            three_channel = np.zeros((224, 224, 3), dtype=np.uint8)  
            three_channel[:,:,0] = prev_resized  
            three_channel[:,:,1] = current_resized  
            three_channel[:,:,2] = next_resized  
            
            output_filename = f"{patient_name}_{layer}.jpg"  
            output_path = os.path.join(output_class_dir, output_filename)  
            cv2.imwrite(output_path, three_channel)  
            processed_count += 1  
            
        return processed_count  
        
    except Exception as e:  
        print(f"处理 {patient_dir} 时出错: {str(e)}")  
        return 0  

def main():  
    base_dirs = [  

    ]  
    
    output_root =
    os.makedirs(output_root, exist_ok=True)  
    
    total_patients = 0  
    total_processed_images = 0  
  
    for base_dir in base_dirs:  
        dataset_name = os.path.basename(base_dir)  

        dataset_output_dir = os.path.join(output_root, dataset_name)  
        os.makedirs(dataset_output_dir, exist_ok=True)  
        
        output_class_0 = os.path.join(dataset_output_dir, "class_0")  
        output_class_1 = os.path.join(dataset_output_dir, "class_1")  
        os.makedirs(output_class_0, exist_ok=True)  
        os.makedirs(output_class_1, exist_ok=True)  
        
        for class_dir in ["0", "1"]:  
            class_path = os.path.join(base_dir, class_dir)  
            if not os.path.exists(class_path):  
                continue  
            output_class_dir = output_class_0 if class_dir == "0" else output_class_1  

            a_folder = os.path.join(class_path, "A")  
            if not os.path.exists(a_folder):  
                continue  
                
            patient_folders = [f for f in os.listdir(a_folder) if os.path.isdir(os.path.join(a_folder, f))]  
            
            for patient in tqdm(patient_folders, desc=f"处理 {dataset_name} - 类别 {class_dir}"):  
                total_patients += 1  
                patient_dir = os.path.join(a_folder, patient)  
                processed_images = process_patient_data(patient_dir, output_class_dir, patient, dataset_name)  
                total_processed_images += processed_images  

if __name__ == "__main__":  
    main()  
