import os  
import cv2  
import numpy as np  
from tqdm import tqdm  


from scipy.ndimage import gaussian_filter, map_coordinates  


def apply_spatial_transform(image, angle, scale, tx, ty):  
    height, width = image.shape[:2]  
    center = (width/2, height/2)  
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0) 
    M[0, 2] += tx  
    M[1, 2] += ty  
    
    rotated = cv2.warpAffine(image, M, (width, height))  
    
    if scale != 1.0:  
        if scale < 1.0: 
            new_w = int(width * scale)  
            new_h = int(height * scale)  
            
            resized = cv2.resize(rotated, (new_w, new_h))  

            pad_x = (width - new_w) // 2  
            pad_y = (height - new_h) // 2  
            
            result = np.zeros_like(image)  
            
            result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized  
            
        else: 
            large_w = int(width * scale)  
            large_h = int(height * scale)  
            
            resized = cv2.resize(rotated, (large_w, large_h))  
            
            start_x = (large_w - width) // 2  
            start_y = (large_h - height) // 2  
            
            result = resized[start_y:start_y+height, start_x:start_x+width]  
    else:  
        result = rotated  
    
    return result

def augment_image(image, seed=42):  
    if seed is not None:  
        valid_seed = abs(seed) % (2**32 - 1)  
        np.random.seed(valid_seed)  
    
    height, width = image.shape[:2]  
    augmented_image = image.copy()  
    
    augmentations = {  
        'rotation': np.random.random() < 0.7,  
        'scale': np.random.random() < 0.7,  
        'translation': np.random.random() < 0.7,  
        'flip_horizontal': np.random.random() < 0.7  
    }  

    if augmentations['rotation'] or augmentations['scale'] or augmentations['translation']:  
        angle = np.random.uniform(-5, 5) if augmentations['rotation'] else 0  
        scale = np.random.uniform(0.8, 1.2) if augmentations['scale'] else 1.0  
        tx = np.random.uniform(-width*0.1, width*0.1) if augmentations['translation'] else 0  
        ty = np.random.uniform(-height*0.1, height*0.1) if augmentations['translation'] else 0  
        
        try:  
            augmented_image = apply_spatial_transform(  
                augmented_image,  
                angle,  
                scale,  
                tx,  
                ty  
            )  
        except Exception as e:  
            print(f"Warning: Transform failed with error {str(e)}")  
            print(f"Parameters: angle={angle}, scale={scale}, tx={tx}, ty={ty}")  
            return image 
    
    if augmentations['flip_horizontal']:  
        augmented_image = cv2.flip(augmented_image, 1)  
    
    return augmented_image 

def process_dataset(input_dir, output_dir, augment_times):  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]  
    
    print(f"\nProcessing directory: {input_dir}")  
    print(f"Original images: {len(image_files)}")  
    print(f"Augmentation times: {augment_times}")  
    
    for img_file in tqdm(image_files, desc="Processing images"):  
        img_path = os.path.join(input_dir, img_file)  
        original = cv2.imread(img_path)  
        
        if original is None:  
            print(f"Warning: Could not read image {img_path}")  
            continue  
        
        cv2.imwrite(os.path.join(output_dir, img_file), original)  
        
        for i in range(augment_times):  
            augmented = augment_image(original, seed=hash(img_file + str(i)))  
            
            base_name = os.path.splitext(img_file)[0]  
            aug_name = f"{base_name}_aug{i+1}.jpg"  
            
            cv2.imwrite(os.path.join(output_dir, aug_name), augmented)  

def main():  

    dataset_config = {  
        "train": {  
            "0": {  
                "input": r"E:\JTZ\processed_images\Train\class_0",  
                "output": r"E:\JTZ\processed_images\Train_Aug\class_0",  
                "augment_times": 10 
            },  
            "1": {  
                "input": r"E:\JTZ\processed_images\Train\class_1",  
                "output": r"E:\JTZ\processed_images\Train_Aug\class_1",  
                "augment_times": 10
            }  
        }  

    }  

    for split, classes in dataset_config.items():  
        print(f"\nProcessing {split} set:")  
        for class_id, config in classes.items():  
            print(f"\nProcessing class {class_id}:") 
            process_dataset(  
                config["input"],  
                config["output"],  
                config["augment_times"]  
            )  

if __name__ == "__main__":  
    main()   

