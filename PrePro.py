import os  
import glob  
import SimpleITK as sitk  
import numpy as np  
from pathlib import Path  
import time  

def resample_image(input_image, is_label=False, new_spacing=[1.0, 1.0, 1.0]):  

    original_spacing = input_image.GetSpacing()  
    original_size = input_image.GetSize()  
    
    new_size = [int(round(original_size[0] * original_spacing[0] / new_spacing[0])),  
                int(round(original_size[1] * original_spacing[1] / new_spacing[1])),  
                int(round(original_size[2] * original_spacing[2] / new_spacing[2]))]  
    
    if is_label:  
        interpolator = sitk.sitkNearestNeighbor  
    else:  
        interpolator = sitk.sitkLinear  
     
    resample = sitk.ResampleImageFilter()  
    resample.SetInterpolator(interpolator)  
    resample.SetOutputSpacing(new_spacing)  
    resample.SetSize(new_size)  
    resample.SetOutputDirection(input_image.GetDirection())  
    resample.SetOutputOrigin(input_image.GetOrigin())  
    resample.SetDefaultPixelValue(0)  
    
    return resample.Execute(input_image)  

def process_patient_data(patient_dir):  
    dicom_path = os.path.join(patient_dir, "dicom.nii.gz")  
    mask_path = os.path.join(patient_dir, "mask.nii")  
    
    if not os.path.exists(dicom_path) or not os.path.exists(mask_path):  
        return  
    
    try:  
        dicom_image = sitk.ReadImage(dicom_path)  
        original_spacing = dicom_image.GetSpacing()  

        resampled_dicom = resample_image(dicom_image, is_label=False)  
        
        mask_image = sitk.ReadImage(mask_path)  
        
        resampled_mask = resample_image(mask_image, is_label=True)  
        
        resampled_dicom_path = os.path.join(patient_dir, "dicom_resampled.nii.gz")  
        resampled_mask_path = os.path.join(patient_dir, "mask_resampled.nii")  
        
        sitk.WriteImage(resampled_dicom, resampled_dicom_path)  
        sitk.WriteImage(resampled_mask, resampled_mask_path)  
        
    except Exception as e:  
        print(f"{patient_dir} ERROR : {str(e)}")  

def main():  
    base_dirs = [  
    ]  
    
    total_patients = 0  
    processed_patients = 0  
    start_time = time.time()  
    
    for base_dir in base_dirs:  
        
        for class_dir in ["0", "1"]:  
            class_path = os.path.join(base_dir, class_dir)  
            if not os.path.exists(class_path):  
                continue  
                
            a_folder = os.path.join(class_path, "A")  
            if not os.path.exists(a_folder):  
                continue  

            patient_folders = [f for f in os.listdir(a_folder) if os.path.isdir(os.path.join(a_folder, f))]  
            
            for patient in patient_folders:  
                total_patients += 1  
                patient_dir = os.path.join(a_folder, patient)  
                process_patient_data(patient_dir)  
                processed_patients += 1  
    
    end_time = time.time()  

if __name__ == "__main__":  
    main()  
