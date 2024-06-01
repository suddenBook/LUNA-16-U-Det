import os
import concurrent.futures
import SimpleITK as sitk
import matplotlib.pyplot as plt

def convert_mhd_to_png(mhd_file_path, output_dir):
    try:
        image = sitk.ReadImage(mhd_file_path)
        
        pixel_type = image.GetPixelIDTypeAsString()
        
        if pixel_type in ["16-bit signed integer", "16-bit unsigned integer"]:
            image = sitk.Cast(image, sitk.sitkFloat32)
        
        windowed_image = sitk.IntensityWindowing(image, windowMinimum=-1000, windowMaximum=300)
        image_array = sitk.GetArrayFromImage(windowed_image)

        filename = os.path.splitext(os.path.basename(mhd_file_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        for i in range(image_array.shape[0]):
            output_path = os.path.join(output_dir, f"{filename}_{i}.png")
            if not os.path.exists(output_path):
                plt.imsave(output_path, image_array[i], cmap='gray')

    except Exception as e:
        print(f"Error processing {mhd_file_path}: {str(e)}")

input_dir = "../Dataset/subset/"
output_dir = "../Dataset/jpg_file/"

with concurrent.futures.ThreadPoolExecutor() as executor:
    tasks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".mhd"):
            mhd_file_path = os.path.join(input_dir, filename)
            task = executor.submit(convert_mhd_to_png, mhd_file_path, output_dir)
            tasks.append(task)

    concurrent.futures.wait(tasks)

print("Conversion completed.")