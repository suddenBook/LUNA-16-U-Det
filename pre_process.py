import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import cv2
from sklearn.cluster import KMeans
from skimage import measure

def load_mhd(file):
    mhdimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(mhdimage)
    origin = np.array(list(mhdimage.GetOrigin()))
    space = np.array(list(mhdimage.GetSpacing()))
    return ct_scan, origin, space

def make_mask(img, center, diam):
    mask = np.zeros_like(img, dtype=np.uint8)
    mask = cv2.circle(mask, (abs(int(center[0])),abs(int(center[1]))), int(abs(diam//2)), 255, -1)
    return mask

def preprocess_data(root, target_root):
    file_list = glob(f"{root}/subset/*.mhd")
    annotations_df = pd.read_csv(f"{root}/annotations.csv")
    annotations_df = annotations_df[annotations_df["diameter_mm"] >= 3.0]

    nodule_mask_dir = os.path.join(target_root, "masks/")
    lungs_roi_dir = os.path.join(target_root, "imgs/")
    os.makedirs(nodule_mask_dir, exist_ok=True)
    os.makedirs(lungs_roi_dir, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for i, file in tqdm(enumerate(np.unique(annotations_df['seriesuid'].values))):
        annotations = annotations_df[annotations_df["seriesuid"] == file]
        ct, origin, space = load_mhd(glob(f"{root}/subset*/{file}.mhd")[0])
        num_z, height, width = ct.shape
        ct_norm = cv2.normalize(ct, None, 0, 255, cv2.NORM_MINMAX)

        for idx, row in annotations.iterrows():
            node_x, node_y, node_z, diam = int(row["coordX"]), int(row["coordY"]), int(row["coordZ"]), int(row["diameter_mm"])
            center = np.array([node_x, node_y, node_z])
            v_center = np.rint((center-origin)/space)
            v_diam = int(diam/space[0]) + 5

            n_neighbour = 2 if 18 < v_diam < 22 else 4
            min_i, max_i = max(0, (int(v_center[2])-n_neighbour)), min((int(v_center[2])+n_neighbour), (num_z-1))
            n = max_i - min_i

            img_norm = ct_norm[int(v_center[2]),:,:]
            img_norm = cv2.resize(img_norm, (512,512))
            img_norm_improved = clahe.apply(img_norm.astype(np.uint8))
            mask = make_mask(img_norm, v_center, v_diam)

            if v_diam > 18:
                img_norm_neighbours = []
                img_norm_improved_neighbours = []
                mask_neighbours = []
                for i in range(min_i, max_i+1):
                    if i == int(v_center[2]):
                        continue
                    im_n = ct_norm[i,:,:]
                    im_n = cv2.resize(im_n, (512,512))
                    im_n_improved = clahe.apply(im_n.astype(np.uint8))
                    dia = int(2*abs(v_center[2]-i))
                    msk = make_mask(im_n, v_center, v_diam-dia)
                    img_norm_neighbours.append(im_n)
                    img_norm_improved_neighbours.append(im_n_improved)
                    mask_neighbours.append(msk)
                assert len(img_norm_neighbours)==len(img_norm_improved_neighbours)==len(mask_neighbours)==n

            mask = cv2.bitwise_and(img_norm, img_norm, mask=cv2.dilate(mask,kernel=np.ones((5,5))))
            pts = mask[mask>0]
            kmeans2 = KMeans(n_clusters=2).fit(np.reshape(pts,(len(pts),1)))
            centroids2 = sorted(kmeans2.cluster_centers_.flatten())
            threshold2 = np.mean(centroids2)
            _, mask = cv2.threshold(mask, threshold2, 255, cv2.THRESH_BINARY)

            if v_diam > 18:
                for i in range(n):
                    mask_neighbours[i] = cv2.bitwise_and(img_norm_neighbours[i], img_norm_neighbours[i], mask=cv2.dilate(mask_neighbours[i],kernel=np.ones((5,5))))
                    _, mask_neighbours[i] = cv2.threshold(mask_neighbours[i], threshold2, 255, cv2.THRESH_BINARY)

            centeral_area = img_norm[100:400, 100:400]
            kmeans = KMeans(n_clusters=2).fit(np.reshape(centeral_area, [np.prod(centeral_area.shape), 1]))
            centroids = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centroids)

            ret, lung_roi = cv2.threshold(img_norm, threshold, 255, cv2.THRESH_BINARY_INV)
            lung_roi = cv2.erode(lung_roi, kernel=np.ones([4,4]))
            lung_roi = cv2.dilate(lung_roi, kernel=np.ones([13,13]))
            lung_roi = cv2.erode(lung_roi, kernel=np.ones([8,8]))

            labels = measure.label(lung_roi)
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
                    good_labels.append(prop.label)
            lung_roi_mask = np.zeros_like(labels)
            for N in good_labels:
                lung_roi_mask = lung_roi_mask + np.where(labels == N, 1, 0)

            contours, hirearchy = cv2.findContours(lung_roi_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            external_contours = np.zeros(lung_roi_mask.shape)
            for i in range(len(contours)):
                if hirearchy[0][i][3] == -1:
                    area = cv2.contourArea(contours[i])
                    if area > 518.0:
                        cv2.drawContours(external_contours,contours,i,(1,1,1),-1)
            external_contours = cv2.dilate(external_contours, kernel=np.ones([4,4]))

            external_contours = cv2.bitwise_not(external_contours.astype(np.uint8))
            external_contours = cv2.erode(external_contours, kernel=np.ones((7,7)))
            external_contours = cv2.bitwise_not(external_contours)
            external_contours = cv2.dilate(external_contours, kernel=np.ones((12,12)))
            external_contours = cv2.erode(external_contours, kernel=np.ones((12,12)))

            img_norm_improved = img_norm_improved.astype(np.uint8)
            external_contours = external_contours.astype(np.uint8)
            extracted_lungs = cv2.bitwise_and(img_norm_improved, img_norm_improved, mask=external_contours)

            mask = mask.astype(np.uint8)
            mask_3d = np.expand_dims(mask, axis=0)  # Convert 2D mask to 3D
            extracted_lungs_3d = np.expand_dims(extracted_lungs, axis=0)  # Convert 2D lungs to 3D
            
            np.save(os.path.join(nodule_mask_dir, f"masks_{idx}.npy"), mask_3d)
            np.save(os.path.join(lungs_roi_dir, f"images_{idx}.npy"), extracted_lungs_3d)

            if v_diam > 18:
                for i in range(n):
                    img_norm_improved_neighbours[i] = img_norm_improved_neighbours[i].astype(np.uint8)
                    extracted_lungs_neighbours = cv2.bitwise_and(img_norm_improved_neighbours[i], img_norm_improved_neighbours[i], mask=external_contours)
                    mask_neighbours[i] = mask_neighbours[i].astype(np.uint8)
                    mask_neighbours_3d = np.expand_dims(mask_neighbours[i], axis=0)  # Convert 2D mask to 3D
                    extracted_lungs_neighbours_3d = np.expand_dims(extracted_lungs_neighbours, axis=0)  # Convert 2D lungs to 3D
                    
                    np.save(os.path.join(nodule_mask_dir, f"masks_{idx}_{i}.npy"), mask_neighbours_3d)
                    np.save(os.path.join(lungs_roi_dir, f"images_{idx}_{i}.npy"), extracted_lungs_neighbours_3d)

if __name__ == "__main__":
    root = os.path.normpath('C:/Users/huixi/OneDrive - Durham University/Final Project/Dataset/LUNA-16/')
    target_root = os.path.normpath('C:/Users/huixi/OneDrive - Durham University/Final Project/Main_Code_Paper/0-Code_Data/Data/')
    preprocess_data(root, target_root)