import os
import json
import shutil

def create_single_image_coco():
    base_dir = r"C:\Users\anna_\OneDrive\Dokumente\Object Detection\vos-main\COCO_DATASET_ROOT"
    temp_dir = r"C:\Users\anna_\OneDrive\Dokumente\Object Detection\vos-main\TEMP_COCO"
    
    # Create temporary structure
    os.makedirs(os.path.join(temp_dir, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "annotations"), exist_ok=True)
    
    # Copy the single image
    src_image = os.path.join(base_dir, "val2017", "Bear.jpg")
    dst_image = os.path.join(temp_dir, "val2017", "000000000001.jpg")
    shutil.copy2(src_image, dst_image)
    
    # Create annotation file for single image
    coco_format = {
        "images": [{
            "id": 1,
            "file_name": "000000000001.jpg",
            "width": 640,
            "height": 480,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}]
    }
    
    with open(os.path.join(temp_dir, "annotations", "instances_val2017.json"), "w") as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Created single image COCO structure at: {temp_dir}")
    return temp_dir

if __name__ == "__main__":
    temp_path = create_single_image_coco()