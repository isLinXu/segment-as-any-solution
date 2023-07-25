import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys

from PIL import Image

from Saas.utils.clip_tool import Clip_Predict, classes
from Saas.utils.image_tool import concat_images
from Saas.utils.os_tool import mkdir_dir

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from tqdm import tqdm


class SaasPredictor:

    def __init__(self, sam_checkpoint, model_type, device, dev=False):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device

        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)

        self.mask_generator_2 = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        self.dev = dev
        self.save_path = 'dst_img.jpg'

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    def seg_anns(self, original_image):
        anns = self.mask_generator_2.generate(original_image)
        global img_bgr
        clip_prompt = []
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0

        count = 1
        src = original_image.copy()

        if self.dev:
            mkdir_dir("../temp/croped_object")
            mkdir_dir("../temp/segment_object")
            mkdir_dir("../temp/original_segment")

        for i in tqdm(range(len(sorted_anns))):
            ann = sorted_anns[i]
            area = ann['area']
            m = ann['segmentation']

            # x, y, w, h = ann['crop_box']
            x, y, w, h = cv2.boundingRect(m.astype(np.uint8))

            # 截取出最外围矩形框的部分图像
            cropped_image = original_image[y:y + h, x:x + w]

            img_crop = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_crop)

            clip_predictor = Clip_Predict(classes)
            instance_info = clip_predictor.predict_clip_classes(im_pil)
            instance_name, instance_confidences = instance_info
            clip_prompt.append(instance_name)

            # 设置随机颜色，可以用于目标检测画不同目标边界框的随机颜色
            bgr = np.random.randint(0, 255, 3, dtype=np.int32)
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            cv2.rectangle(src, (x, y), (x + w, y + h), color, 2)

            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

            # Convert the image to display in OpenCV
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)

            if self.dev:
                # 在矩形框上添加类别标签和置信度
                label_text = instance_name + ': ' + str(instance_confidences)
                cv2.putText(src, label_text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                print("indices_name:", instance_name)
                # # 绘制矩形框
                print("x, y, w, h:", x, y, w, h)
                cv2.imshow('bbox Image', src)
                # Show the image in an OpenCV window
                cv2.imshow("img", img_bgr)
                cv2.waitKey(0)
            # Convert original image to BGRA format
            original_image_bgra = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
            # Set the alpha channel of the mask's complement as transparent
            original_image_bgra[np.bitwise_not(m.astype(np.bool_))] = [0, 0, 0, 0]

            # Apply the mask to the original image
            original_segment = cv2.bitwise_and(original_image_bgra, original_image_bgra, mask=m.astype(np.uint8))

            # cv2.imshow('Cropped Image', cropped_image)
            cv2.imwrite(f"../temp/croped_object/croped_object_{count}.png", cropped_image)
            # Save the image to local disk
            cv2.imwrite(f"segmented_object/segmented_object_{count}.png", img_bgr)
            cv2.imwrite(f"../temp/segment_object/segmented_object_{count}.png", img_bgr)
            # Save the RGBA image with a transparent background
            cv2.imwrite(f"original_segment/original_segment_{count}.png", original_segment)
            cv2.imwrite(f"../temp/original_segment/original_segment_{count}.png", original_segment)
            # if self.dev:
            #     cv2.waitKey(0)
            count += 1

        # concatenate images horizontally
        concat_image = concat_images(original_image, src, img_bgr)
        cv2.imshow('Concatenated Image', concat_image)
        cv2.imwrite(self.save_path, concat_image)
        cv2.waitKey(0)
        ax.imshow(img)
        print("clip_prompt:", clip_prompt)


if __name__ == '__main__':
    img_path = "/media/linxu/MobilePan/0-Projects/psd-ai/images/dog.jpg"
    image = cv2.imread(img_path)
    image = cv2.resize(image, (512, 512))
    sam_checkpoint = "../ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    saaspredictor = SaasPredictor(sam_checkpoint, model_type, device)
    saaspredictor.seg_anns(image)
