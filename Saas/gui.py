import cv2
import os
import numpy as np
from typing import List, Tuple
from segment_anything import sam_model_registry, SamPredictor


class ImageSegmentationTool:
    def __init__(self, input_dir: str, output_dir: str, crop_mode: bool = True, model_type='vit_b',
                 checkpath='sam_vit_b_01ec64.pth'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.crop_mode = crop_mode

        self.image_files = [f for f in os.listdir(input_dir) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'))]

        self.sam = sam_model_registry[model_type](checkpoint=checkpath)
        self._ = self.sam.to(device="cuda")  # 注释掉这一行，会用cpu运行，速度会慢很多
        self.predictor = SamPredictor(self.sam)

        self.current_index = 0
        self.input_point = []
        self.input_label = []
        self.selected_mask = None
        self.logit_input = None

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouse_click)

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.input_point.append([x, y])
            self.input_label.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.input_point.append([x, y])
            self.input_label.append(0)

    def apply_mask(self, image: np.ndarray, mask: np.ndarray, alpha_channel: bool = True) -> np.ndarray:
        if alpha_channel:
            alpha = np.zeros_like(image[..., 0])
            alpha[mask == 1] = 255
            image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))
        else:
            image = np.where(mask[..., None] == 1, image, 0)
        return image

    def apply_color_mask(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
                         color_dark: float = 0.5) -> np.ndarray:
        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - color_dark) + color_dark * color[c],
                                      image[:, :, c])
        return image

    def get_next_filename(self, base_path: str, filename: str) -> str:
        name, ext = os.path.splitext(filename)
        for i in range(1, 101):
            new_name = f"{name}_{i}{ext}"
            if not os.path.exists(os.path.join(base_path, new_name)):
                return new_name
        return None

    def save_masked_image(self, image: np.ndarray, mask: np.ndarray, filename: str) -> None:
        if self.crop_mode:
            y, x = np.where(mask)
            y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
            cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
            masked_image = self.apply_mask(cropped_image, cropped_mask)
        else:
            masked_image = self.apply_mask(image, mask)

        filename = filename[:filename.rfind('.')] + '.png'
        new_filename = self.get_next_filename(self.output_dir, filename)

        if new_filename:
            if masked_image.shape[-1] == 4:
                cv2.imwrite(os.path.join(self.output_dir, new_filename), masked_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(os.path.join(self.output_dir, new_filename), masked_image)
            print(f"Saved as {new_filename}")
        else:
            print("Could not save the image. Too many variations exist.")

    def predict_mask(self, image: np.ndarray) -> None:
        if len(self.input_point) > 0 and len(self.input_label) > 0:

            self.predictor.set_image(image)
            input_point_np = np.array(self.input_point)
            input_label_np = np.array(self.input_label)

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point_np,
                point_labels=input_label_np,
                mask_input=self.logit_input[None, :, :] if self.logit_input is not None else None,
                multimask_output=True,
            )

            mask_idx = 0
            num_masks = len(masks)

            while (1):
                color = tuple(np.random.randint(0, 256, 3).tolist())
                image_select = self.image_orign.copy()
                self.selected_mask = masks[mask_idx]
                selected_image = self.apply_color_mask(image_select, self.selected_mask, color)
                mask_info = f'Total: {num_masks} | Current: {mask_idx} | Score: {scores[mask_idx]:.2f} | Press w to confirm | Press d to next mask | Press a to previous mask | Press q to remove last point | Press s to save'
                cv2.putText(selected_image, mask_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.imshow("image", selected_image)

                key = cv2.waitKey(1)
                if key == ord('q') and len(self.input_point) > 0:
                    self.input_point.pop(-1)
                    self.input_label.pop(-1)
                elif key == ord('s'):
                    self.save_masked_image(self.image_crop, self.selected_mask, self.filename)
                elif key == ord('a'):
                    if mask_idx > 0:
                        mask_idx -= 1
                    else:
                        mask_idx = num_masks - 1
                elif key == ord('d'):
                    if mask_idx < num_masks - 1:
                        mask_idx = 1
                    else:
                        mask_idx = 0
                elif key == ord('w'):
                    break
                elif key == ord(" "):
                    self.input_point = []
                    self.input_label = []
                    self.selected_mask = None
                    self.logit_input = None
                    break
                elif key == 27:
                    break
            self.logit_input = logits[mask_idx, :, :]
            # print('max score:', np.argmax(scores), ' select:', mask_idx)

    def run(self) -> None:
        filename = self.image_files[self.current_index]
        self.filename = filename
        self.image_orign = cv2.imread(os.path.join(self.input_dir, filename))
        self.image_crop = self.image_orign.copy()
        image = cv2.cvtColor(self.image_orign.copy(), cv2.COLOR_BGR2RGB)
        self.selected_mask = None
        self.logit_input = None

        while True:
            image_display = self.image_orign.copy()
            display_info = f'{filename} | Press s to save | Press w to predict | Press d to next image | Press a to previous image | Press space to clear | Press q to remove last point '
            cv2.putText(image_display, display_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)
            for point, label in zip(self.input_point, self.input_label):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(image_display, tuple(point), 5, color, -1)
            if self.selected_mask is not None:
                color = tuple(np.random.randint(0, 256, 3).tolist())
                selected_image = self.apply_color_mask(image_display, self.selected_mask, color)

            cv2.imshow("image", image_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                self.input_point = []
                self.input_label = []
                self.selected_mask = None
                self.logit_input = None
            elif key == ord("w"):
                if len(self.input_point) > 0 and len(self.input_label) > 0:
                    self.predict_mask(image)
            elif key == ord('a'):
                self.current_index = max(0, self.current_index - 1)
                self.input_point = []
                self.input_label = []
                break
            elif key == ord('d'):
                self.current_index = min(len(self.image_files) - 1, self.current_index + 1)
                self.input_point = []
                self.input_label = []
                break
            elif key == ord('q') and len(self.input_point) > 0:
                self.input_point.pop(-1)
                self.input_label.pop(-1)
            elif key == ord('s') and self.selected_mask is not None:
                self.save_masked_image(self.image_crop, self.selected_mask, self.filename)
            elif key == 27:
                break
            elif key == ord('e'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    input_dir = '/media/linxu/MobilePan/0-Projects/segment-as-any-solution/Saas/input_img'
    output_dir = 'output'
    crop_mode = True
    model_type = 'vit_b'
    checkpath = '/media/linxu/MobilePan/0-Projects/segment-as-any-solution/Saas/sam_vit_b_01ec64.pth'
    tool = ImageSegmentationTool(input_dir=input_dir, output_dir='output', crop_mode=crop_mode, model_type=model_type,
                                 checkpath='/media/linxu/MobilePan/0-Projects/segment-as-any-solution/Saas/sam_vit_b_01ec64.pth')
    tool.run()
