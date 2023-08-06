import clip
import cv2
import numpy as np
import torch
from PIL import Image

from Saas.utils.yaml_tool import get_classes_from_yaml



class Clip_Predict():
    def __init__(self, classes, debug=True):
        '''
        :param classes:
        :param debug:
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device)
        self.model = model
        self.device = device
        self.preprocess = preprocess
        self.classes = classes
        self.debug = debug

    def predict_clip_classes(self, image):
        '''
        :param image:
        :return:
        '''
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        # 生成文字描述
        text_inputs = torch.cat([clip.tokenize(f"This is a photo of a {c}") for c in classes]).to(self.device)

        # 特征编码
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        # 选取参数最高的标签
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # 对图像描述和图像特征
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        # 输出结果
        print("\nTop predictions:\n")
        print('classes:{} score:{:.2f}'.format(self.classes[indices.item()], values.item()))

        return classes[indices.item()], values.item()

    def show_clip_predict(self,img):
        numpy_img = np.array(img)
        img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))
        # 文本位置
        x, y = 10, 30
        # 字体及字号
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        # 绘制文本
        text = f'{class_name}: {conf:.2f}'
        cv2.putText(img, text, (x, y), font, font_scale, (0, 255, 0), thickness)
        # 显示图像
        cv2.imshow('result', img)
        cv2.waitKey(0)

if __name__ == "__main__":
    img_path = '../../data/input_img/cat.jpg'

    image = Image.open(img_path)
    yaml_path = "../../data/config/coco_cls_80.yaml"
    classes = get_classes_from_yaml(yaml_path)
    clip_predictor = Clip_Predict(classes)
    indices_name = clip_predictor.predict_clip_classes(image)
    class_name, conf = indices_name
    print("indices_name:", indices_name)
    clip_predictor.show_clip_predict(image)