import yaml


def get_classes_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # 确保YAML文件中的确有"classes"键
    if 'classes' in data:
        return data['classes']
    else:
        return None

    # 使用函数并打印结果


if __name__ == '__main__':
    yaml_path = "../../data/config/lvis_cls_1203.yaml"
    classes = get_classes_from_yaml(yaml_path)
    if classes is not None:
        print(classes)
    else:
        print("Classes not found in the YAML file.")
