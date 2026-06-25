import os


def load_lines(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def parse_img_label_list(filepath):
    lines = load_lines(filepath)
    img_names = [l.split()[0] for l in lines]
    label_names = [l.split()[1] for l in lines]
    return img_names, label_names


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
