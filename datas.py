import torch


def train_path_list(start,end):
    video_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/image_a/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/image_b/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/image_c/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/image_d/train/",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/image_e/train/"]

    label_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/class_a/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/class_b/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/class_c/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/class_d/train.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/class_e/train.csv"]

    return video_path_list[start-1:end],label_path_list[start-1:end]


def test_path_list(start,end):
    video_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/image_a/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/image_b/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/image_c/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/image_d/test/",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/image_e/test/"]

    label_path_list=["../../../local/dataset/work_detect/mogi_data/worker_a/class_a/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_b/class_b/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_c/class_c/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_d/class_d/test.csv",
                     "../../../local/dataset/work_detect/mogi_data/worker_e/class_e/test.csv"]

    return video_path_list[start-1:end],label_path_list[start-1:end]

def class_list():
    class_list=["n","tp","al","ac","ca","gd","as","hp","st","ch","co"]

    return class_list
    