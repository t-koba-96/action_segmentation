import torch



def my_collate_fn(batch):
    
    images = []
    targets = []
    labels=[]
    poses=[]
    for sample in batch:
        image, target,label, pose = sample
        images.append(image)
        targets.append(target)
        labels.append(label)
        poses.append(pose)
    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)
    labels = torch.stack(labels, 0)
    poses = torch.stack(poses, 0)

    return [images, targets, labels, poses]