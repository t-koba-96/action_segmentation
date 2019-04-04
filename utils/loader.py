import torch



def my_collate_fn(batch):
    
    images = []
    targets = []
    labels=[]
    for sample in batch:
        image, target,label = sample
        images.append(image)
        targets.append(target)
        labels.append(label)
    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)
    labels = torch.stack(labels, 0)

    return [images, targets, labels]