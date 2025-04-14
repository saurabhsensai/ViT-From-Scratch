from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

def get_transforms(image_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(10), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(data_dir='/data', image_size=224, batch_Size=16, num_workers=4):
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, trandform=get_transforms(image_size, train=True)
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(image_size, train=False)
    )

    train_loader = DataLoader(train_dataset, batch_Size=batch_Size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_Size=batch_Size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader