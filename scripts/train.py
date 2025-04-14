import torch 
import yaml 
from vit.model import VisionTransformer
from vit.dataset import get_dataloaders
from vit.trainer import train_model, evaluate_model
from vit.utils import get_device


def load_configs(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def main():
    config = load_configs('../configs/config.yaml')

    torch.manual_seed(42)

    device = get_device()
    print(f'Using Device: {device}')


    train_loader, val_loader = get_dataloaders(
        data_dir = config['data_dir'], 
        image_size=config['image_size'], 
        batch_Size=config['batch_size'], 
        num_workers=config['num_workers']
        )
    
    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['num_layers'],
        num_heads=config['num_heads']
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epoche=config['num_epoche'], device=device
    )

    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()