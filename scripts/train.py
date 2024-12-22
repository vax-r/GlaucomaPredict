import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from src.models.network import GlaucomaNet
from src.data.dataset import GlaucomaDataset
from src.training.trainer import Trainer

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = GlaucomaDataset(
        'data/processed/train',
        transform=transform
    )
    val_dataset = GlaucomaDataset(
        'data/processed/validation',
        transform=transform
    )
    
    # Data loaders
    # num_workers should be decreased if utilizing docker containers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
        # persistent_workers=True # Turn this on when using docker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
        # persistent_workers=True # Turn this on when using docker
    )
    
    # Model
    model = GlaucomaNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        print(len(train_loader), len(val_loader))
        return

    # Training
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer
    )
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/checkpoints/model_epoch_{epoch+1}.pt')

if __name__ == '__main__':
    main()