import torch
import yaml
from pathlib import Path
from src.models.network import GlaucomaNet
from src.data.dataset import GlaucomaDataset
from src.evaluation.metrics import ModelEvaluator
from torchvision import transforms
from torch.utils.data import DataLoader

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "config.yaml"
    model_path = project_root / "models" / "checkpoints" / "model_epoch_100.pt"
    results_dir = project_root / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = GlaucomaDataset(
        project_root / "data" / "processed" / "test",
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Load model
    model = GlaucomaNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, device)
    y_true, y_pred, y_prob = evaluator.evaluate()
    
    # Save results
    evaluator.plot_confusion_matrix(
        y_true, y_pred,
        results_dir / "confusion_matrix.png"
    )
    evaluator.plot_roc_curve(
        y_true, y_prob,
        results_dir / "roc_curve.png"
    )
    
    # Print classification report
    report = evaluator.generate_report(y_true, y_pred)
    print("\nEvaluation Results:")
    print(report)
    
    # Save report to file
    with open(results_dir / "evaluation_report.txt", 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
