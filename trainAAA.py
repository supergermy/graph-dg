import torch
from torch_geometric.loader import DataLoader
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from torch_geometric.data import Batch

from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
import os
import numpy as np

# Import dataset
def get_dataset(split, embedding):
    if embedding == "prott5":
        if split == "thermompnn":
            # from dataset_thermompnn import ThermoMPNNInMemoryDatasetProtT5 as Dataset
            # ds = "dataset_thermompnn"
            from dataset0403 import MegaThermoProtT5Template as Dataset
            ds = "dataset0403"
            
    elif embedding == "esmif":
        print("ESM-IF1 is not ready yet. Please choose other option.")
        raise KeyError
    else:
        print(f"Could not found a dataset from the given options. split: {split}, embedding: {embedding}")
        raise KeyError
    print(f"🤹‍♂️ Found dataset at {ds}. embedding: {embedding}, split: {split}")
    
    return Dataset

# Import model
def get_model(model):
    if model == "GATfly3_pair_distogram":
        from model0408 import GATfly3_pair_distogram as Model
        md = "model0408"
    elif model == "GCNfly3_pair_distogram_dual":
        from model0408 import GCNfly3_pair_distogram_dual as Model
        md = "model0408"
    else:
        print(f"Could not found {model}")
        raise KeyError
    print(f"👾 Found model architecture at {md}. model: {model}")
    
    return Model("KLD")


def parse():
    parser = argparse.ArgumentParser(description='GRAPH-DG')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--rbf-scale', type=int, default=2,
                            help="Returns softmax(rbf_scale*RBF) while generating the distograms.")
    parser.add_argument("--conf-weight", action="store_true")
    parser.add_argument("--flip", action="store_true",
                            help="If set, flips validation and test datasets.")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    # parser.add_argument("--hidden-dim", type=int, default=64)
    # parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--distogram-bin", type=int, default=25)
    parser.add_argument("--dataset-split", type=str, default="thermompnn")
    parser.add_argument("--model-dir", type=str, default= "model-epochs/")
    parser.add_argument("--wandb-suffix", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--ignore-unreliable", action="store_true")
    parser.add_argument("--embedding", type=str, required=True, choices=['prott5','esmif','onehot'],
                            help="Graph node embedding type: ProtT5-XL-UniRef50[1024], ESM-IF1[512], One-Hot-Encoding[20]")
    args = parser.parse_args()
    
    return args


def set_wandb(args):
    # Run wandb in offline
    os.environ["WANDB_API_KEY"] = "0f2861a95a5fc403cbdaf9bffedb88a439644fc8"
    os.environ["WANDB_MODE"] = "dryrun"
    
    wandb.init(project=args.model_name, name = f'{now}-{args.wandb_suffix}' if args.wandb_suffix else f'{now}', config=vars(args))
    wandb.watch(model, log='all')
    os.makedirs(os.path.join(args.model_dir, args.model_name, now), exist_ok=True)


def sets_and_loaders(dataset: torch.utils.data.Dataset, args) -> DataLoader:
    train_dataset = dataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = dataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    test_dataset = dataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def get_debug(dataset, args):
    debug_dataset = dataset(split="debug")
    debug_loader = DataLoader(debug_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return debug_loader


def cal_distogram_metrics(labels: torch.tensor , preds: torch.tensor) -> list[float]:
    
    # [0, args.distogram_bin-1] -> [-1., 5.]
    rescaling_factor = (args.distogram_bin-1)/6
    y = labels / rescaling_factor - 1
    y_hat = preds / rescaling_factor - 1
    
    acc = (y_hat == y)
    acc = len(acc[acc]) / len(acc) # clever
    mse = torch.square((y_hat) - (y)).mean()
    rmse = torch.sqrt(mse)
    
    inacc = y_hat - y
    inacc = len(inacc[torch.abs(inacc) <= .5])/len(y)
    
    return float(acc), float(mse), float(rmse), float(inacc), y, y_hat


class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def get_metrics() -> dict:
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
        "pearson": PearsonCorrCoef(),
    }

def check_isnan(x:torch.tensor) -> bool :
    """check if input tensor contains nan
    
    Args:
        x (torch.tensor): any tensor
    
    Returns:
        bool: True if x contains nan.
    """
    return torch.isnan(x).any()

def distogram(D:torch.tensor, args) -> torch.tensor :
    D_min, D_max, D_count = 0., 1., args.distogram_bin
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    softmax = torch.nn.Softmax(dim=1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return softmax(args.rbf_scale*RBF)


def get_high_conf_loss(pred, label, criterion, coeff=1.0):
    max_pred, _ = torch.max(pred, dim=1)
    min_pred, _ = torch.min(pred, dim=1)
    conf_mat = max_pred / min_pred
    conf_bool = (conf_mat >= conf_mat.median())
    high_conf_label = label[conf_bool]
    high_conf_pred = pred[conf_bool]
    high_conf_loss = criterion(high_conf_pred, high_conf_label)
    
    return coeff * high_conf_loss


# Training and Validation function
def train_and_evaluate(model, dataset, optimizer, criterion, args):
    # Setup datasets and dataloaders
    if args.debug:
        debug_loader = get_debug(dataset, args)
        train_loader, val_loader, test_loader = debug_loader, debug_loader, debug_loader
    else:
        train_loader, val_loader, test_loader = sets_and_loaders(dataset, args)
    
    best_val_r2 = float('-inf')
    best_epoch = 0
    
    for epoch in tqdm(range(1,args.epochs+1), desc="Epoch:"):
        # Training
        model.train()
        total_train_loss = 0
        labels, preds = torch.tensor([]), torch.tensor([])
        metrics = get_metrics()
        
        for in_data, tp_data in tqdm(train_loader, desc=f"[Training]   | Epoch [{epoch}/{args.epochs}] "):
            if args.ignore_unreliable:
                in_data = Batch().from_data_list(in_data[-1 < in_data.y])
                in_data = Batch().from_data_list(in_data[ 5 > in_data.y])
            in_data = in_data.to(device)
            tp_data = tp_data.to(device)
            label = (torch.clamp(in_data.y,-1,5) + 1) / 6
            label = distogram(label, args)
            optimizer.zero_grad()
            pred = model(in_data, tp_data)
            assert not check_isnan(pred), "model output contains nan"
            loss = criterion(pred, label)
            
            if args.conf_weight:
                loss += 0.5*get_high_conf_loss(pred, label, criterion)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
                
            labels = torch.concat((labels,torch.argmax(label, dim=-1).detach().cpu()))
            preds = torch.concat((preds,torch.argmax(pred, dim=-1).detach().cpu()))
        
        avg_train_loss = total_train_loss / len(train_loader)
        acc, mse, rmse, inacc, y, y_hat = cal_distogram_metrics(labels, preds)
        
        for metric in metrics.values():
            metric.update(y_hat, y)
        metric_results_train = {name: metric.compute() for name, metric in metrics.items()}
        
        log_data = {'epoch': epoch, 'train_JSDloss': avg_train_loss, 'train_acc': acc, 'train_inacc': inacc}
        log_data.update({f"train_{name}": value.item() for name, value in metric_results_train.items()})
        print(f"🏋️‍♂️ train_JSDloss: {avg_train_loss:.4f}, train_acc: {acc:.4f}, train_mse: {mse:.4f}, train_rmse: {rmse:.4f}, train_inacc: {inacc:.4f}")
        
        ####### END OF TRAINING CODE #######
        
        # Validation
        model.eval()
        total_val_loss = 0
        labels, preds = torch.tensor([]), torch.tensor([])
        metrics = get_metrics()
        
        with torch.no_grad():
            for in_data, tp_data in tqdm(val_loader, desc=f"[Validation] | Epoch [{epoch}/{args.epochs}] "):
                if args.ignore_unreliable:
                    in_data = Batch().from_data_list(in_data[-1 < in_data.y])
                    in_data = Batch().from_data_list(in_data[ 5 > in_data.y])
                in_data = in_data.to(device)
                tp_data = tp_data.to(device)
                label = (torch.clamp(in_data.y,-1,5) + 1) / 6
                label = distogram(label, args)
                pred = model(in_data, tp_data)
                loss = criterion(pred, label)
                
                if args.conf_weight:
                    loss += 0.5*get_high_conf_loss(pred, label, criterion)
                
                total_val_loss += loss.item()
                
                labels = torch.concat((labels,torch.argmax(label, dim=-1).detach().cpu()))
                preds = torch.concat((preds,torch.argmax(pred, dim=-1).detach().cpu()))
        
        avg_val_loss = total_val_loss / len(val_loader)
        acc, mse, rmse, inacc, y, y_hat = cal_distogram_metrics(labels, preds)
        
        for metric in metrics.values():
            metric.update(y_hat, y)
        metric_results_val = {name: metric.compute() for name, metric in metrics.items()}
        
        log_data.update({'val_JSDloss': avg_val_loss, 'val_acc': acc, 'val_inacc': inacc})
        log_data.update({f"val_{name}": value.item() for name, value in metric_results_val.items()})
        print(f"🧪 val_JSDloss: {avg_val_loss:.4f}, val_acc: {acc:.4f}, val_mse: {mse:.4f}, val_rmse: {rmse:.4f}, val_inacc: {inacc:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name, now, f'epoch-{epoch}.pth'))
        
        ####### END OF VALIDATION CODE #######
        
        ### testing ###
        model.eval()
        total_test_loss = 0
        labels, preds = torch.tensor([]), torch.tensor([])
        metrics = get_metrics()
        
        with torch.no_grad():
            for in_data, tp_data in tqdm(test_loader, desc=f"[Testing]   | Epoch [{epoch}/{args.epochs}] "):
                if args.ignore_unreliable:
                    in_data = Batch().from_data_list(in_data[-1 < in_data.y])
                    in_data = Batch().from_data_list(in_data[ 5 > in_data.y])
                in_data = in_data.to(device)
                tp_data = tp_data.to(device)
                label = (torch.clamp(in_data.y,-1,5) + 1) / 6
                label = distogram(label, args)
                pred = model(in_data, tp_data)
                loss = criterion(pred, label)
                
                if args.conf_weight:
                    loss += 0.5*get_high_conf_loss(pred, label, criterion)
                
                total_test_loss += loss.item()
                
                labels = torch.concat((labels,torch.argmax(label, dim=-1).detach().cpu()))
                preds = torch.concat((preds,torch.argmax(pred, dim=-1).detach().cpu()))
        
        acc, mse, rmse, inacc, y, y_hat = cal_distogram_metrics(labels, preds)
        avg_test_loss = total_test_loss / len(test_loader)
        
        for metric in metrics.values():
            metric.update(y_hat, y)
        metric_results_test = {name: metric.compute() for name, metric in metrics.items()}
        
        # Log loss to wandb
        log_data.update({'test_loss': avg_test_loss, 'test_acc': acc, 'test_inacc': inacc})
        log_data.update({f"test_{name}": value.item() for name, value in metric_results_test.items()})
        print(f"👀 test_JSDloss: {avg_test_loss:.4f}, test_acc: {acc:.4f}, test_mse: {mse:.4f}, test_rmse: {rmse:.4f}, test_inacc: {inacc:.4f}")
        
        wandb.log(log_data)
        
        if best_val_r2 <= metric_results_val['r2']:
            best_val_r2 = metric_results_val['r2']
            best_epoch = epoch
            print(f"⭐️ [Epoch {epoch}] New best val_r2 = {best_val_r2:.4f}")
            for name, value in metric_results_train.items():
                print(f"train_{name}: {value.item():.4f}")
            for name, value in metric_results_val.items():
                print(f"val_{name}: {value.item():.4f}")
            for name, value in metric_results_test.items():
                print(f"test_{name}: {value.item():.4f}")
                
    return best_epoch

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    now = datetime.now().strftime("%m%d-%H%M")
    print(f'⏰ Now is {now}')
    print('👹 This is NOT valid model training script, its AAA!')
    
    # Setup argparse and configuration
    args = parse()
    if args.debug:
        print("🐞 Warning! Entering debugging mode... Using debugging dataset(random 1% of train) for train/val/test")
        
    if args.ignore_unreliable:
        print("⚠️  Warning! --ignore-unreliable is set, ignoring all ~ -1 < dG_ML < 5")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🤖 Using device: {device}')
    
    # Initialize dataset, model, optimizer, and loss function
    dataset = get_dataset(args.dataset_split, args.embedding)
    model = get_model(args.model_name)
    
    if args.checkpoint:
        print(f"🧙‍♂️ Loading parameters of {args.checkpoint} and train")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    if args.conf_weight:
        print("⚠️  Warning! conf-weight is set, am going to penalize high-confidence incorrect predictions more than low-confidence ones.")
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = JSD()
    criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    # Initialize wandb run and log model
    set_wandb(args)
    
    # Main Loop
    best_epoch = train_and_evaluate(model, dataset, optimizer, criterion, args)
    
    # Finish wandb run
    wandb.finish()

