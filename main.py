import torch
import os
from dataloader import get_cifar10_loaders
from backbone import DualBackbone
from graph_builder import HomophilyGraphBuilder

def extract_and_cache(num_batches=8, cache_path="cache_graph_features.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting features on: {device}...")

    train_loader, _ = get_cifar10_loaders(batch_size=128, img_size=224, root="./data")
    backbones = DualBackbone(pretrained=True).to(device)
    backbones.eval()

    f1_list, f2_list, img_list = [], [], []
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            f1, f2 = backbones(images)
            f1_list.append(f1.cpu())
            f2_list.append(f2.cpu())
            img_list.append(images.cpu())
            print(f"  Processed batch {i+1}/{num_batches} ({images.shape[0]} images)")

    F1 = torch.cat(f1_list, dim=0).to(device)
    F2 = torch.cat(f2_list, dim=0).to(device)
    all_images = torch.cat(img_list, dim=0)

    # Threshold 0.30 calibrated to empirical correlation distribution (top ~2%)
    print("Building homophily-heterophily graph...")
    builder = HomophilyGraphBuilder(threshold=0.30, local_threshold=0.40).to(device)
    adj, core_nodes, local1, local2, bridges = builder(F1, F2)

    cache_data = {
        "F1": F1.cpu(), "F2": F2.cpu(),
        "images": all_images,
        "adj": adj.cpu(),
        "core_nodes": core_nodes.cpu(),
        "local1": local1.cpu(), "local2": local2.cpu(),
        "bridges": bridges
    }
    torch.save(cache_data, cache_path)
    print(f"\nCached to {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.1f} MB)")
    print(f"  Total Samples : {F1.shape[0]}")
    print(f"  Core Nodes    : {len(core_nodes)} | BB1 Local: {len(local1)} | BB2 Local: {len(local2)}")
    print(f"  Bridge Edges  : {len(bridges)}")
    return cache_data

if __name__ == "__main__":
    extract_and_cache()
