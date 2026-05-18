# train_worker.py
import os,sys
os.chdir("./trinity/buffer/operators/filters")
sys.path.append("./trinity/buffer/operators/filters")
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import argparse
import json
import os
from utils import PreferenceClassifier, PreferencePairDataset

def evaluate_preference_classifier(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for h_max, h_min, weight in dataloader:
            h_max, h_min, weight = h_max.to(device), h_min.to(device), weight.to(device)
            batch_size = h_max.size(0)

            weights = torch.cat([weight, weight], dim=0)

            h_diff_pos = h_max - h_min
            h_diff_neg = h_min - h_max
            
            inputs = torch.cat([h_diff_pos, h_diff_neg], dim=0)
            
            labels_pos = torch.ones(batch_size, 1, device=device, dtype=torch.float32)
            labels_neg = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
            
            labels = torch.cat([labels_pos, labels_neg], dim=0)

            outputs = model(inputs)
            loss_per_sample = criterion(outputs, labels)
            weighted_loss = (loss_per_sample.squeeze() * weights).mean()
            

            total_loss += weighted_loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def train_binary_classifier_for_hidden_space(raw_data: List, 
                                            model: PreferenceClassifier,
                                            min_epochs: int = 10, 
                                            max_epochs: int = 100,
                                            early_stop_acc: float = 0.85,
                                            batch_size: int = 64, 
                                            learning_rate: float = 1e-4,
                                            test_split: float = 0.1, 
                                            model_save_path: str = "this_folder/") -> Tuple[str, float]:
    
    print("\n" + "="*50)
    print("Training a binary classifier to align the latent space...")
    print("="*50)
    
    done_path = os.path.join(model_save_path, 'hidden_space_classifier.done')
    model_save_path = os.path.join(model_save_path, 'hidden_space_classifier.pth')
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    try:
        full_dataset = PreferencePairDataset(raw_data)
        
        total_size = len(full_dataset)
        test_size = int(test_split * total_size)
        train_size = total_size - test_size
        
        if test_size == 0 and total_size > 0:
            print("Warning: Dataset too small, test set size is 0. Using the entire dataset for training.")
            train_size = total_size
            train_dataset = full_dataset
            test_dataset = None
            test_dataloader = None
        elif total_size == 0:
            print("Error: Dataset is empty, cannot train.")
            return
        else:
            train_dataset, test_dataset = random_split(
                full_dataset, 
                [train_size, test_size]
            )
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Total preference pairs in dataset: {total_size}")
        print(f" - Training set size: {train_size}")
        print(f" - Test set size: {test_size}")
        
    except ValueError as e:
        print(f"Error preparing dataset: {e}")
        return

    print("Starting training...")
    
    best_test_accuracy = 0.0
    stop_training = False

    for epoch in range(max_epochs):
        model.train()
        device = next(model.parameters()).device
        train_total_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        for h_max, h_min, weight in train_dataloader:
            h_max, h_min, weight = h_max.to(device), h_min.to(device), weight.to(device)
            batch_size_actual = h_max.size(0)

            h_diff_pos = h_max - h_min
            h_diff_neg = h_min - h_max
            inputs = torch.cat([h_diff_pos, h_diff_neg], dim=0)
            
            labels_pos = torch.ones(batch_size_actual, 1, device=device, dtype=torch.float32)
            labels_neg = torch.zeros(batch_size_actual, 1, device=device, dtype=torch.float32)
            labels = torch.cat([labels_pos, labels_neg], dim=0)

            weights = torch.cat([weight, weight], dim=0)
            
            outputs = model(inputs)
            loss_per_sample = criterion(outputs, labels)
            weighted_loss = (loss_per_sample.squeeze() * weights).mean()
            
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            train_total_loss += weighted_loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct_predictions += (preds == labels).sum().item()
            train_total_samples += labels.size(0)
            
        train_avg_loss = train_total_loss / len(train_dataloader)
        train_accuracy = train_correct_predictions / train_total_samples
        
        if test_dataloader:
            test_avg_loss, test_accuracy = evaluate_preference_classifier(
                model, 
                test_dataloader, 
                criterion, 
                device
            )
            print(f"Epoch [{epoch+1}/{min_epochs}] | Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.4f} | Test Loss: {test_avg_loss:.4f}, Test Acc: {test_accuracy:.4f}")

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"--> Test accuracy improved ({best_test_accuracy:.4f}). Model saved to: {model_save_path}")

            if epoch >= min_epochs - 1:
                if test_accuracy >= early_stop_acc:
                    print(f"Early stopping triggered: test accuracy ({test_accuracy:.4f}) reached target ({early_stop_acc:.4f}).")
                    stop_training = True
            
            if stop_training:
                break
        else:
            print(f"Epoch [{epoch+1}/{min_epochs}] | Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            torch.save(model.state_dict(), model_save_path)

    print("\n" + "="*50)
    print("Training complete!")
    if test_dataloader:
        print(f"Best test accuracy: {best_test_accuracy:.4f}")
    else:
        print(f"Final training accuracy: {train_accuracy:.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*50)

    with open("/home/daoyuan_dj/zt_RLHF_RM_Refine/log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n\nFinal training accuracy: {best_test_accuracy:.4f}")

    with open(done_path, "w") as f:
        f.write("ok")

    return model_save_path, best_test_accuracy, total_size

def train_standalone(args):
    # 1. Load data produced by the main process
    data_path = os.path.join(args.model_save_path, 'raw_data.pkl')
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # 2. Initialize model
    # input_dim can be inferred from the data or passed as an argument
    input_dim = raw_data[0]['responses'][0]['hidden_state'].shape[-1]
    model = PreferenceClassifier(input_dim=input_dim, hidden_layer_size=args.output_hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_saved_path, best_test_accuracy, total_size = train_binary_classifier_for_hidden_space(raw_data, model,
                                             args.min_epochs, 
                                             args.max_epochs,
                                             args.early_stop_acc,
                                             args.batch_size, 
                                             args.learning_rate,
                                             args.test_split,
                                             args.model_save_path)

    # Save final accuracy metrics to a json file for the main process to read
    result_metrics = {
        "total_preference_pair": total_size,
        "best_test_accuracy": float(best_test_accuracy),
        "model_path": model_saved_path
    }
    result_log_path = os.path.join(args.model_save_path, 'result_log.json')

    with open(result_log_path, 'w') as f:
        json.dump(result_metrics, f)
    
    print(f"Worker: Training complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--output_hidden_dim", type=int, default=512)
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop_acc", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--test_split", type=float, default=0.1)
    
    
    
    args = parser.parse_args()
    train_standalone(args)