import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from services.sign_recognition import ISLContinuousTranslator, SIGN_LABELS

# --- Configuration ---
DATA_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
SEQ_LEN = 60 

class ISLDataset(Dataset):
    def __init__(self, file_paths, labels, label_map):
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = label_map

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_str = self.labels[idx]
        label_idx = self.label_map[label_str]

        seq_tensor = torch.zeros((SEQ_LEN, 126))

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.npy':
                data = np.load(file_path)
            elif ext == '.parquet':
                data = pd.read_parquet(file_path).values
            elif ext == '.csv':
                data = pd.read_csv(file_path).values
            else:
                return seq_tensor, torch.tensor(label_idx, dtype=torch.long)

            actual_len = min(len(data), SEQ_LEN)
            cols = min(data.shape[1], 126)
            seq_tensor[:actual_len, :cols] = torch.tensor(data[:actual_len, :cols], dtype=torch.float32)
            
        except Exception:
            pass 

        return seq_tensor, torch.tensor(label_idx, dtype=torch.long)

def train():
    print(f"🔎 Scanning '{DATA_DIR}' directory for landmark files...")
    file_paths = []
    labels = []

    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: The folder '{DATA_DIR}' does not exist.")
        return

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(('.npy', '.parquet', '.csv')):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()

                for target in SIGN_LABELS:
                    if target in folder_name:
                        file_paths.append(file_path)
                        labels.append(target)
                        break

    if not file_paths:
        print(f"❌ No landmark files found in '{DATA_DIR}' for the selected signs.")
        return

    print(f"✅ Found {len(file_paths)} files across {len(set(labels))} classes.")

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )

    train_dataset = ISLDataset(train_paths, train_labels, label_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ISLContinuousTranslator(input_dim=126, num_classes=len(unique_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "isl_model.pth")
    print("🎉 Training complete! Model saved as 'isl_model.pth'")

if __name__ == "__main__":
    train()