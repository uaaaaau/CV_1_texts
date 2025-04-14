import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from gensim.models import FastText

# ПАРАМЕТРЫ
CHUNK_SIZE = 500
STEP_SIZE = 200
EPOCHS = 60
LR = 0.000267
BATCH_SIZE = 64
VAL_SIZE = 0.2
EMBEDDING_DIM = 300  # размерность векторов FastText

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=200, step=200):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        chunk = tokens[start:start+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        start += step
    return chunks

def prepare_author_chunks(author_files, chunk_size=200, step=200):
    texts, labels, label2author = [], [], {}
    for label_idx, filepath in enumerate(author_files):
        author_name = os.path.splitext(os.path.basename(filepath))[0]
        label2author[label_idx] = author_name
        chunks = chunk_text(read_file(filepath), chunk_size, step)
        texts.extend(chunks)
        labels.extend([label_idx] * len(chunks))
    return texts, labels, label2author

# Вычисление усреднённого вектора чанка с использованием FastText
def compute_average_vector(text, model, embedding_dim):
    tokens = text.lower().split()
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

class TextDataset(Dataset):
    def __init__(self, X_vectors, y_labels):
        # X_vectors – numpy массив размерности (n_samples, embedding_dim)
        self.X = torch.tensor(X_vectors, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Обновлённый класс MLP с дополнительным скрытым слоем
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=100, hidden_dim2=50, output_dim=6, dropout_rate=0.47):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc, best_state_dict = 0.0, None

    for epoch in range(1, epochs+1):
        model.train()
        total_train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        val_loss, all_preds, all_targets = 0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        print(f"Эпоха [{epoch}/{epochs}]: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    if best_state_dict:
        model.load_state_dict(best_state_dict)
    return model

def predict_text(model, ft_model, text, embedding_dim, chunk_size=200, step=200, device='cpu'):
    model.eval()
    chunks = chunk_text(text, chunk_size, step)
    if not chunks:
        return 0
    # Вычисление усреднённого вектора для каждого чанка
    X = np.array([compute_average_vector(chunk, ft_model, embedding_dim) for chunk in chunks])
    X_torch = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_torch)
        return torch.argmax(logits.sum(dim=0)).item()

def predict_text_dynamic(model, ft_model, text, embedding_dim, device='cpu'):
    chunks = chunk_text(text, CHUNK_SIZE, STEP_SIZE)
    if not chunks:
        return 0  # fallback в случае отсутствия чанков
    X = np.array([compute_average_vector(chunk, ft_model, embedding_dim) for chunk in chunks])
    X_torch = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_torch)
        return torch.argmax(logits.sum(dim=0)).item()

# Основная часть кода
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

author_files = ["Fry.txt", "Genri.txt", "Simak.txt", "Bulgakov.txt", "Bradbury.txt", "Strugatskie.txt"]
path_to_files = "texts/"
full_paths = [os.path.join(path_to_files, filename) for filename in author_files]
texts, labels, label2author = prepare_author_chunks(full_paths, CHUNK_SIZE, STEP_SIZE)
num_classes = len(label2author)

print("Всего чанков (фрагментов) после нарезки:", len(texts))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=VAL_SIZE, random_state=42, stratify=labels)
print(f"Train chunks: {len(train_texts)}, Val chunks: {len(val_texts)}")

# Обучение модели FastText на всем корпусе чанков
tokenized_texts = [chunk.lower().split() for chunk in texts]
ft_model = FastText(sentences=tokenized_texts,
                    vector_size=EMBEDDING_DIM,
                    window=5,
                    min_count=1,
                    workers=4)

# Преобразование текстов в векторное представление посредством усреднения FastText векторов
X_train = np.array([compute_average_vector(text, ft_model, EMBEDDING_DIM) for text in train_texts])
X_val = np.array([compute_average_vector(text, ft_model, EMBEDDING_DIM) for text in val_texts])

train_dataset = TextDataset(X_train, train_labels)
val_dataset = TextDataset(X_val, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = EMBEDDING_DIM  # размер входного слоя соответствует размерности FastText векторов

# При создании модели теперь передаём два параметра для скрытых слоёв, например: hidden_dim1=45 и hidden_dim2=30
model = MLP(input_dim, hidden_dim1=93, hidden_dim2=65, output_dim=num_classes, dropout_rate=0.2).to(device)

print("\n--- Обучение модели ---")
model = train_model(model, train_loader, val_loader, EPOCHS, LR, device)

# Оценка на валидационной выборке
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        preds = torch.argmax(model(X), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y.cpu().numpy())

val_acc = accuracy_score(all_true, all_preds)
print("\n=== Итоговая оценка на валидационной выборке ===")
print(f"Val Accuracy: {val_acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_true, all_preds))
print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=[label2author[i] for i in range(num_classes)]))

# Классификация 21 тестового отрывка на уже обученной модели
df = pd.read_csv("author_classification.csv")
test_true = []
test_pred = []
test_files_dir = "texts/"

print("\n--- Классификация тестовых файлов (на уже обученной модели) ---")
for _, row in df.iterrows():
    fname = os.path.join(test_files_dir, row['filename'])
    true_author = row['author']
    
    if not os.path.exists(fname):
        print(f"Файл {fname} не найден, пропуск...")
        continue
    
    text = read_file(fname)
    pred_label = predict_text(model, ft_model, text, EMBEDDING_DIM, chunk_size=CHUNK_SIZE, step=STEP_SIZE, device=device)
    pred_author = label2author[pred_label]
    
    test_true.append(true_author)
    test_pred.append(pred_author)
    
    print(f"Файл {fname} -> предсказанный автор: {pred_author} (истинный: {true_author})")

print("\n=== Оценка на 21 тестовом отрывке ===")
print("Accuracy:", accuracy_score(test_true, test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(test_true, test_pred, labels=list(label2author.values())))
print("\nClassification Report:")
print(classification_report(test_true, test_pred, labels=list(label2author.values())))
