import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

CHUNK_SIZE = 500
STEP_SIZE = 200
EPOCHS = 30            # Полное число эпох для финального обучения
EPOCHS_OPT = 10        # Число эпох при оптимизации
BATCH_SIZE = 64
VAL_SIZE = 0.2

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

class TextDataset(Dataset):
    def __init__(self, X_vectors, y_labels):
        if hasattr(X_vectors, 'toarray'):
            X_vectors = X_vectors.toarray()
        self.X = torch.tensor(X_vectors, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=6, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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
        print(f"Epoch [{epoch}/{epochs}]: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    if best_state_dict:
        model.load_state_dict(best_state_dict)
    return model

def predict_text(model, vectorizer, text, chunk_size=200, step=200, device='cpu'):
    model.eval()
    chunks = chunk_text(text, chunk_size, step)
    if not chunks:
        return 0
    X = vectorizer.transform(chunks)
    X_torch = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_torch)
        return torch.argmax(logits.sum(dim=0)).item()

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

vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), analyzer='word', token_pattern=r'\w+')
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

train_dataset = TextDataset(X_train, train_labels)
val_dataset = TextDataset(X_val, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[1]


# Функция для байесовской оптимизации
from sklearn.model_selection import StratifiedKFold

def objective(params):
    # Извлечение гиперпараметров
    lr = params['lr']
    hidden_dim = int(params['hidden_dim'])
    dropout_rate = params['dropout_rate']
    
    print(f"\nОптимизируем: lr={lr:.5f}, hidden_dim={hidden_dim}, dropout_rate={dropout_rate:.3f}")
    
    # Параметры k-fold валидации
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    
    # Преобразуем train_labels в numpy массив
    labels_arr = np.array(train_labels)
    
    # Перебор фолдов
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_arr)), labels_arr), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        # Выборка подфолдов для обучения и валидации
        X_tr = X_train[train_idx]
        y_tr = labels_arr[train_idx]
        X_val_cv = X_train[val_idx]
        y_val_cv = labels_arr[val_idx]
        
        # Создание поддатасетов и DataLoader-ов
        train_dataset_cv = TextDataset(X_tr, y_tr)
        val_dataset_cv = TextDataset(X_val_cv, y_val_cv)
        train_loader_cv = DataLoader(train_dataset_cv, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_cv = DataLoader(val_dataset_cv, batch_size=BATCH_SIZE, shuffle=False)
        
        # Инициализация модели с данными гиперпараметрами
        model = MLP(input_dim, hidden_dim=hidden_dim, output_dim=num_classes, dropout_rate=dropout_rate).to(device)
        
        # Обучение модели на текущем фолде
        model = train_model(model, train_loader_cv, val_loader_cv, epochs=EPOCHS_OPT, lr=lr, device=device)
        
        # Оценка точности на фолд-валидации
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader_cv:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        fold_acc = accuracy_score(all_targets, all_preds)
        print(f"Точность для фолда {fold}: {fold_acc:.4f}")
        fold_accuracies.append(fold_acc)
    
    # Усреднение точности по всем фолдам
    avg_acc = np.mean(fold_accuracies)
    loss = 1 - avg_acc  # минимизируем 1 - accuracy
    print(f"\nСредняя точность по {n_splits} фолдам: {avg_acc:.4f}, Loss: {loss:.4f}")
    
    return {'loss': loss, 'status': STATUS_OK}


# Пространство поиска гиперпараметров
space = {
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),           # лог-равномерное распределение
    'hidden_dim': hp.quniform('hidden_dim', 10, 100, 1),               # целые значения от 10 до 100
    'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7)              # равномерное распределение от 0.2 до 0.7
}

# Запуск оптимизации
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
print("\nОптимальные гиперпараметры (в логарифмическом виде для lr и float для остальных):")
print(best)


# Финальное обучение с оптимальными гиперпараметрами
best_hidden_dim = int(best['hidden_dim'])
best_lr = best['lr']
best_dropout = best['dropout_rate']

print("\nФинальное обучение с найденными гиперпараметрами:")
final_model = MLP(input_dim, hidden_dim=best_hidden_dim, output_dim=num_classes, dropout_rate=best_dropout).to(device)
final_model = train_model(final_model, train_loader, val_loader, epochs=EPOCHS, lr=best_lr, device=device)

# Финальная оценка
final_model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        preds = torch.argmax(final_model(X), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y.cpu().numpy())

val_acc = accuracy_score(all_true, all_preds)
print("\n=== Итоговая оценка на валидационной выборке ===")
print(f"Val Accuracy: {val_acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_true, all_preds))
print("\nClassification Report:")
from sklearn.metrics import classification_report
print(classification_report(all_true, all_preds, target_names=[label2author[i] for i in range(num_classes)]))
