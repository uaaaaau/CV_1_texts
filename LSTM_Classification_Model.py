import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter

# ПАРАМЕТРЫ
CHUNK_SIZE = 500    # максимальная длина чанка в токенах (в LSTM надо фиксировать input_length)
STEP_SIZE = 200     # шаг при нарезке
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 64
VAL_SIZE = 0.2      # доля валидационной выборки
MAX_VOCAB_SIZE = 10000  # ограничение на размер словаря
EMBED_DIM = 100     # размер эмбеддинга
HIDDEN_DIM = 128    # скрытый размер LSTM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ФУНКЦИИ ДЛЯ РАБОТЫ С ТЕКСТОМ
def read_file(filepath):
    """Считывает текст из файла."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=200, step=200):
    """
    Нарезает текст на куски (чанки), каждый длиной chunk_size (в токенах).
    Сдвигается на step токенов при переходе к следующему куску.
    """
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
    """
    Для каждого файла автора:
        - Считывает текст
        - Нарезает на чанки
        - Возвращает списки (texts, labels) + label2author
    """
    texts, labels, label2author = [], [], {}
    for label_idx, filepath in enumerate(author_files):
        author_name = os.path.splitext(os.path.basename(filepath))[0]
        label2author[label_idx] = author_name
        chunks = chunk_text(read_file(filepath), chunk_size, step)
        texts.extend(chunks)
        labels.extend([label_idx] * len(chunks))
    return texts, labels, label2author

# ПОСТРОЕНИЕ СЛОВАРЯ
def build_vocab(texts, max_size=10000):
    """
    Собирает словарь из списка текстов (каждый текст — это строка).
    max_size — максимальный размер словаря (редкие слова будут заменяться на <UNK>).
    """
    freq = Counter()
    for t in texts:
        tokens = t.split()
        freq.update(tokens)
    # Самые частотные слова
    most_common = freq.most_common(max_size - 2)  # -2, чтобы оставить место для <PAD> и <UNK>
    
    # Строим word2idx
    # Зарезервируем индексы:
    #   0 -> <PAD> (паддинг)
    #   1 -> <UNK> (неизвестное слово)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, _ in most_common:
        word2idx[word] = idx
        idx += 1
    return word2idx

def text_to_indices(text, word2idx, chunk_size):
    """
    Преобразует строку (текст чанка) в список индексов фиксированной длины (chunk_size).
    Если токенов меньше chunk_size — паддим, если больше — обрезаем.
    """
    tokens = text.split()
    indices = []
    for token in tokens:
        if token in word2idx:
            indices.append(word2idx[token])
        else:
            indices.append(word2idx["<UNK>"])
    # Теперь обрезаем/паддим до длины chunk_size
    if len(indices) > chunk_size:
        indices = indices[:chunk_size]
    else:
        indices += [word2idx["<PAD>"]] * (chunk_size - len(indices))
    return indices

# DATASET И ДАТАЛОУДЕР 
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, chunk_size=200):
        """
        texts — список строк (чанков),
        labels — список целых меток (одинаковой длины с texts),
        word2idx — словарь слово->индекс
        chunk_size — какая длина будет у каждого примера после паддинга
        """
        self.chunk_size = chunk_size
        self.word2idx = word2idx
        self.labels = labels
        self.text_indices = []
        
        for t in texts:
            idxs = text_to_indices(t, self.word2idx, self.chunk_size)
            self.text_indices.append(idxs)

        self.X = torch.tensor(self.text_indices, dtype=torch.long)
        self.y = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ОПРЕДЕЛЕНИЕ МОДЕЛИ (LSTM)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        # Состояния lstm
        output, (h_n, c_n) = self.lstm(embedded)  
        last_hidden = h_n[-1]  
        logits = self.fc(last_hidden)
        return logits

# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, epochs+1):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Оценка на валидации
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss_val = criterion(val_outputs, y_val)
                val_loss += loss_val.item()
                preds = torch.argmax(val_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_val.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        print(f"Epoch [{epoch}/{epochs}]: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # Сохраняем "лучшую" версию модели (по val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # Восстанавливаем лучшую модель
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


# ФУНКЦИИ ПРЕДСКАЗАНИЯ НА ОДНОМ ТЕКСТЕ
def predict_text(model, text, word2idx, chunk_size=500, step_size=200, device='cpu'):
    """
    Нарезает входной текст на чанки, для каждого чанка считает логиты,
    суммирует их, и берёт самый вероятный класс.
    """
    model.eval()
    chunks = chunk_text(text, chunk_size, step_size)
    
    if not chunks:
        return 0  # fallback если вдруг текст слишком короткий
    
    sum_logits = None
    with torch.no_grad():
        for ch in chunks:
            # Преобразовать чанк в индексы
            idxs = text_to_indices(ch, word2idx, chunk_size)
            # Превращаем в батч размером 1
            X_torch = torch.tensor([idxs], dtype=torch.long).to(device)
            logits = model(X_torch)  # [1, num_classes]
            
            if sum_logits is None:
                sum_logits = logits
            else:
                sum_logits += logits
    
    # sum_logits: [1, num_classes]
    pred_class = torch.argmax(sum_logits, dim=1).item()
    return pred_class

# ОСНОВНОЙ КОД 
if __name__ == "__main__":
    # Исходные тренировочные файлы (каждый файл - тексты одного автора)
    author_files = ["Fry.txt", "Genri.txt", "Simak.txt",
                    "Bulgakov.txt", "Bradbury.txt", "Strugatskie.txt"]
    path_to_files = "texts/"
    full_paths = [os.path.join(path_to_files, fname) for fname in author_files]

    # Подготовка обучающих чанков
    texts, labels, label2author = prepare_author_chunks(full_paths,
                                                        chunk_size=CHUNK_SIZE,
                                                        step=STEP_SIZE)
    num_classes = len(label2author)
    print("Всего чанков (фрагментов) после нарезки:", len(texts))

    # Разделим на train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=VAL_SIZE, random_state=42, stratify=labels)

    print(f"Train chunks: {len(train_texts)}, Val chunks: {len(val_texts)}")

    # Строим словарь только на тренировочных текстах
    word2idx = build_vocab(train_texts, max_size=MAX_VOCAB_SIZE)
    vocab_size = len(word2idx)
    print("Размер словаря:", vocab_size)

    # Датасеты и даталоадеры
    train_dataset = TextDataset(train_texts, train_labels, word2idx, chunk_size=CHUNK_SIZE)
    val_dataset = TextDataset(val_texts, val_labels, word2idx, chunk_size=CHUNK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Инициализация модели
    model = LSTMClassifier(vocab_size=vocab_size,
                           embed_dim=EMBED_DIM,
                           hidden_dim=HIDDEN_DIM,
                           num_classes=num_classes,
                           pad_idx=word2idx["<PAD>"],
                           num_layers=1).to(device)

    print("\n--- Обучение модели (LSTM) ---")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

    # Оценка на валидации
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    val_acc = accuracy_score(all_true, all_preds)
    print("\n=== Итоговая оценка на валидационной выборке ===")
    print(f"Val Accuracy: {val_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=[label2author[i] for i in range(num_classes)]))

    # Теперь классифицируем 21 тестовый отрывок (csv-файл) 
    test_files_dir = "texts/"
    csv_path = "author_classification.csv"

    if not os.path.exists(csv_path):
        print(f"\nФайл {csv_path} не найден! Тестовая классификация пропущена.")
    else:
        df = pd.read_csv(csv_path)
        
        test_true = []
        test_pred = []
        print("\n--- Классификация тестовых файлов (LSTM) ---")
        for _, row in df.iterrows():
            fname = os.path.join(test_files_dir, row['filename'])
            true_author = row['author']
            
            if not os.path.exists(fname):
                print(f"Файл {fname} не найден, пропуск...")
                continue

            text = read_file(fname)
            pred_label = predict_text(model, text, word2idx,
                                      chunk_size=CHUNK_SIZE,
                                      step_size=STEP_SIZE,
                                      device=device)
            pred_author = label2author[pred_label]
            
            test_true.append(true_author)
            test_pred.append(pred_author)
            
            print(f"Файл {fname} -> предсказанный автор: {pred_author} (истинный: {true_author})")

        print("\n=== Оценка на 21 тестовом отрывке ===")
        print("Accuracy:", accuracy_score(test_true, test_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(test_true, test_pred, labels=list(label2author.values())))
        print("\nClassification Report:")
        print(classification_report(test_true, test_pred,
                                    labels=list(label2author.values())))

