#библиотеки

import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


#фиксируем случайность
torch.manual_seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")


X = train.drop(columns=["heartbeat_id", "prediction"])
y = train["prediction"]
X_test = test[X.columns]

print("Сигналов для обучения:", len(train))
print("Сигналов для предсказания:", len(test))

# разделение TRAIN и VAL


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#нормализация
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

#правильные классы переводим в целочисленные тензоры
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)


#D-CNN
pooled_length = X_train.shape[2] // 4

#создаём сеть последовательно
model = torch.nn.Sequential(
    # Первая свёртка ищет локальные особенности сигнала.
    torch.nn.Conv1d(1, 32, kernel_size=5, padding=2),
    torch.nn.ReLU(),

    torch.nn.MaxPool1d(2),

    torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
    torch.nn.ReLU(),
    torch.nn.MaxPool1d(2),

#превращаем признаки в один вектор
    torch.nn.Flatten(),
#Объединяем найденные признаки
    torch.nn.Linear(64 * pooled_length, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 5)
)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#обучение с ранней остановкой
batch_size = 32

best_loss = float("inf")

#количество эпох без улучшения
epochs_without_improvement = 0

best_weights = None

for epoch in range(100):
    model.train()
    indices = torch.randperm(len(X_train))
    total_loss = 0.0

#обрабатываем сигналы порциями по 32
    for start in range(0, len(X_train), batch_size):
        batch_indices = indices[start:start + batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        optimizer.zero_grad()
        outputs = model(X_batch)

        #сравниваем оценки с правильными классами
        loss = criterion(outputs, y_batch)

        #рассчитываем градиенты ошибки
        loss.backward()

        #обновляем параметры нейросети
        optimizer.step()

        #учитываем количество примеров в порции
        total_loss += loss.item() * len(X_batch)

    #Средняя ошибка на обучающей выборке
    train_loss = total_loss / len(X_train)

    #включаем режим проверки
    model.eval()

    #при проверке градиенты рассчитывать не нужно
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

        #выбираем класс с максимальной оценкой
        val_predictions = val_outputs.argmax(dim=1)

        val_accuracy = (
            (val_predictions == y_val).float().mean().item()
        )

    print(
        f"Эпоха {epoch + 1}/100 | "
        f"loss: {train_loss:.4f} | "
        f"val_loss: {val_loss:.4f} | "
        f"val_accuracy: {val_accuracy:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0

        best_weights = {}

        for name, value in model.state_dict().items():
            best_weights[name] = value.clone()

    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= 10:
        print("\nРанняя остановка: 10 эпох без улучшения.")
        break

#возвращаем веса с минимальной ошибкой
model.load_state_dict(best_weights)

#оценка качества

model.eval()

with torch.no_grad():
    #получаем оценки после восстановления лучших весов
    val_outputs = model(X_val)

    #выбираем классы и переводим их в массив для sklearn
    val_predictions = val_outputs.argmax(dim=1).numpy()

#правильные ответы также переводим в массив
y_val_array = y_val.numpy()

#выводим долю правильных ответов
print("\nAccuracy:", accuracy_score(y_val_array, val_predictions))

#выводим средний F1 по пяти классам
print(
    "F1-macro:",
    f1_score(
        y_val_array,
        val_predictions,
        labels=[0, 1, 2, 3, 4],
        average="macro",
        zero_division=0
    )
)

#строки — настоящие классы, столбцы — предсказанные
print("\nМатрица ошибок:")
print(
    confusion_matrix(
        y_val_array,
        val_predictions,
        labels=[0, 1, 2, 3, 4]
    )
)


with torch.no_grad():
    #получаем оценки классов для тестовых сигналов
    test_outputs = model(X_test)

    #выбираем класс с максимальной оценкой
    test_predictions = test_outputs.argmax(dim=1).numpy()

#создаём итоговую таблицу
result = pd.DataFrame({
    "heartbeat_id": test["heartbeat_id"],
    "prediction": test_predictions
})

#
assert len(result) == len(test), "Неверное количество строк"
assert not result.isna().any().any(), "Есть пропуски"
assert result["heartbeat_id"].is_unique, "Дублирование"
assert result["prediction"].between(0, 4).all(), "Неверные классы"

os.makedirs("outputs", exist_ok=True)
result.to_csv("outputs/task2_1_predictions.csv", index=False)
print(result.head())