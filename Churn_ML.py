import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn import tree

df = pd.read_csv(r"C:\Users\hajdu\Documents\Rowanie\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_charge = round(df['TotalCharges'].median(), 2)
df['TotalCharges'] = df['TotalCharges'].fillna(median_charge)
df_1hot = pd.get_dummies(
    df,
    columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
               'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'],
    drop_first=True,
    dtype = int
)

X = df_1hot.drop(columns=['customerID', 'Churn_Yes']).values
y = df_1hot['Churn_Yes'].values

model_type = tree.DecisionTreeClassifier()
model = model_type.fit(X, y) 
feat_importances = pd.Series(model.feature_importances_, index=df_1hot.drop(columns=['customerID', 'Churn_Yes']).columns)
important_features = feat_importances[feat_importances > 0.01].index.tolist()

X = df_1hot[important_features]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

class ChurnModel(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super(ChurnModel, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.l3(x))
        return x

def train_and_eval(lr, hidden1, hidden2, epochs=200):
    model = ChurnModel(X_train_tensor.shape[1], hidden1, hidden2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_tensor)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor)
        test_loss = criterion(test_out, y_test_tensor).item()
        preds = (test_out > 0.5).float()
        acc = (preds == y_test_tensor).float().mean().item()
        
    return model, test_loss, acc



learning_rates = [0.001, 0.01, 0.1]
hidden_layers = [(16, 8), (32, 16), (64, 32)] 

results = []
best_acc = 0
best_model = None
best_params = None

print("Rozpoczęto analizę wrażliwości. Testowanie modeli...")
for lr in learning_rates:
    for h1, h2 in hidden_layers:
        m, val_loss, val_acc = train_and_eval(lr, h1, h2, epochs=250)
        results.append({'LR': lr, 'Hidden1': h1, 'Hidden2': h2, 'Val_Loss': val_loss, 'Val_Acc': val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = m
            best_params = (lr, h1, h2)

res_df = pd.DataFrame(results)

pivot_res = res_df.pivot(index='LR', columns='Hidden1', values='Val_Acc')
plt.figure(figsize=(8, 5))
sns.heatmap(pivot_res, annot=True, cmap='viridis', fmt=".4f")
plt.title('Dokładność na zbiorze walidacyjnym (LR vs Rozmiar pierwszej warstwy ukrytej)')
plt.ylabel('Learning Rate (Współczynnik uczenia)')
plt.xlabel('Rozmiar Hidden Layer 1')
plt.tight_layout()
plt.show()

print(f"\nNajlepsze parametry: Learning Rate = {best_params[0]}, Warstwy ukryte = {best_params[1]} i {best_params[2]}")

best_model.eval()
with torch.no_grad():
    y_pred_prob = best_model(X_test_tensor).numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
y_test_np = y_test_tensor.numpy()

print("\nRaport Klasyfikacji (Dla Najlepszego Modelu):")
print(classification_report(y_test_np, y_pred))

auc = roc_auc_score(y_test_np, y_pred_prob)
print(f"ROC-AUC Score: {auc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(y_test_np, y_pred_prob)
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'Model (AUC = {auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlabel('False Positive Rate (Odsetek fałszywie pozytywnych)')
axes[0].set_ylabel('True Positive Rate (Odsetek prawdziwie pozytywnych)')
axes[0].set_title('Krzywa ROC')
axes[0].legend(loc="lower right")

cm = confusion_matrix(y_test_np, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_xlabel('Przewidywana Klasa (Predicted)')
axes[1].set_ylabel('Prawdziwa Klasa (Actual)')
axes[1].set_title('Macierz Pomyłek (Confusion Matrix)')

plt.tight_layout()
plt.show()