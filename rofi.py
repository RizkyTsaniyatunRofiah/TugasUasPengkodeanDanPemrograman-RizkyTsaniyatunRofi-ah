import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Membaca data dari file CSV
df = pd.read_csv('data_penjualan.csv')
print(df.head())
# Memeriksa data yang hilang
print(df.isnull().sum())

# Memeriksa duplikasi
print(df.duplicated().sum())

# Mengoreksi tipe data
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
# Menambahkan kolom baru Total Penjualan
df['Total Penjualan'] = df['Jumlah'] * df['Harga Satuan']
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# Statistik dasar
print(df.describe())

# Distribusi jenis kelamin
sns.countplot(data=df, x='Jenis Kelamin')
plt.show()

# Distribusi jenis barang
sns.countplot(data=df, x='Jenis Barang')
plt.show()

# Total penjualan berdasarkan jenis barang
sns.barplot(data=df, x='Jenis Barang', y='Total Penjualan')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Memilih fitur dan target
X = df[['Jumlah', 'Harga Satuan']]
y = df['Total Penjualan']

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
model = LinearRegression()
model.fit(X_train, y_train)
# Prediksi menggunakan model
y_pred = model.predict(X_test)

# Mengukur kinerja model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R2 Score: {r2}')
# Menyajikan hasil dalam bentuk visualisasi
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
# Deploy model (contoh sederhana dengan menyimpan model menggunakan joblib)
import joblib

# Menyimpan model
joblib.dump(model, 'model_penjualan.pkl')

# Memantau model (simulasi dengan prediksi data baru)
data_baru = [[3, 200000]]  # Contoh data baru
prediksi_baru = model
