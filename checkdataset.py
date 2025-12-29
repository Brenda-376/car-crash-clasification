import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'dataset/menabrak_minor/trial_07.csv'
#'dataset/menabrak_severe/trial_03.csv'

df = pd.read_csv(CSV_PATH)

# 1. Durasi
duration = df['time'].iloc[-1] - df['time'].iloc[0]
print(f"Durasi data: {duration:.2f} detik")

# 2. Jumlah sampel
print("Jumlah sampel:", len(df))

# 3. Sampling rate
dt = df['time'].diff().dropna()
fs = 1 / dt.mean()
print(f"Sampling rate: {fs:.1f} Hz")

# 4. Plot cepat
plt.plot(df['time'], df['accelX'])
plt.xlabel('Time (s)')
plt.ylabel('Accel X')
plt.title('IMU accelX')
plt.show()
