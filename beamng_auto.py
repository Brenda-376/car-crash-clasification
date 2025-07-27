import csv
import time
import os
import math
from math import sqrt
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU

# --- FUNGSI UNTUK MENYIMPAN DATA ---
def save_data_to_csv(filepath, data):
    """Menyimpan data sensor ke file CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = ['time', 'accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Data berhasil disimpan di: {filepath}")

# --- KONFIGURASI UTAMA ---
SIMULATOR_PATH = 'D:\\BeamNG\\BeamNG.tech.v0.32.5.0\\'
BNG_USER = "C:\\Users\\Brenda\\AppData\\Local\\BeamNG.drive"
CRASH_THRESHOLD_G = 20.0
RECORD_DURATION_AFTER_CRASH = 3

def main():
    set_up_simple_logging()
    beamng = BeamNGpy('localhost', 64256, home=SIMULATOR_PATH, user=BNG_USER)
    
    variations = [
        ('tabrakan_frontal', 70, {'pos': (-660.4, 157.6, 118), 'rot_quat': (0, 0, 0.9238795, -0.3826834)}),
        # ('tabrakan_samping_kiri', 60, {'pos': (5, -50, 100), 'rot_quat': (0, 0, 1, 0)}),
    ]

    bng = beamng.open(launch=True)
    try:
        for name, speed_kph, pos2 in variations:
            for trial in range(1, 16):
                print(f"--- Menjalankan: {name} | Kecepatan: {speed_kph} kph | Percobaan: {trial} ---")
                
                scenario = Scenario('west_coast_usa', f'{name}_trial_{trial}')
                
                # Di v1.29, Vehicle() langsung membuat objek yang bisa digunakan
                ego_vehicle = Vehicle('ego_vehicle', model='van', licence='BEAMNG', colour='Yellow')
                other_vehicle = Vehicle('other_vehicle', model='van', licence='BEAMNG', colour='Red')
                
                # Posisi awal mobil
                ego_pos = (-717, 101, 118)
                other_pos = pos2['pos']
                
                scenario.add_vehicle(ego_vehicle, pos=ego_pos, rot_quat=(0, 0, 0.3826834, 0.9238795))
                scenario.add_vehicle(other_vehicle, pos=other_pos, rot_quat=pos2['rot_quat'])
                
                scenario.make(bng)
                bng.scenario.load(scenario)
                bng.scenario.start()
                
                for _ in range(60):
                    bng.step(1)

                # Buat instance dari kelas AdvancedIMU secara langsung
                imu = AdvancedIMU('ego_imu', bng, ego_vehicle, is_send_immediately=True)
                
                # 2. Berikan perintah ke mobil EGO
                ego_vehicle.ai_set_mode('span')
                ego_vehicle.ai_set_target(ego_pos, 'position')
                ego_vehicle.ai_set_speed(speed_kph / 3.6, 'speed')
                
                # 3. Berikan perintah ke mobil LAIN
                other_vehicle.ai_set_mode('span')
                other_vehicle.ai_set_target(other_pos, 'position')
                other_vehicle.ai_set_speed(speed_kph / 3.6, 'speed')
                
                is_crashed = False
                crash_time = None
                sensor_data = []

                while True:
                    bng.step(1)
                    
                    readings = imu.poll()
                    
                    if readings['time'] is None: continue

                    current_time = readings['time']
                    accel = readings['accRaw']
                    gyro = readings['angVel']
                    
                    # Gabungkan data untuk disimpan
                    row = [current_time] + accel + gyro
                    sensor_data.append(row)
                    
                    g_force = sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2) / 9.81
                    if not is_crashed and g_force > CRASH_THRESHOLD_G:
                        print(f"Tabrakan terdeteksi pada {g_force:.2f} G!")
                        is_crashed = True
                        crash_time = current_time

                    if is_crashed and (current_time - crash_time > RECORD_DURATION_AFTER_CRASH):
                        break
                    
                    if current_time > 60: 
                        print("Timeout, tidak ada tabrakan terdeteksi.")
                        break

                if is_crashed:
                    filename = f'data/{name}_{speed_kph}kph_trial_{str(trial).zfill(2)}.csv'
                    save_data_to_csv(filename, sensor_data)
                
                # --- ADDED: Hapus sensor setelah selesai ---
                imu.remove()

    finally:
        # Pastikan koneksi ditutup meskipun terjadi error
        bng.close()

if __name__ == '__main__':
    main()