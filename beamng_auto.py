import csv
import time
import os
import math
from math import sqrt
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU, Lidar

# --- FUNGSI UNTUK MENYIMPAN DATA IMU (Tidak berubah) ---
def save_data_to_csv(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = ['time', 'accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Data IMU berhasil disimpan di: {filepath}")

# --- CHANGED: Fungsi save_lidar_data_to_csv sekarang menerima satu NumPy array besar ---
def save_lidar_data_to_csv(filepath, final_lidar_array):
    """Menyimpan data streaming Lidar dari satu NumPy array besar ke file CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = ['time', 'x', 'y', 'z', 'r', 'g', 'b', 'a']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(final_lidar_array) # Langsung tulis NumPy array
    print(f"Data Lidar berhasil disimpan di: {filepath}")

# --- (Konfigurasi Utama tidak berubah) ---
SIMULATOR_PATH = 'D:\\BeamNG\\BeamNG.tech.v0.32.5.0\\'
BNG_USER = "C:\\Users\\Brenda\\AppData\\Local\\BeamNG.drive"
CRASH_THRESHOLD_G = 2.0
RECORD_DURATION_AFTER_CRASH = 3
TRIAL_COUNT = 15

def main():
    set_up_simple_logging()
    beamng = BeamNGpy('localhost', 64256, home=SIMULATOR_PATH, user=BNG_USER)
    
    # ... (Bagian Kalkulasi Posisi tidak berubah) ...
    ego_pos = (-661, 157, 90)
    ego_rot_quat = (0, 0, 0.3826834, 0.9238795)
    distance_m = 120
    speed_kph = 70
    ego_angle_rad = 2 * math.atan2(ego_rot_quat[2], ego_rot_quat[3])
    fwd_vec_x = math.cos(ego_angle_rad)
    fwd_vec_y = math.sin(ego_angle_rad)
    other_pos_x = ego_pos[0] - fwd_vec_x * distance_m
    other_pos_y = ego_pos[1] - fwd_vec_y * distance_m
    other_pos_z = ego_pos[2]
    other_pos = (other_pos_x, other_pos_y, other_pos_z)
    other_angle_rad = ego_angle_rad + math.pi
    half_angle = other_angle_rad / 2
    other_rot_quat = (0, 0, math.sin(half_angle), math.cos(half_angle))
    variations = [('tabrakan_frontal', speed_kph, {'pos': other_pos, 'rot_quat': other_rot_quat})]

    bng = beamng.open(launch=True)
    try:
        for name, _, pos2 in variations:
            for trial in range(1, TRIAL_COUNT + 1):
                print(f"--- Menjalankan: {name} | Percobaan: {trial} ---")
                
                # ... (setup scenario dan vehicle sama) ...
                scenario = Scenario('west_coast_usa', f'{name}_trial_{trial}')
                ego_vehicle = Vehicle('ego_vehicle', model='etkc', licence='EGO', colour='Silver')
                other_vehicle = Vehicle('other_vehicle', model='etkc', licence='OTHER', colour='Red')
                scenario.add_vehicle(ego_vehicle, pos=ego_pos, rot_quat=ego_rot_quat)
                scenario.add_vehicle(other_vehicle, pos=pos2['pos'], rot_quat=pos2['rot_quat'])
                scenario.make(bng)
                bng.scenario.load(scenario)
                bng.scenario.start()
                
                for _ in range(60):
                    bng.step(1)

                imu = AdvancedIMU('ego_imu', bng, ego_vehicle, is_send_immediately=True)
                lidar = Lidar('lidar_360', bng, ego_vehicle, is_360_mode=True, vertical_angle=26.9,
                              vertical_resolution=64, frequency=5, max_distance=120, is_streaming=True)
                
                is_crashed = False
                crash_time = None
                imu_sensor_data = []
                lidar_stream_data = [] # List ini sekarang akan berisi blok-blok NumPy array

                while True:
                    bng.step(1)
                    ego_vehicle.control(throttle=1.0, steering=0)
                    other_vehicle.control(throttle=1.0, steering=0)
                    
                    readings_imu = imu.poll()
                    if readings_imu['time'] is None: continue
                    current_time = readings_imu['time']
                    
                    # Proses data IMU
                    accel = readings_imu['accRaw']
                    gyro = readings_imu['angVel']
                    row_imu = [current_time] + accel + gyro
                    imu_sensor_data.append(row_imu)
                    
                    # Mengambil dan memproses data Lidar menggunakan NumPy ---
                    lidar_data = lidar.poll()
                    points = lidar_data['pointCloud'].reshape(-1, 3)
                    colors = lidar_data['colours'].reshape(-1, 4)
                    num_points = points.shape[0]
                    
                    # 1. Buat kolom timestamp
                    time_col = np.full((num_points, 1), current_time)
                    
                    # 2. Gabungkan semua array secara horizontal
                    combined_block = np.hstack((time_col, points, colors))
                    
                    # 3. Tambahkan blok data ke list
                    lidar_stream_data.append(combined_block)
                    # --------------------------------------------------------------------
                    
                    g_force = sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2) / 9.81
                    if not is_crashed and g_force > CRASH_THRESHOLD_G:
                        print(f"\nTabrakan terdeteksi pada {g_force:.2f} G!")
                        is_crashed = True
                        crash_time = current_time

                    if is_crashed and (current_time - crash_time > RECORD_DURATION_AFTER_CRASH):
                        break

                print()

                if is_crashed:
                    filename_imu = f'data/{name}_trial_{str(trial).zfill(2)}.csv'
                    save_data_to_csv(filename_imu, imu_sensor_data)
                    
                    # --- CHANGED: Gabungkan semua blok data Lidar sebelum menyimpan ---
                    if lidar_stream_data:
                        final_lidar_array = np.vstack(lidar_stream_data)
                        filename_lidar = f'data/{name}_trial_{str(trial).zfill(2)}_lidar.csv'
                        save_lidar_data_to_csv(filename_lidar, final_lidar_array)
                
                imu.remove()
                lidar.remove()
                bng.scenario.stop()

    finally:
        bng.close()

if __name__ == '__main__':
    main()

# import csv
# import time
# import os
# import math
# from math import sqrt
# from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
# from beamngpy.sensors import AdvancedIMU

# # --- FUNGSI UNTUK MENYIMPAN DATA ---
# def save_data_to_csv(filepath, data):
#     """Menyimpan data sensor ke file CSV."""
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
#     header = ['time', 'accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
#     with open(filepath, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         writer.writerows(data)
#     print(f"Data berhasil disimpan di: {filepath}")

# # --- KONFIGURASI UTAMA ---
# SIMULATOR_PATH = 'D:\\BeamNG\\BeamNG.tech.v0.32.5.0\\'
# BNG_USER = "C:\\Users\\Brenda\\AppData\\Local\\BeamNG.drive"
# CRASH_THRESHOLD_G = 10.0
# RECORD_DURATION_AFTER_CRASH = 3
# TRIAL_COUNT = 15

# def main():
#     set_up_simple_logging()
#     beamng = BeamNGpy('localhost', 64256, home=SIMULATOR_PATH, user=BNG_USER)
    
#     ego_pos = (-661, 157, 118)
#     ego_rot_quat = (0, 0, 0.3826834, 0.9238795) # Rotasi 45 derajat
    
#     # Definisikan parameter tabrakan
#     distance_m = 120
#     speed_kph = 70
    
#     # --- KALKULASI POSISI & ROTASI MOBIL TARGET ---
    
#     # 1. Hitung sudut rotasi ego dari quaternion
#     ego_angle_rad = 2 * math.atan2(ego_rot_quat[2], ego_rot_quat[3])
    
#     # 2. Tentukan vektor arah "depan" dari mobil ego
#     fwd_vec_x = math.cos(ego_angle_rad)
#     fwd_vec_y = math.sin(ego_angle_rad)
    
#     # 3. Hitung posisi mobil target
#     # Posisi Target = Posisi Ego + (Jarak * Vektor Arah)
#     other_pos_x = ego_pos[0] - fwd_vec_x * distance_m
#     other_pos_y = ego_pos[1] - fwd_vec_y * distance_m
#     other_pos_z = ego_pos[2] # Ketinggian sama
#     other_pos = (other_pos_x, other_pos_y, other_pos_z)
#     print(other_pos)
#     # 4. Hitung rotasi mobil target (berhadapan = rotasi ego + 180 derajat)
#     other_angle_rad = ego_angle_rad + math.pi # math.pi adalah 180 derajat
#     half_angle = other_angle_rad / 2
#     other_rot_quat = (0, 0, math.sin(half_angle), math.cos(half_angle))
    
#     # 5. Masukkan hasil kalkulasi ke dalam list 'variations'
#     variations = [
#         ('tabrakan_frontal', speed_kph, {'pos': other_pos, 'rot_quat': other_rot_quat}),
#         # ('tabrakan_samping_kiri', 60, {'pos': (5, -50, 100), 'rot_quat': (0, 0, 1, 0)}),
#     ]

#     bng = beamng.open(launch=True)
#     try:
#         for name, speed_kph, pos2 in variations:
#             for trial in range(1, TRIAL_COUNT):
#                 # print(f"--- Menjalankan: {name} | Kecepatan: {speed_kph} kph | Percobaan: {trial} ---")
#                 print(f"--- Menjalankan: {name} | Percobaan: {trial} ---")
                
#                 scenario = Scenario('west_coast_usa', f'{name}_trial_{trial}')
                
#                 ego_vehicle = Vehicle('ego_vehicle', model='van', licence='EGO', colour='Yellow')
#                 other_vehicle = Vehicle('other_vehicle', model='van', licence='OTHER', colour='Red')
                
#                 # Gunakan variabel yang sudah didefinisikan di atas
#                 scenario.add_vehicle(ego_vehicle, pos=ego_pos, rot_quat=ego_rot_quat)
#                 scenario.add_vehicle(other_vehicle, pos=pos2['pos'], rot_quat=pos2['rot_quat'])
                
#                 scenario.make(bng)
#                 bng.scenario.load(scenario)
#                 bng.scenario.start()
                
#                 for _ in range(60):
#                     bng.step(1)

#                 # Buat instance dari kelas AdvancedIMU secara langsung
#                 imu = AdvancedIMU('ego_imu', bng, ego_vehicle, is_send_immediately=True)
                
#                 # # 2. Berikan perintah ke mobil EGO
#                 # ego_vehicle.ai_set_mode('manual')
#                 # ego_vehicle.ai_set_speed(speed_kph / 3.6, 'speed')
                
#                 # # # 3. Berikan perintah ke mobil target
#                 # other_vehicle.ai_set_mode('manual')
#                 # other_vehicle.ai_set_speed(speed_kph / 3.6, 'speed')
                
#                 is_crashed = False
#                 crash_time = None
#                 sensor_data = []

#                 while True:
#                     bng.step(1)

#                     ego_vehicle.control(throttle=1.0, steering=0)
#                     other_vehicle.control(throttle=1.0, steering=0)
                    
#                     readings = imu.poll()
                    
#                     if readings['time'] is None: continue

#                     current_time = readings['time']
#                     accel = readings['accRaw']
#                     gyro = readings['angVel']
                    
#                     # Gabungkan data untuk disimpan
#                     row = [current_time] + accel + gyro
#                     sensor_data.append(row)
                    
#                     g_force = sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2) / 9.81
#                     if not is_crashed and g_force > CRASH_THRESHOLD_G:
#                         print(f"Tabrakan terdeteksi pada {g_force:.2f} G!")
#                         is_crashed = True
#                         crash_time = current_time

#                     if is_crashed and (current_time - crash_time > RECORD_DURATION_AFTER_CRASH):
#                         break
                    
#                     # if current_time > 60: 
#                     #     print("Timeout, tidak ada tabrakan terdeteksi.")
#                     #     break

#                 print()

#                 if is_crashed:
#                     # filename = f'data/{name}_{speed_kph}kph_trial_{str(trial).zfill(2)}.csv'
#                     filename = f'data/{name}_trial_{str(trial).zfill(2)}.csv'
#                     save_data_to_csv(filename, sensor_data)
                
#                 # --- Hapus sensor setelah selesai ---
#                 imu.remove()

#     finally:
#         # Pastikan koneksi ditutup meskipun terjadi error
#         bng.close()

# if __name__ == '__main__':
#     main()