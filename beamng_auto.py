import csv
import time
import os
import math
from math import sqrt
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU


# ---------------------------------------------------------------------
# FUNGSI SIMPAN DATA
# ---------------------------------------------------------------------
def save_data_to_csv(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = ['time', 'accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"✔ Data IMU berhasil disimpan: {filepath}")


# ---------------------------------------------------------------------
# SETTING UTAMA
# ---------------------------------------------------------------------
SIMULATOR_PATH = 'D:\\BeamNG\\BeamNG.tech.v0.32.5.0\\'
BNG_USER = "C:\\Users\\Brenda\\AppData\\Local\\BeamNG.drive"

CRASH_THRESHOLD_G = 5.0
RECORD_DURATION_AFTER_CRASH = 3
TRIAL_COUNT = 10   # jumlah sample per kelas


# ---------------------------------------------------------------------
# DEFINISI VELOCITY PER KELAS (KM/H)
# ---------------------------------------------------------------------
SPEED_MAP = {
    "minor": (20, 30),
    "normal": (40, 60),
    "severe": (70, 100)
}


# ---------------------------------------------------------------------
# PROGRAM UTAMA
# ---------------------------------------------------------------------
def main():
    set_up_simple_logging()
    beamng = BeamNGpy('localhost', 64256, home=SIMULATOR_PATH, user=BNG_USER)

    # posisi awal EGO
    ego_pos = (-661, 157, 118)
    ego_rot_quat = (0, 0, 0.3826834, 0.9238795)

    # rotasi → sudut
    ego_angle_rad = 2 * math.atan2(ego_rot_quat[2], ego_rot_quat[3])
    fwd_x = math.cos(ego_angle_rad)
    fwd_y = math.sin(ego_angle_rad)

    distance_m = 60  # jarak antar kendaraan

    # posisi kendaraan lain di depan EGO
    other_pos = (
        ego_pos[0] - fwd_x * distance_m,
        ego_pos[1] - fwd_y * distance_m,
        ego_pos[2]
    )

    # rotasi OTHER = berhadapan 180°
    other_angle_rad = ego_angle_rad + math.pi
    other_rot_quat = (
        0,
        0,
        math.sin(other_angle_rad / 2),
        math.cos(other_angle_rad / 2)
    )

    # ---------------------------------------------------------
    # DEFINISI 6 KELAS
    # ---------------------------------------------------------
    variations = [
        ("menabrak_minor",     "menabrak", SPEED_MAP["minor"]),
        ("menabrak_normal",    "menabrak", SPEED_MAP["normal"]),
        ("menabrak_severe",    "menabrak", SPEED_MAP["severe"]),
        ("ditabrak_minor",     "ditabrak", SPEED_MAP["minor"]),
        ("ditabrak_normal",    "ditabrak", SPEED_MAP["normal"]),
        ("ditabrak_severe",    "ditabrak", SPEED_MAP["severe"])
    ]

    # ---------------------------------------------------------
    # START SIMULATOR
    # ---------------------------------------------------------
    bng = beamng.open(launch=True)

    try:
        for class_name, mode, speed_range in variations:

            for trial in range(1, TRIAL_COUNT + 1):

                print(f"\n=== {class_name} | trial {trial} ===")

                scenario = Scenario('west_coast_usa', f'{class_name}_{trial}')

                # spawn mobil
                ego_vehicle = Vehicle('ego_vehicle', model='etk800', licence='EGO')
                other_vehicle = Vehicle('other_vehicle', model='etk800', licence='OTHER')

                scenario.add_vehicle(ego_vehicle, pos=ego_pos, rot_quat=ego_rot_quat)
                scenario.add_vehicle(other_vehicle, pos=other_pos, rot_quat=other_rot_quat)

                scenario.make(bng)
                bng.scenario.load(scenario)
                bng.scenario.start()

                imu = AdvancedIMU('imu', bng, ego_vehicle, is_send_immediately=True)

                speed_kph = np.random.uniform(speed_range[0], speed_range[1])
                speed_ms = speed_kph / 3.6

                is_crashed = False
                crash_time = None
                imu_buffer = []

                # -----------------------------------------------------
                # LOOP SIMULASI
                # -----------------------------------------------------
                while True:
                    bng.step(1)

                    readings = imu.poll()
                    if readings['time'] is None:
                        continue

                    t = readings['time']
                    accel = readings['accRaw']
                    gyro = readings['angVel']
                    imu_buffer.append([t] + accel + gyro)

                    # --- KONTROL KENDARAAN ---
                    if mode == "menabrak":
                        ego_vehicle.control(throttle=1.0, steering=0)
                        other_vehicle.control(throttle=0, steering=0)

                    elif mode == "ditabrak":
                        ego_vehicle.control(throttle=0, steering=0)
                        other_vehicle.control(throttle=1.0, steering=0)

                    # --- DETEKSI TABRAKAN ---
                    g = sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2) / 9.81

                    if not is_crashed and g > CRASH_THRESHOLD_G:
                        print(f"  Tabrakan terdeteksi: {g:.2f} G")
                        is_crashed = True
                        crash_time = t

                    if is_crashed and (t - crash_time > RECORD_DURATION_AFTER_CRASH):
                        break

                # save file
                filename = f"data/{class_name}_trial_{str(trial).zfill(2)}.csv"
                save_data_to_csv(filename, imu_buffer)

                imu.remove()
                bng.scenario.stop()

    finally:
        bng.close()


if __name__ == '__main__':
    main()
