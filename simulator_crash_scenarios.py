import random
import csv
import time
import os
import keyboard
import numpy as np
import joblib
import pandas as pd
import telepot
from time import sleep
from tensorflow.keras.models import load_model
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU

BNG_HOME = "E:\\BeamNG\\BeamNG.tech.v0.32.5.0"
BNG_USER = "C:\\Users\\DELL\\AppData\\Local\\BeamNG.drive"

MODEL_PATH = "E:\\Pasca_Sarjana_PENS\\Tesis\\project\\motion_detection\\train_model\\model\\model_imu_cnn2d_rawData.h5"
SCALER_PATH = "E:\\Pasca_Sarjana_PENS\\Tesis\\project\\motion_detection\\train_model\\scalerIMU_cnn2d_rawData.save"

def high_precision_sleep(duration):
    start_time = time.perf_counter()
    while True:
        elapsed_time = time.perf_counter() - start_time
        remaining_time = duration - elapsed_time
        if remaining_time <= 0:
            break
        if remaining_time > 0.02:  # Sleep for 5ms if remaining time is greater
            time.sleep(max(remaining_time/2, 0.0001))  # Sleep for the remaining time or minimum sleep interval
        else:
            pass

def mode_info():
    print("SET THE MODE:")
    print("- Press [9] to start recording IMU data")
    print("- Press [8] to start prediction mode")
    print("- Press [0] to exit the program\n")

def generate_google_maps_link(latitude, longitude):
    base_url = "https://www.google.com/maps?q="
    return f"{base_url}{latitude},{longitude}"

def send_massage_telegram(massage):
    token = '6139373279:AAH8TxbgQ4ucicqxdbiHzKuaIqUz323--Cc' # telegram token
    receiver_id = 970998661 # https://api.telegram.org/bot<TOKEN>/getUpdates
    bot = telepot.Bot(token)
    return bot.sendMessage(receiver_id, massage) # send a activation message to telegram receiver id

def main():
    random.seed(1703)
    set_up_simple_logging()
    
    # Load model CNN
    model = load_model(MODEL_PATH)
    
    scaler = joblib.load(SCALER_PATH) 
    labels = ['Minor Crash', 'Normal', 'Severe Crash']  # Ganti dengan label kelas Anda sendiri    
    feature_names = ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']  # kolom data
    recording = False
    filename = 'imu_dataset.csv'
    buffer = []  # Buffer untuk menyimpan data IMU selama 3 detik
    FRAME_SIZE = 60  # 3 detik @ 20 Hz
    send_status = False
    
    beamng = BeamNGpy('localhost', 64256, home=BNG_HOME, user=BNG_USER)
    bng = beamng.open(launch=True)
    
    scenario = Scenario('west_coast_usa', 'advanced_IMU_demo', description='Advanced IMU sensor scenario')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='BEAMNG', color='Red')
    
    scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    # bng.ui.hide_hud()
    bng.scenario.start()
    
    IMU = AdvancedIMU('accelGyro', bng, vehicle, pos=(0.45,-0.6,0.68), is_send_immediately=True)
    
    mode_info()
    
    while True:
        event = keyboard.read_event() 
        if event.event_type == 'down':  # Only process key press (not release)
            if event.name == '8':  # 8 key for prediction mode
                print("Starting prediction mode...")
                while True:
                    high_precision_sleep(0.04)  # Sampling interval for 20 Hz
                    data = IMU.poll()
                    
                    accX = f"{data['accRaw'][0]:.5f}"
                    accY = f"{data['accRaw'][1]:.5f}"
                    accZ = f"{data['accRaw'][2]:.5f}"
                    gyrX = f"{data['angVel'][0]:.5f}"
                    gyrY = f"{data['angVel'][1]:.5f}"
                    gyrZ = f"{data['angVel'][2]:.5f}"
                    
                    buffer.append([accX, accY, accZ, gyrX, gyrY, gyrZ])
                    
                    if len(buffer) == FRAME_SIZE:
                        # Preprocessing data
                        frame_data = pd.DataFrame(buffer, columns=feature_names)
                        frame_data = scaler.transform(frame_data)  # Standarisasi data
                        frame_data = np.array(frame_data)
                        frame_data = frame_data.reshape(FRAME_SIZE, 6, 1)  # input shape for model
                        frame_data = np.expand_dims(frame_data, axis=0)
                        
                        # Predict of model
                        prediction = model.predict(frame_data)
                        predicted_label = labels[np.argmax(prediction)]
                        print("Predicted label:", predicted_label)
                        
                        # Hapus data dari buffer
                        buffer = []
                        
                        if ((predicted_label == 'Minor Crash' or predicted_label == 'Severe Crash') and send_status==False):
                            latitude = -7.276431713832524 
                            longitude = 112.7930945817668
                            link = generate_google_maps_link(latitude, longitude)
                            send_massage_telegram(f"CAR ACCIDENT DETECTED !\nClass: {predicted_label}\nLocation: {link}")
                            send_status = True
                        elif predicted_label == 'Normal' and send_status==True:
                            send_status = False
                    
                    if keyboard.is_pressed('0'):  # Exit prediction mode
                        print("Exiting prediction mode...")
                        mode_info()
                        break
            
            if event.name == '9':  # 9 key for recording mode
                recording = True
                print("CHOOSE THE CLASS")
                print("- Press [1] for NORMAL class")
                print("- Press [2] for MINOR CRASH class")
                print("- Press [3] for SEVERE CRASH class\n")
                while True:
                    index_class = input("Enter a class name: ")
                    if index_class == "1":
                        class_label = 'normal'
                        duration = 10
                        break
                    elif index_class == "2":
                        class_label = 'minor_crash'
                        duration = 3
                        break
                    elif index_class == "3":
                        class_label = 'severe_crash'
                        duration = 3
                        break
                    else:
                        print("That's not a valid the class name.")
                        
                skenario_counter = int(input("Initial set of scenario numbers: ")) # Counter for the scenario index
                while True:  # Loop to allow multiple recordings with the same class
                    if recording:
                        print("Starting data recording...")
                        # Open the file in append mode
                        is_new_file = not os.path.exists(filename)
                        file = open(filename, mode='a', newline='')
                        writer = csv.writer(file)
                        
                        # Write header only if the file is new
                        if is_new_file:
                            writer.writerow(['subject', 'class', 'scenario', 'timestamp', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'])
                        
                        print(f"Ready for simulation in 3 seconds...")
                        for i in range(3, 0, -1):
                            print(i)
                            sleep(1)
                        
                        for i in range(20*duration):  # Run the scenario (frequency sampling * duration)
                            high_precision_sleep(0.04)  # Sampling interval for 20 Hz
                            data = IMU.poll()
                            
                            timestamp = int(time.time() * 1e3)
                            accX = f"{data['accRaw'][0]:.5f}"
                            accY = f"{data['accRaw'][1]:.5f}"
                            accZ = f"{data['accRaw'][2]:.5f}"
                            gyrX = f"{data['angVel'][0]:.5f}"
                            gyrY = f"{data['angVel'][1]:.5f}"
                            gyrZ = f"{data['angVel'][2]:.5f}"
                            
                            writer.writerow(['subject1', class_label, skenario_counter, timestamp, accX, accY, accZ, gyrX, gyrY, gyrZ])
                            print(f"Recorded: {timestamp} - Acc: {accX}, {accY}, {accZ}; Gyro: {gyrX}, {gyrY}, {gyrZ}")
                        
                        print("Stopping recording...")
                        print(f"Normal {class_label} in scenario {skenario_counter} has completed !")
                        recording = False
                        file.close()
                        print(f"Data saved to {filename}")
                        
                        # Increment the scenario counter
                        skenario_counter += 1
                    
                    choice = input("Do you want to record more data with the same class? (y/n): ")
                    if choice.lower() == 'y':
                        recording = True
                        continue
                    elif choice.lower() == 'n':
                        print("Exiting recording mode...\n")
                        mode_info()
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
            
            elif event.name == '0':  # 0 key for exit mode
                print("Exiting program...")
                IMU.remove()
                bng.close()
                return

if __name__ == '__main__':
    main()
