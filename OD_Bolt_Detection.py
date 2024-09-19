# import torch
#
# print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
#
# # Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(cuda_id)
#
# print(f"ID of current CUDA device:{torch.cuda.current_device()}")
#
# print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

from ultralytics import YOLO
import cv2
import numpy as np
import math
import datetime
import os
import cvzone
import serial
import time
import sqlite3
import sys
import csv
from pathlib import Path

model = YOLO('D:\Kuliah\PKKM\Dashboard\YOLO\Bolt_Det\V2\Weights_100\Model_1.pt')
model.to('cuda')
last_time_saved = time.time()
mask = cv2.imread('D:\Kuliah\PKKM\Dashboard\YOLO\Bolt_Det\Skripsi_Picture\MaskingArea340x320.png')
# contours = ([(150, 80), (490, 80), (150, 400), (490, 400)])
start_p = (150, 80)
end_p = (490, 400)

def canny_edge_detection(frame): # For Canny Edge Display a.k.a Anime Display, Not Used
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)
    v = np.median(frame)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma)* v))
    edges = cv2.Canny(blurred, lower, upper)

    return blurred, edges

def activate_relay(port, state): # For Activate the relay through Arduino Serial Port
    ser = serial.Serial(port, 9600, timeout=10)
    ser.write(state)
    data = ser.readline().decode('utf-8', 'ignore')
    if data:
        return data

def read_serial(port): # Reading the Arduino Serial port for Button Usage, Not Used
    read_ser = serial.Serial(port, 9600, timeout=10)
    data_ser = read_ser.readline().decode('utf-8', 'ignore')
    if data_ser:
        return data_ser

def create_folder(main_dir): # Create and directing to Local Image Saving
    date_now = datetime.datetime.now().strftime("%Y_%m_%d")
    new_folder_path = os.path.join(main_dir, date_now)

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder for today's detection: {new_folder_path}")
    else:
        print(f"Folder for today's already exists: {new_folder_path}")

    return new_folder_path


# def run_yolov8_and_capture_output(yolov8_command):
#     process = subprocess.Popen(yolov8_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     output, _ = process.communicate()
#     return output


# def save_output_to_csv(output, csv_file_path):
#     lines = output.strip().split('\n')
#
#     header = lines[0].split(',')
#
#     with open(csv_file_path, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(header)
#
#         for line in lines[1:]:
#             row = line.split(',')
#             csv_writer.writerow(row)

def create_txt():
    date = datetime.datetime.now()
    c_month = date.strftime("%Y_%m")
    folder_path = os.path.join('D:\Kuliah\PKKM\Dashboard\YOLO\Bolt_Det\hasil', c_month)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder for {folder_path} created.")
    else:
        print(f"Folder for {folder_path} already exists.")

    file_name = date.strftime("%Y_%m_%d") + ".txt"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        sys.stdout = open(file_path, 'w')
    else:
        print(f"File {file_name} already exists.")
    return file_path

def save_output_to_txt():
    txt_file_path = create_txt()
    sys.stdout.close()
    with open(txt_file_path, 'r') as f:
        print(f.read())
    return txt_file_path

def save_img(frame, boxes, conn, cursor, count_b, count_n, Desc, SerNum, speed): # For Saving G and NG Images
    global last_time_saved
    global main_dirr
    global command
    if Desc == 'Not-Good':
        main_dirr = 'D:\Kuliah\PKKM\Dashboard\YOLO\Bolt_Det\data_webcam\Defect'
        command = "INSERT INTO NG_Frame (Timestamp, Image, Bolt_Inserted, Bolt_Uninserted, SN, Description, Speed) VALUES (?, ?, ?, ?, ?, ?, ?)"
    elif Desc == 'Good':
        main_dirr = 'D:\Kuliah\PKKM\Dashboard\YOLO\Bolt_Det\data_webcam\Good'
        command = "INSERT INTO OK_Frame (Timestamp, Image, Bolt_Inserted, Bolt_Uninserted, SN, Description, Speed) VALUES (?, ?, ?, ?, ?, ?, ?)"
    folder_path = create_folder(main_dir=main_dirr)
    curr_time = time.time()
    current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = os.path.join(folder_path, f"defect_{current_time}.jpg")
    img_copy = frame.copy()
    # for i, box in enumerate(boxes):
    #     a, b, c, d = map(int, box.xyxy[0])
    #     cv2.rectangle(img_copy, (a, b), (c, d), (255, 0, 255), 3)
    time_diff = curr_time - last_time_saved
    if time_diff >= 10:
        cv2.imwrite(output_path, img_copy)
        last_time_saved = curr_time
        print(f"Image saved on local: {output_path}")
        # Save image to db as blob
        img_blob = cv2.imencode('.jpg', img_copy)[1].tobytes()
        # command = "INSERT INTO defect_detected (Timestamp, Images, Bolt_Inserted, No_Bolt, SN, Description) VALUES (?, ?, ?, ?, ?, ?)" #Old ones
        data_tuple = (current_time, img_blob, count_b, count_n, SerNum, Desc, speed)
        cursor.execute(command, data_tuple)
        conn.commit()
        print(f"Image saved to database at {current_time}")
    # output_path = create_folder(main_dir=main_dirr)
    # date_noww = datetime.datetime.now().strftime("%Y_%m_%d")
    # csv_file_name = f"output_{date_noww}.csv"
    # csv_file_path = os.path.join(output_path, csv_file_name)
    # save_output_to_csv(output='', csv_file_path=csv_file_path)

    return output_path

def prediction_table(img, predictions): # Make a Prediction Table on Display, Not Used
    bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    table_x, table_y = 10, 10
    table_width, table_height = 300, 150
    # background
    cv2.rectangle(img, (table_x, table_y), (table_x + table_width, table_y + table_height), bg_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    line_height = 15

    for i, prediction in enumerate(predictions):
        text = f"{i + 1}. {prediction['class']}: {prediction['confidence']:.2f}"
        y_position = table_y + (i + 1) * line_height
        cv2.putText(img, text, (table_x + 10, y_position), font, font_scale, text_color, font_thickness)

def Text(img, predictions, color): # Create Text on Display
    # cv2.rectangle(img, (10, 10), (10 + 30, 10 + 30), (0, 0, 0), -1)
    cv2.putText(img, predictions, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def calculate_iou(box, boxes): #Calculate Interception over Union, Not Used
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.maximum(box[2], boxes[:, 2])
    y2 = np.maximum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3]) - box[1]
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = intersection / (area_box + area_boxes - intersection)
    return iou

def resize_to_fullscreen(image, w, h): #Resize Display Function
    screen_width, screen_height = w, h  # Gantilah dengan ukuran monitor sebenarnya

    img_height, img_width = image.shape[:2]

    scale_factor_width = screen_width / img_width
    scale_factor_height = screen_height / img_height
    scale_factor = min(scale_factor_width, scale_factor_height)

    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    return resized_image

def main():
    total_bolt = 3
    global speed_inference
    # sys.stdout = open('D:\Kuliah\PKKM\Dashboard\YOLO\Bolt_Det\Results_Terminal.txt', 'w') Gausa dipake
    # Saving Inference Time
    # txt = create_txt()
    # sys.stdout = open(txt, 'w')

    # Camera Error Handling
    try:
        cap = cv2.VideoCapture(0) # 1 or 0 for external or internal camera
        if not cap.isOpened():
            raise Exception("Could not open camera. Check the connection!")
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    # Predicting...
    while cap.isOpened():
        success, img = cap.read()
        # Apply Mask to get ROI
        img = cv2.rectangle(img, start_p, end_p, (0, 0, 255), 2)
        imgRegion = cv2.bitwise_and(img, mask)
        # Predict
        result = model.predict(imgRegion, stream=True, device=0, conf=0.55, iou=0.7) # Change ImgRegion for Masking Usage
        if not success:
            break
        # Connecting to DB...
        conn = sqlite3.connect('D:\Kuliah\Semester 8\Skripsi\Database\Pengujian_Skripsi_3_v2_LEDON_d1.db', timeout=0)
        cursor = conn.cursor()
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS OK_Frame (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Timestamp datetime NOT NULL,
                        Image BLOB NOT NULL,
                        Bolt_Inserted INTEGER NOT NULL,
                        Bolt_Uninserted INTEGER NOT NULL,
                        SN TEXT,
                        Description TEXT,
                        Speed TEXT NOT NULL
                    )
                ''')
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS NG_Frame (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Timestamp datetime NOT NULL,
                        Image BLOB NOT NULL,
                        Bolt_Inserted INTEGER NOT NULL,
                        Bolt_Uninserted INTEGER NOT NULL,
                        SN TEXT,
                        Description TEXT,
                        Speed TEXT NOT NULL
                    )
                ''')
        conn.commit()

        class_counts_bolt = {}
        class_counts_nobolt = {}
        # Iterate through every pixels
        for r in result:
            boxes = r.boxes
            speed = r.speed
            speed_inference = speed.get('inference') # get inference using dict method
            # probs = r.probs
            # print('speed : ', speed_inference)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=5)

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])

                cvzone.putTextRect(
                    img, f'{model.names[cls]} {conf}', (max(0, x1), (max(25, y1))),
                    thickness=1, scale=0.5, offset=2,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    colorT=(255, 255, 255), colorR=(255, 0, 255), colorB=(0, 255, 0)
                )  # max box agar tetap kelihatan/tidak kepotong

                # Class Counting
                class_label = model.names[cls]
                label = "bolt" if model.names[cls] == 'bolt' else 'no-bolt'
                if label == 'bolt':
                    if class_label not in class_counts_bolt:
                        class_counts_bolt[class_label] = 1
                    else:
                        class_counts_bolt[class_label] += 1

                elif label == 'no-bolt':
                    if class_label not in class_counts_nobolt:
                        class_counts_nobolt[class_label] = 1
                    else:
                        class_counts_nobolt[class_label] += 1

        for class_label, count_bo in class_counts_bolt.items():
            print(f"{class_label}: {count_bo} bolt")
            # print(class_counts_bolt[class_label])
            if class_counts_bolt[class_label] == 3:
                print('Good')
                texts = "G"
                color = (124, 252, 0)
                Text(img, texts, color)
                Ket = 'Good'
                # print('Kamu masuk Good')
                save_img(img, boxes, conn, cursor, count_n=total_bolt-count_bo, count_b=int(count_bo),
                     Desc=Ket, SerNum='-', speed=speed_inference)

        for class_label, count_no in class_counts_nobolt.items():
            print(f"{class_label}: {count_no} hole no bolt")
            # print(class_counts_nobolt[class_label])
            print(class_label)
            if class_counts_nobolt[class_label] >= 1 and class_label == 'no-bolt':
                print('Not Good')
                texts = "NG"
                color = (0, 0, 255)
                Text(img, texts, color)
                # button_input = read_serial('COM5')
                Ket = 'Not-Good'
                save_img(img, boxes, conn, cursor, count_n=int(count_no), count_b=total_bolt-count_no,
                     Desc=Ket, SerNum='-', speed=speed_inference)
                activate_relay('COM10', state=b'1')  # Aktivasi Relay dan Delay nya dari Arduino atau dibawah

        # Fullscreen Resize
        fullscreen=resize_to_fullscreen(img, w=854, h=480)

        #Calculation for CannyEdge
        # blurred, edges = canny_edge_detection(fullscreen)

        cv2.imshow("Bolt-Detection", fullscreen)
        # cv2.imshow("YOLO-V8_Canny", edges)
        # cv2.imshow("YOLO-V8_Canny", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Saving to this day TXT
    sys.stdout.close()
    # with open(txt, 'r') as f:
    #     print(f.read())
    # Close all Function
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()


# Cache Save_Img
# for i, box in enumerate(boxes):
    #     a, b, c, d = map(int, box.xyxy[0])
    #     # print(a,'\n', b,'\n', c,'\n', d,'\n')
    #     a, b, c, d = max(0, a), max(0, b), min(img_copy.shape[1], c), min(img_copy.shape[0], d)
    #     cv2.rectangle(img_copy, (a, b), (c, d), (255, 0, 255), 3)
    #     cropped_image = frame[b:d, a:c]
    #     cv2.imwrite(output_path, cropped_image)