"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""

from djitellopy import tello
from My_TTC import find_TTC, decision_sys
#from arUco_tracking import arUco_tracking
from arUco_tracker import movement, arUco_tracking
# from sklearn import preprocessing
import matplotlib.pyplot as plt
from YOLO import yolo_box, yolo_movemtn
from YOLO import Calculate_position
import os , sys
import cv2
import numpy as np
import time as time
from threading import Thread, Event
import random



# ========================== global dependencies ========================================================

human = False
obstacle = False
gray_img = None
thread_cmd = ""
counter = 0
end = False
drone = "" # which will later become a drone object
my_TTC = {"sec_1": [],
        "sec_2": [],
        "sec_3": [],
        "sec_4": [],
        "sec_5": [],
        "sec_6": [],
        "sec_7": [],
        "sec_8": [],
        "sec_9": []
        }
Rel_Vel = {"sec_1": [],
        "sec_2": [],
        "sec_3": [],
        "sec_4": [],
        "sec_5": [],
        "sec_6": [],
        "sec_7": [],
        "sec_8": [],
        "sec_9": []
        }
# ========================================== defining drone object ======================================

drone = tello.Tello()
drone.connect()
drone.set_speed(20)
print(f"Battery : {drone.get_battery()}% ")
drone.streamon()
drone.takeoff()
time.sleep(1)
# drone.move_up(75)  
drone.send_rc_control(0,0,20,0)
time.sleep(75/17)
time.sleep(1)
record = True
i = 0
def save_image(i,frame):
    cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/drone_recording5' + '/rec_' +str(i).zfill(4) + '.png', frame)
    cv2.imshow('results', frame)
    # time.sleep(1/20)

# ==================================== execute cmd function =============================================
def execute_command(command):
    global drone
    # Logic to execute commands on the Tello drone
    if command == "":
        time.sleep(.01)
    elif command == "takeoff":
        drone.takeoff()
        time.sleep(.2)
    elif command == "land":
        drone.land()
    elif command == "move_forward":
        # drone.move_forward(10)
        drone.send_rc_control(0,10,0,0)
        time.sleep(1.1)
        # time.sleep(.1)
    elif command == "move_back":
        # drone.move_back(10)
        drone.send_rc_control(0,-10,0,0)
        time.sleep(1.1)
    elif command == "move_up":
        # drone.move_up(10)
        drone.send_rc_control(0,0,10,0)
        time.sleep(1.1)
    elif command == "move_down":
        # drone.move_down(10)
        drone.send_rc_control(0,0,-10,0)
        time.sleep(1.1)
    elif command == "move_left":
        # drone.move_left(10)
        drone.send_rc_control(10,0,0,0)
        time.sleep(1.1)
    elif command == "move_right":
        # drone.move_right(10)
        drone.send_rc_control(-10,0,0,0)
        time.sleep(1.1)
    elif command == "rotate_10":
        drone.rotate_clockwise(10)
        time.sleep(.1)
    elif command == "rotate_180":
        drone.rotate_clockwise(180)
        time.sleep(.1)
    elif command[0:10] == "rotate_by_":
        drone.rotate_clockwise(int(command[10::]))
        time.sleep(.1)
    elif command[0:16] == "move_forward_by_":
        drone.rotate_clockwise(int(command[16::]))
        time.sleep(.1)


# placeholder
# def execute_command(command):
#     print(command)

#====================================== threading function ==============================================

def img_processing():
    # ===================================== calling global dependencies==================================
    global human
    global obstacle
    global gray_img
    global thread_cmd
    global counter
    global end
    global drone
    global my_TTC
    global Rel_Vel

    # ======================================== function dependencies ====================================

    result1 = []
    result2 = []
    result3 = []
    feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_points = None
    prev_gray = None
    my_TTC_counter = 0

    # ================================== Using files in a directory =====================================


    # path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames2"
    # dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
    # print(len(dir))
    # count = 0
    # for _frame in dir:  # using the saved frames ( pictures)

    #     cmd = ""
    #     frame = cv2.imread( path + "/"+_frame ) # Colored image 

    # ======================================= using tello ===============================================
    count = 0    
    while record:
        cmd = ""
        frame = drone.get_frame_read().frame
        save_image(count, frame)
        count += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    # ======================================= using WebCam ============================================= 

    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # webcam by defualt is 640x480
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 936) # best posible width resolutions 1236 , 612 these are in x(12) for this special case
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # best matching height resolutns 720 , 936

    # cam_on = True
    # while cam_on == True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    # =================================== frame to Gray ==================================================

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
        gray = np.array(gray)
        gray_img = gray


    # ==================================== Checking for people ===========================================

        # Checking for people
        directn = 0
        if count > 1:

            people_trackx = Calculate_position(frame) # format in z,y,x
            if people_trackx == -1:
                print("No objects to track") # place holder
            else:
                human = True
                directn = people_trackx[::-1] # reversing output to x,y,z 
                cmd = yolo_movemtn(directn)
            thread_cmd = cmd


    # =================================== Using my_TTC ===================================================
            
        if count > 1:
            if prev_gray is None:
                prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
                prev_gray = gray
            else:
                ttc, _ , prev_points, vel = find_TTC(gray, prev_gray, prev_points)
                my_TTC_counter += 1
                if my_TTC_counter == 1:
                    result1 = ttc
                if my_TTC_counter == 2:
                    result2 = ttc
                if my_TTC_counter == 3:
                    result3 = ttc  
                if my_TTC_counter == 4:
                    my_TTC_counter = 0     
                
                if human == False:
                    cmd = decision_sys(result1, result2, result3)
                    obstacle = True
                i = 0
                while i < len(ttc):
                    index = "sec_" + str(i+1)
                    my_TTC[index].append(ttc[i])  
                    i+=1
                while i < len(vel):
                    index = "sec_" + str(i+1)
                    Rel_Vel[index].append(vel[i])
                    i+=1   
            thread_cmd = cmd 

    # =================================== loop and fuction control =======================================
            
        human = False
        count += 1
        counter = count
    end = True


# ======================================== Threading =============================================
thread = Thread(target= img_processing)
thread.start()

# ======================================= main loop ==============================================

# ========================= Aruco Dependensis ==========================

status = False
result0 = []
start_marker = 0
target_marker = 1
threshold_XY = 7
thresholds = [9, 150, 5] # [XY, Z, Theta]
prev_counter = 0


#  arUco Processing

while (counter>0) and (end == False):
    state = arUco_tracking(gray_img)
    if state != -1:
        _, ID, _1 = state
        if ID == 0 and status == False:
            status = True # this tells us if the drone already found the first marker and is in search for the second
    result0 = [state, status]
    if human == False and  obstacle == False: # runs only when there is no human and there is no obstacle in the frame
        aruco = movement(gray_img, thresholds, start_marker, target_marker, result0)
        if aruco == "land":
            thread.join()
            end = True
    if (thread_cmd != "") and (prev_counter != counter):
        execute_command(thread_cmd)
    else:
        execute_command(aruco)


# demmo loop run
# run = True

# while run and not human:
    
#     if (thread_cmd != "") and prev_counter != counter:
#         print(thread_cmd)
#         prev_counter = counter
#         print(counter)
#     else:
#         a = random.randint(0, 10)
#         if a < 5:
#             print("lower")
#         else:
#             print("higher")
#         print(prev_counter)
#     if end:
#         thread.join()
#         run = False


# clossing the thread
        
thread.join()



# Moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# drawing the TTC on a graph

fig, AXS = plt.subplots(3, 3)
# plt.ylim(-5, 70)
row = 0
col = 0
counter = 0
window_size = 3
for key in my_TTC:
    x_vals = np.arange(0,len(my_TTC[key]))
    y_vals = np.array(my_TTC[key])
    # normalized values
    # y_vals =  np.where(np.isnan(y_vals), -3, y_vals)
    # revert 22222 and 11111 exceptions to native -2 and -1
    y_vals = [-2 if value == 22222 else value for value in y_vals]
    y_vals = [-1 if value == 11111 else value for value in y_vals]
    # y_norm = (preprocessing.normalize([y_vals])*20).T
    y_vals = moving_average(y_vals, window_size)
    x_vals = np.arange(0,len(y_vals))
    if row>= 3:
        col += 1
        row = 0
            
    AXS[row,col].plot(x_vals,y_vals)
    # AXS[row,col].plot(x_vals,y_norm)
    AXS[row,col].set_ylim(-5, 20)
    AXS[row,col].set_title("TTC of Section "+ str(counter+1) )
    row += 1
    counter += 1



plt.show()
# print(TTC)
cv2.destroyAllWindows()