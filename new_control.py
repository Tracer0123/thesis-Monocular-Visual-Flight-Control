"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""

from djitellopy import tello
from My_TTC import find_TTC 
#from arUco_tracking import arUco_tracking
from arUco_tracker import movement, arUco_tracking
from sklearn import preprocessing
import matplotlib.pyplot as plt
from TTC_Midas import TTC_map, decision_sys
from TTC_Midas import create_side_by_side
from YOLO import yolo_box, yolo_movemtn
from YOLO import Calculate_position
import os , sys
import cv2
import numpy as np
import time as time
from threading import Thread, Event
import random




if __name__ == "__main__":
    frame_counter = 1
    my_TTC_counter = 0
    TTC_midas_counter = -1
    result1 = []
    result2 = []
    result3 = []
    m_result1 = []
    m_result2 = []
    m_result3 = []
    people_trackx = []
    prev_points = None
    prev_depth = None
    prev_box = None 
    prev_gray = None
    feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    TTC = {"sec_1": [],
           "sec_2": [],
           "sec_3": [],
           "sec_4": [],
           "sec_5": [],
           "sec_6": [],
           "sec_7": [],
           "sec_8": [],
           "sec_9": []
           }
    mapTTC = {"sec_1": [],
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
    ttc = None
    foe_all = None
    prev_rel_vel_depth = []

    # ======================================= Choise of run ============================================
    '''
    decision for which algorithm to run

    '''
    human = False
    obstacle = False
    # ======================================= Using Tello ==============================================

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


    # ============================================= Dependencies and leading function =======================
    record = True
    def execute_command(command):
        print(command)
            
    gray_img = None
    cmd_pr = ""
    count = 0
    def img_processing(event):
        global count
        global gray_img
        global cmd_pr
        global pics
        global result1
        global result2
        global result3
        global my_TTC_counter
        human = False

        # ======================================== Using Tello =============================================

        while record:
            cmd = ""
            frame = drone.get_frame_read().frame
            save_image(i, frame)
            i += 1
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

        # ================================== Using files in a directory =====================================


        # path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames2"
        # dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
        # prev_gray = None
        # t1 = time.time()
        # t2 = time.time()
        # for _frame in dir:  # using the saved frames ( pictures)

        #     cmd = ""
        #     frame = cv2.imread( path + "/"+_frame ) # Colored image 

        # =================================== frame to Gray ==================================================

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
            gray = np.array(gray)
            gray_img = gray

        # ==================================== Checking for people ===========================================

        # Checking for people
            directn = 0
            if count%2 == -1 and count > 1:
                # time2 = time.time() - t2
                # t2 = time.time()
                boxes = yolo_box(frame)
                people_trackx = Calculate_position(frame) # format in z,y,x
                if people_trackx == -1:
                    boxes # place holder
                else:
                    human = True
                    directn = people_trackx[::-1] # reversing output to x,y,z 
                    cmd = yolo_movemtn(directn)
                #print(shw)
                    

        # =================================== Image Processing My TTC =========================================


            # getting the TTC and foe

            # processing the image for ttc, this is done so we can calculate the flow better
            #if count%2 == 0 and count > 2:
            if count > 2:
                if prev_gray is None:
                    prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
                    prev_gray = gray
                else:
                    ttc, foe_all, prev_points, vel = find_TTC(gray, prev_gray, prev_points)
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
                        drone = ""
                        cmd = decision_sys(result1, result2, result3, drone)
                    i = 0
                    while i < len(ttc):
                        index = "sec_" + str(i+1)
                        TTC[index].append(ttc[i])  
                        i+=1
                    while i < len(vel):
                        index = "sec_" + str(i+1)
                        Rel_Vel[index].append(vel[i])
                        i+=1    

            cmd_pr = cmd
            count += 1
        pics = False
        event.set()
        
    start_marker = 0
    target_marker = 1
    threshold_XY = 7
    thresholds = [9, 150, 5] # [XY, Z, Theta]
    status = False # is true only when the starter marker has been seen  
    ran_before = False
    record = True
    human = False ############ helps to decide which algorithm runs first
    obstacle = False

    positn = [] # OX, OY, OZ format


    # =================================== using Threads ==================================================
    event = Event()
    t = Thread(target= img_processing, args=(event, ))
    t.start()

    # =================================== arUco Processing  ==============================================
    run = True
    pics = True
    num = 0
    last_count = 0
    cmd_aruco = ""
    while run and pics:

        state = arUco_tracking(gray_img)
        result0 = []
        if state != -1:
            _, ID, _1 = state
            if ID == 1: ID_1 = 1
            if ID == 0 and status == False:
                status = True # this tells us if the drone already found the first marker and is in search for the second
        result0 = [state, status]
        if human == False : # runs only when there is no human and there is no obstacle in the frame
            cmd_aruco = movement(drone, gray_img, thresholds, start_marker, target_marker, result0)
            
            if cmd_aruco == "land":
                record = False
                run = False
                t.join()
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the video stream
            drone.streamoff()
            break


        # print(count)
        # if cv2.waitKey(10) & 0xFF == ord('q'): 
        #     cam_on = False
        #     break

        if cmd_pr == "":
            execute_command(cmd_aruco)
            print("\n")
        if cmd_pr != "" and count != last_count:
            execute_command(cmd_pr)
            cmd_pr = ""

        last_count = count
        # num += 1
    # cap.release()

    t.join()
    # print(mapTTC)

    fig, AXS = plt.subplots(3, 3)
    # plt.ylim(-5, 70)
    row = 0
    col = 0
    counter = 0
    for key in TTC:
        x_vals = np.arange(0,len(TTC[key]))
        y_vals = np.array(TTC[key])
        # normalized values
        # y_vals =  np.where(np.isnan(y_vals), -3, y_vals)
        # revert 22222 and 11111 exceptions to native -2 and -1
        y_vals = [-2 if value == 22222 else value for value in y_vals]
        y_vals = [-1 if value == 11111 else value for value in y_vals]
        # y_norm = (preprocessing.normalize([y_vals])*20).T
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