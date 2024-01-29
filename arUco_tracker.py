"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""

import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import time as time
import math
from djitellopy import tello
drone = tello.Tello()
# Camera Data
cam_mat = np.array([[912.64927388, 0.00000, 489.32738527], [0.00000, 914.2160340, 369.59634003], [0.00000, 0.00000, 1.00000]])
dist_coef = np.array([[-1.90833014e-02, -1.59402727e-01, 2.06866280e-03, 1.84387869e-04, 7.18877408e-01]])

# Initializing Variables

MARKER_SIZE = 187  # centimeters
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters()
marker_start = 0
marker_target = 1

def arUco_tracking(gray_frame):
    corners, IDs, rejects = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)
    rVec, tVec, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
    markers = IDs
    if rVec is not None and tVec is not None:
        a,b = corners[0][0][2]
        cv.drawFrameAxes(gray_frame, cam_mat, dist_coef, rVec[0], tVec[0], 40, 4)
        cv.putText(
            gray_frame,
            f"x:{round(tVec[0][0][0],1)} y: {round(tVec[0][0][1],1)}  Z: {round(tVec[0][0][2],1)}",
            (int(a),int(b)),
            cv.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        markers = IDs[0][0]
        ty = tVec[0][0][1]
        tz = tVec[0][0][2]
        print(markers)
        theta_degrees = int(math.degrees(math.atan(ty / tz)))
        # print(theta_degrees)
        image = aruco.drawDetectedMarkers(gray_frame, corners, IDs, (0,250,0))

    image = gray_frame # aruco.drawDetectedMarkers(gray_frame, corners, IDs, (0,250,0))
    cv.imshow("frame", image)
    if rVec is None:
        print("rVec or tVec is None, cannot draw frame axes.")
        return -1
    if markers is not None:
        markers = IDs[0][0]
        ty = tVec[0][0][1]
        tz = tVec[0][0][2]
        print(markers)
        theta_degrees = int(math.degrees(math.atan(ty / tz)))
        displt = tVec[0][0]
        out = [displt, markers, theta_degrees]
        # print(out)
        return out
    else:
        return -1
    
    # Movement decision is taken from here based on infromation from the tracking

commands = {
            0: "move_forward",
            1: "move_back",
            2: "move_up",
            3: "move_down",
            4: "move_left",
            5: "move_right",
            6: "rotate_10",
            7: "rotate_180",
            8: "rotate_by_",
            9: "move_forward_by_"
            }
def movement(drone, gray, thresholds, start_marker, target_marker, result0):
    if gray is not None:
        threshold_XY, threshold_Z, threshold_theta = thresholds
        result , status = result0
        if result is None: 
            time.sleep(3)

        elif result == -1:
            # drone.rotate_clockwise(15)
            return commands[6]
        else:
            positn, Id, theta = result
            print(theta)
            print(180+theta)
            if Id == start_marker:
                x, y, z = positn
                # drift to align the drone to the seen axis
                # drone.rotate_clockwise(180+theta)
                out = commands[8] + str(180+theta)
                return out 
                # print(posi)
            if Id == target_marker and status == True:
                x, y, z = positn
                z = int(z/10)
                print(z)
                if z > threshold_Z:
                    # drone.move_forward(50)
                    out = commands[0]
                    return out
                else:
                    # time.sleep(3)
                    # drone.flip_back()
                    # drone.land()
                    return "land"
            else: return commands[6]


'''
Complex drone movement


def movement(drone, gray, thresholds, start_marker, target_marker, result0):
    if gray is not None:
        threshold_XY, threshold_Z, threshold_theta = thresholds
        result , status = result0
        if result is None: 
            time.sleep(3)

        elif result == -1:
            drone.rotate_clockwise(15)
        else:
            positn, Id, theta = result
            # print(result)
            #print(positn)
            if Id == start_marker:
                x, y, z = positn
                # drift to align the drone to the seen axis
                if abs(x) > threshold_XY:  # Define your own threshold
                    if x < -threshold_XY:
                        # drone.move_left(int(abs(5))) # using x will go all the way we want small shifts and inference hence 5cm
                        drone.go_xyz_speed(-5,0,0,20)
                    else:
                        # drone.move_right(int(abs(5)))
                        drone.go_xyz_speed(5,0,0,20)

                elif abs(y) > threshold_XY:
                    if y < -threshold_XY:
                        drone.move_down(int(abs(5)))
                    else:
                        drone.move_up(int(abs(5)))

                elif abs(theta)> threshold_theta: # rotate drone to aline with z axis
                    if theta > 0:
                        drone.rotate_clockwise(5)
                    else:
                        drone.rotate_clockwise(-5)
                else:
                    if z < 100:
                        drone.move_back(10)
                        
                    drone.rotate_clockwise(180)
            elif Id == target_marker and status:
                x, y, z = positn
                if abs(x) > threshold_XY:  # Define your own threshold
                    if x < -threshold_XY:
                        drone.move_left(int(abs(5)))
                    else:
                        drone.move_right(int(abs(5)))

                elif abs(y) > threshold_XY:
                    if y < -threshold_XY:
                        drone.move_down(int(abs(5)))
                    else:
                        drone.move_up(int(abs(5)))

                elif abs(theta)> threshold_theta: # rotate drone to aline with z axis
                    if theta > 0:
                        drone.rotate_clockwise(5)
                    else:
                        drone.rotate_clockwise(-5)

                elif abs(z) > threshold_Z:
                    if z < -threshold_Z:
                        drone.move_backward(int(abs(10)))
                    else:
                        drone.move_forward(int(abs(10)))

                else:
                    time.sleep(5)
                    drone.flip_back()
                    drone.land()

'''


# cap = cv.VideoCapture(1, cv.CAP_DSHOW)

# cam_on = True
# while cam_on == True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     arUco_tracking(frame)

#     key = cv.waitKey(10)
#     if key == 27:
#         cam_on = False
    
# cap.release()
# cv.destroyAllWindows()