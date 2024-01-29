"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""


from ultralytics import YOLO
import cv2
import torch
import math
import numpy as np

# Initializing the yolo model
yolo = YOLO('~/yolobot/src/yolobot_recognition/scripts/yolov8n.pt') #Import Yolov8 nano model
def yolo_box(img):
    box_positions = []
    ids_poistions = []
    result = yolo.track(
        img, #image to be evaluated
        # show= True,
        classes= 0, #only people class detected
        conf= 0.55, #Confidence score allowed
        tracker = "bytetrack.yaml" # the tracker used ### alternative tracker "botsort.yaml"
    )
    for r in result:

        boxes = r.boxes.cpu().numpy()

        if not r:
            return -1, -1
        else:
            for box in boxes:
                # get box coordinates in (top, left, bottom, right) format
                b = box.xyxy
                id = box.id
                box_positions.append([b])
                ids_poistions.append(id)

    return box_positions, ids_poistions

def distance_from_point_to_line_through_two_points(point, line):
    x0 , y0 = point
    x1, y1, x2, y2 = line
    """
    Calculate the distance from a point (x0, y0) to a line defined by two points (x1, y1) and (x2, y2).
    :param x0: x-coordinate of the point
    :param y0: y-coordinate of the point
    :param x1: x-coordinate of the first point on the line
    :param y1: y-coordinate of the first point on the line
    :param x2: x-coordinate of the second point on the line
    :param y2: y-coordinate of the second point on the line
    :return: Distance from the point to the line
    """
    # Calculate the slope of the line
    if x2 - x1 == 0:
        # Line is vertical
        return abs(x0 - x1)
    elif y2 - y1 == 0:
        # Line is Horizontal
        return abs(y0 - y1)
    else:
        m = (y2 - y1) / (x2 - x1)

        # Convert line equation to standard form: Ax + By + C = 0
        A = -m
        B = 1
        C = -y1 + m * x1

        # Calculate the distance using the formula
        distance = abs(A*x0 + B*y0 + C) / math.sqrt(A**2 + B**2)
        return distance


def Calculate_position(img):

    height, width, _ = img.shape
    center = [height/2, width/2]
    boxes, ids = yolo_box(img)
    # print(boxes)
    print(ids)
    positn = "outside"
    response = [0, 0, 0] 
    '''
    The response of the nature that is first value tells it the (0: not in range, 1: in range, -1: move backwards)
    closeness of the human figures and this is used to acess if the move infront or not
    we can also use the result from the previous midas to get 
    the relative distance and then the area to make this judgement
    - the second and third value are only usefull when the first value is 1
    the second value tells us to move up or down (1: up, -1: down)
    the third value tells us to move left or right (1: left, -1: right)
    if the go forward or sideways or how to advance
    '''
    if boxes == -1:
        return -1
    else:
        min_dist = -1
        positn_index = -1 
        '''
        min_dist: is the distance of the side or top or bottom from the center of the frame
        '''
        box_param = [0, 0]
        box_sides = []
        closest_width = []
        in_range_ids = []
        j = 0 # measures the boxes position in the box array which is of size
        for box, id in zip(boxes, ids):
            # print(box[0][0])
            x1,y1,x2,y2 = box[0][0]

            # Using the size of the box to decide to stop/regress/progress
            vert_side = abs(y1-y2)
            hori_side = abs(x1-x2)
            box_sides.append([[x1,y1,x1,y2],[x2,y1,x2,y2],[x1,y1,x2,y1],[x1,y2,x2,y2]]) # sides of the box [L,R,T,B]
            area = vert_side*hori_side
            j+=1
            # box_param.append([area, vert_side, hori_side])
            closest = x1
            if x2 < center[0]: closest = x2
            if (vert_side >= 0.9*height or hori_side >= 0.9*width or area >= 100000) and abs(center[0]-closest) < 0.6*width:
                response[0] = -1
                return response
            # this sees the different boxes that are in range and records the number, id and width size
            if (vert_side >= 0.7*height or hori_side >= 0.7*width or area >= 65000) and abs(center[0]-closest) < 0.45*width:
                box_param[0] += 1
                in_range_ids.append(id)
                closest_width.append(hori_side)
                box_param[1] =j 
                print(vert_side*hori_side)
        # Should sellect which width size is optimal to consider for it next motion and not just all boxes
        if box_param[0] > 1:
            largest = 0
            for d,j in zip(closest_width,range(0,len(closest_width))):
                if d > largest: 
                    largest = d
                    box_param[1] = j
                    box_param[0] = 1 # this forces the third if statment to run after selecting the best box to consider
        
        if box_param[0] == 0:
            return response
        
        if box_param[0] == 1:
            the_box_sides = box_sides[box_param[1]]
            min_dists = []
            
            for line in the_box_sides:
                min_dists.append(distance_from_point_to_line_through_two_points(center, line))
            min_dist = min(min_dists)

            ordered_dist = min_dists.copy()
            ordered_dist.sort()

            positn_index = np.where(min_dists[:] == min_dist)
            '''
            meaning of position index with reference to bounding boxes which is closses to the center
            0: vertical side to the left
            1: vertical side to the right
            2: horizontal side to the top
            3: horizontal side to the buttom

            '''

            # specifying the location of the box
            if center[0] <=  x2 and center[0]>= x1:
                positn = "inside"
                if positn_index[0] == 3:
                    ## return something that takes the drone upwards instead of down
                    posi = np.where(min_dists[:] == ordered_dist[2])
                    response[1] = 1
                    if posi[0] == 1:
                        response[2] = -1
                    if posi[0] == 0:
                        response[2] = 1

                    return response
                
                if positn_index == 2:

                    ## verify if upwards is closser than sidewards and move
                    response[1] = 1
                    return response
                if positn_index == 1:
                    ## move towards the shorter line and push it to the outer quadrants
                    response[2] = -1
                    return response
                if positn_index == 0:
                    ## Push the closes line away
                    response[2] = 1
                    return response
                else:
                    return
            else:
                if positn_index[0] == 3:
                    ## return something that takes the drone upwards instead of down
                    posi = np.where(min_dists[:] == ordered_dist[2])
                    response[1] = 1
                    if posi[0] == 1:
                        response[2] = 1
                    if posi[0] == 0:
                        response[2] = -1

                    return response
                
                if positn_index == 2:

                    ## verify if upwards is closser than sidewards and move
                    response[1] = 1
                    return response
                if positn_index == 1:
                    ## move towards the shorter line and push it to the outer quadrants
                    response[2] = 1
                    return response
                if positn_index == 0:
                    ## Push the closes line away
                    response[2] = -1
                    return response
                else:
                    return                
            #print(id)
commands = {
            0: "move_forward",
            1: "move_back",
            2: "move_up",
            3: "move_down",
            4: "move_left",
            5: "move_right",
            6: "rotate_10",
            7: "rotate_180"
            }

def yolo_movemtn(drone, directn):
    x,y,z = directn
    if z == -1:
        # drone.move_back(10)
        return commands[1]
    elif z == 1:
        if y == 1:
            # drone.move_up(10)
            return commands[2]
        elif y == -1:
            # drone.move_down(10)
            return commands[3]
        elif x == 1:
            # drone.move_left(10)
            return commands[4]
        elif x == -1:
            # drone.move_right(10)
            return commands[5]

# write something to notice change of state when the drone no longer sees
# a human so it can return to the original path  , move down or return to center


