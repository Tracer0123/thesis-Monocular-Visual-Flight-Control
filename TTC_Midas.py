"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""

# Import dependencies
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
# from itertools import count
from matplotlib.animation import FuncAnimation

# Download the MiDaS
# torch.hub.help("intel-isl/MiDaS", "MiDaS_medium", force_reload=True)  # Triggers fresh download of MiDaS repo
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small') # MiDaS_small  or DPT_Hybrid
midas.to('cpu')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

def create_depth(frame):
    
    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    output = []
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()
    return output


# using depth info to get a depth distance
# Developing the midas map
def TTC_map(img, prev_map, frame_time):
    map = create_depth(img)

    # Processing the depth map for usability
    depth_min = map.min()
    depth_max = map.max()
    normalized_depth = 255 * (map -depth_min) / (depth_max - depth_min)
    normalized_depth *= 3
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) /3
    right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)
    gray_map = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)

    # sectioning the depth map 
    height, width = gray_map.shape
    roi_width = width// 18
    roi_height = height//18
    remap = []
    rois = [(x, y, x + roi_width, y + roi_height) for x in range(0, width, roi_width) for y in range(0, height, roi_height)][:324]

    # generating median mini depth map
    for roi in rois:
        x1, y1, x2, y2 = roi
        # Ensure the coordinates stay within the image bounds
        x2 = min(x2, width)
        y2 = min(y2, height)
        
        # Calculate median of the ROI and transformation to relative depth
        roi_median = np.median(gray_map[y1:y2, x1:x2])
        remap.append(roi_median)
    shaping = np.array(remap).reshape(18,18)
    shaping = (shaping*(-0.078) + 20) # transforming gray image to a relative distance map of max distance of 20meters
    if len(prev_map) == 0:
        return -1, shaping
    else:
        TTC = ttc_with_depthmap(shaping, prev_map, frame_time)
        return TTC, shaping
        

def ttc_with_depthmap(depth, prev_depth, time):
    TTC = []
    if np.array(depth).shape == np.array(prev_depth).shape:
        height, width = depth.shape
    else:
        return
    roi_width = width// 3
    roi_height = height//3
    rois_new = [(x, y, x + roi_width, y + roi_height) for x in range(0, width, roi_width) for y in range(0, height, roi_height)][:9]
        # generating median mini depth map
    for roi in rois_new:
        x1, y1, x2, y2 = roi
        # Ensure the coordinates stay within the image bounds
        x2 = min(x2, width)
        y2 = min(y2, height)

        # Calculating the change
        roi_max = np.max(depth[y1:y2, x1:x2] - prev_depth[y1:y2, x1:x2])
        roi_min = np.min(depth[y1:y2, x1:x2] - prev_depth[y1:y2, x1:x2])
        roi_diff = np.array(depth[y1:y2, x1:x2] - prev_depth[y1:y2, x1:x2])
        roi_mid = np.median(roi_diff.flatten())

        positn_max = np.where((depth[y1:y2, x1:x2] - prev_depth[y1:y2, x1:x2]) == roi_max)
        positn_min = np.where((depth[y1:y2, x1:x2] - prev_depth[y1:y2, x1:x2]) == roi_min)
        positn_mid = np.where(roi_diff == roi_mid)

        max_depth = depth[positn_max][0]
        min_depth = depth[positn_min][0]
        mid_depth = np.median(depth[y1:y2, x1:x2])

        if roi_max == 0 and roi_min == 0:
            TTC.append(-1) 
        # if abs(roi_min) > abs(roi_max):
        #     if roi_min < 0:
        #         TTC.append(round(time*min_depth, 3))
        # else:
        #     TTC.append(round((time*abs(max_depth)/abs(roi_max)), 3))
        else:TTC.append(round((time*abs(min_depth)/abs(roi_min)), 3))

    return TTC
    


def create_side_by_side(image, depth):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth -depth_min) / (depth_max - depth_min)
    normalized_depth *= 3
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) /3
    right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)
    return np.concatenate((image, right_side), axis=1)


commands = {
            0: "move_forward",
            1: "move_back",
            2: "move_up",
            3: "move_down",
            4: "move_left",
            5: "move_right",
            6: "rotate_5",
            7: "rotate_-5",
            8: "rotate_180"
            }

def decision_sys(result1, result2, result3, drone):

    # Value holder
    rate_of_change = []
    for a,b,c in zip(result1,result2, result3):
        diff_ab = abs(a-b)
        diff_bc = abs(b-c)
        if a< 10 and c<=3 and diff_bc <= 1.5:
            rate_of_change.append(diff_bc)
        else:
            rate_of_change.append(1000)
        # minimum value of defined range to for sides
    # min_first_3 = min(rate_of_change[:3])
    # min_4 = rate_of_change[3]
    # min_5_to_6 = min(rate_of_change[4:6])
    # min_7_to_9 = min(rate_of_change[6:])
    if len(rate_of_change) > 0:
        minimum = min(rate_of_change)
        rate_of_change = np.array(rate_of_change)
        # positn = np.where(rate_of_change[:] == minimum)
        positn = 30 # place holder
        for i in range(0,len(rate_of_change)):
            if rate_of_change[i] == minimum:
                positn = i
        # print(positn)
        # determination of drift direction or stop
        drift_stop = [0,0,0,0] # drift value arrangement: right, left, up, down
        if positn < 3:
            drift_stop[1] = 1
            # drone.rotate_clockwise(-5)
            return commands[7]
        elif positn == 3:
            drift_stop[3] = 1
            # drone.move_down(10)
            # drone.send_rc_control(0,0,-10,0)
            return commands[3]
            time.sleep(1.1)
        elif positn > 3 and positn<6:
            drift_stop[2] = 1
            # drone.move_up(10)
            # drone.send_rc_control(0,0,10,0)
            return commands[2]
            time.sleep(1.1)
        elif positn > 5 and positn < 10:
            drift_stop[0] = 1
            # drone.rotate_clockwise(5)
            return commands[6]

        

# def track_changes_and_find_minima(data):
#     """
#     Track the rate of change for each of the 9 values across three sets and find the minimum
#     of specific ranges.

#     :param data: A list of three lists, each containing 9 float values.
#     :return: A list containing the minimum values for the specified ranges.
#     """

#     if len(data) != 3 or any(len(row) != 9 for row in data):
#         raise ValueError("Data must consist of 3 sets of 9 values each.")

#     # Calculate the rate of change for each value across the sets
#     rate_of_change = [(data[2][i] - data[0][i]) / 2 for i in range(9)]

#     # Find the minimum values in the specified ranges
#     min_first_3 = min(rate_of_change[:3])
#     min_4_to_6 = min(rate_of_change[3:6])
#     min_7_to_9 = min(rate_of_change[6:])

#     return [min_first_3, min_4_to_6, min_7_to_9]

# # Example data: Three sets of 9 float values each
# example_data = [
#     [1.5, 2.3, 3.7, 4.8, 5.2, 6.3, 7.4, 8.6, 9.8],
#     [1.6, 2.5, 3.9, 4.7, 5.3, 6.1, 7.5, 8.7, 9.9],
#     [1.7, 2.7, 4.0, 4.6, 5.4, 6.0, 7.6, 8.8, 10.0]
# ]

# # Calculate and return the values
# track_changes_and_find_minima(example_data)
