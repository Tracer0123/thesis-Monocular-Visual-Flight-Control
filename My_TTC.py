"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""


# import dependencies 
import cv2
import numpy as np
from display import display
from pyr_lucas_kanade import lucas_pyramidal

    # Initialize featur parameters and Luca-Kanade parameters
feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# calculating focus of expansion
def calculate_foe_ransac(inter_pts, roi):

   
    x1, y1, x2, y2 = roi
    foe = 0
    if  len(inter_pts) < 1 :
        foe = 0
        return foe, 0
    
    else:

        # ransac algorithm
        num_points = 0
        positn_foe_roe = []
        section_area = []
        points_in_sections = []
        section_pt_len = []
        foe_roi_width = abs(x2-x1)//4
        foe_roi_height = abs(y2-y1)//4

        # Generate initial sub-sections
        foe_roi = [(x, y, x + foe_roi_width, y + foe_roi_height) 
                for x in range(x1, x2, foe_roi_width) 
                for y in range(y1, y2, foe_roi_height)][:16]

        # Analyze each section
        sections = []
        for row in range(3):
            for col in range(3):
                # Define the current section top-left corner
                sec_x1 = x1 + col * foe_roi_width
                sec_y1 = y1 + row * foe_roi_height
                
                # Make sure the section does not go out of bounds
                sec_x2 = min(sec_x1 + 2 * foe_roi_width, x2)
                sec_y2 = min(sec_y1 + 2 * foe_roi_height, y2)
                
                # Add the current section to the list
                sections.append((sec_x1, sec_y1, sec_x2, sec_y2))

        # Count points in each section using vectorized operations
        section_pt_len = [
            np.sum((inter_pts[:, 1] >= x1) & (inter_pts[:, 1] < x2) &
                (inter_pts[:, 0] >= y1) & (inter_pts[:, 0] < y2))
            for x1, y1, x2, y2 in sections
        ]

        # Find the section(s) with the maximum number of points
        max_pts = max(section_pt_len)
        max_sections = [sec for sec, pts in zip(sections, section_pt_len) if pts == max_pts]
        # print(max_pts)
        # print(max_sections)
        # Determine the best section based on error analysis
        min_error = float('inf')
        best_section = None
        best_foe = None

        for section in max_sections:
            x1, y1, x2, y2 = section
            foe_x = (x1 + x2) / 2
            foe_y = (y1 + y2) / 2
            foe = [foe_y, foe_x]

            section_points = inter_pts[
                (inter_pts[:, 1] >= x1) & (inter_pts[:, 1] < x2) &
                (inter_pts[:, 0] >= y1) & (inter_pts[:, 0] < y2)
            ]

            err = calculate_error(section_points, foe)
            if err < min_error:
                min_error = err
                best_section = section
                best_foe = foe

        return best_foe, best_section

            



def calculate_error(inter_points, center):

    section_median = None
    y0 = center[0]
    x0 = center[1] 
    def dist_btn_2_pts(point , x0, y0):
        y1, x1 = point
        return np.sqrt(((y1-y0)**2)+((x1-x0)**2))
    
    distances = []
    for x in inter_points:
        distances.append(dist_btn_2_pts(x, x0, y0))

    median = np.median(distances)
    mean = np.mean(distances)

    return median


def calculate_intersection(p1, p2):
    intersections = []
    if len(p1) == 0 or len(p2)== 0:
        return []

    line_data = np.array(list(map(create_line, p1, p2)))

    i = 0 
    while (i+1) < len(line_data):


        m1 = line_data[i][0]
        b1 = line_data[i][1]
        
        j = 1
        while j < len(line_data):
            m2 = line_data[j][0]
            b2 = line_data[j][1]
            if m1 == m2:
                j += 1
                continue # indicates parallel lines            
            if m1 == np.inf:
                x_inter = b1
                y_inter = (m2*x_inter) + b2
            elif m2 == np.inf:
                x_inter = b2
                y_inter = (m1*x_inter) + b1                
            else:
                x_inter = (b2 -b1)/ (m1-m2)
                y_inter = (m1*x_inter) +b1

            intersections.append([y_inter, x_inter])

            j += 1
        i += 1

    return intersections


def create_line(p1,p2):

    y1 = p1[0,0]
    x1 = p1[0,1]
    y2 = p2[0,0]
    x2 = p2[0,1]
    if x2-x1 == 0:
        return [np.inf, x1]
    else:
        m1 = (y2-y1)/(x2-x1) # slope of the line
        d1 = y2 - (m1*x2) # y intercept for the line

        return [m1, d1]



def calculate_ttc(foe, rel_vel , roi):

    ttc = None
    y = foe[0]
    x = foe[1]
    x1, y1, x2, y2 = roi
    edges = [[x1,y1],[x1,y2],[x2,y1],[x2,y2]]
    dis_to_edges = []
    for a in edges:
        dis_to_edges.append(np.sqrt(((a[0]-y)**2) + ((a[1]-x)**2)))

    dis = min(dis_to_edges)
    if rel_vel < 0.00001:
        rel_vel = 0.00001
    ttc = dis/ rel_vel

    return ttc

def optical_flow_roi(gray, prev_gray, prev_point):
    flow= []
    p1 = None
    '''
    the commented code bellow is for using cv2 version of optical flow,
    a costum version was used in the original code
    '''
    #p0 = prev_point
    # p0, p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point, None, **lk_params)
    
    p0, p1 = lucas_pyramidal(prev_gray, gray, 3, 3, 7)

    if p1 is not None:
        flow = np.array(p1) - np.array(p0)
    return flow, p1

def calculate_rel_vel(flow, p1, roi): # we work only with flow of points in the circular foe 
    rel_vel = None
    p1_mag = []
    x1, y1, x2, y2 = roi
    
    indices_x = np.where((p1[:, 0, 1] >= x1) & (p1[:, 0, 1] < x2) & (p1[:, 0, 0] >= y1) & (p1[:, 0, 0] < y2))
    if indices_x == None:
        return -1
    p1_mag = flow[indices_x]
    rel_vel = np.median(p1_mag)

    return rel_vel

def find_TTC(frame, prev_gray, prev_point):
    start = True
    while start:

        # preparing sections for analysis
        height, width = frame.shape
        # frame1 = frame
        roi_width = width // 3
        roi_height = height // 3
        # Create a list of ROIs
        rois = [(x, y, x + roi_width, y + roi_height) for x in range(0, width, roi_width) for y in range(0, height, roi_height)][:9]

        # Initialize prev_gray with the first frame
        if prev_point is None:

            prev_point = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
            return [], [], prev_point, []

        else:

            gray = frame
            # flow_ , p1_ = optical_flow_roi(gray, prev_gray, prev_point)

            p0_1, p1_1 = lucas_pyramidal(prev_gray,gray, 3, 3, 7)

            # Cleaning the flows
            # flow_mag = np.sqrt((flow_[:, 0, 0]**2) + (flow_[:, 0, 1]**2))
            flow_mag = np.abs((p1_1[:,0]-p0_1[:,0])**2 + (p1_1[:,1]-p0_1[:,1])**2)
            flow = np.array([])
            p1 = []
            p0 = []
            for i,j,k in zip(flow_mag, p1_1, p0_1):
                if i < roi_width/2:
                    flow = np.append(flow, i)
                    p1.append([[j[0], j[1]]]) # using open cv lkflow p1.append([[j[0,0], j[0,1]]])
                    p0.append([[k[0], k[1]]]) # using open cv lkflow p0.append([[k[0,0], k[0,1]]])
            prev_point = np.array(p0)
            p1 = np.array(p1)

            # Process each ROI sequentially
            sectionNum = 0
            TTC_all = []
            foe_all = []
            rel_vel_all = []

            points_of_inter = np.array([[0,0]])
            section_out_all = []
            for roi in rois:
                
                x1, y1, x2, y2 = roi
                indices_x = np.where((p1[:, 0, 1] >= x1) & (p1[:, 0, 1] < x2) & (p1[:, 0, 0] >= y1) & (p1[:, 0, 0] < y2))
                p1_roi = np.array(p1[indices_x])
                p0_roi = np.array(prev_point[indices_x])
                inter = np.array(calculate_intersection(p0_roi, p1_roi))

                if len(inter) > 0:

                    inter_indices = np.where((inter[:, 1] >= x1) & (inter[:, 1] < x2) & (inter[:, 0] >= y1) & (inter[:, 0] < y2))
                    # print(inter_indices)
                    inter_in_roi = inter[inter_indices]
                    points_of_inter = np.concatenate((points_of_inter, inter_in_roi), axis= 0)

                    foe, section_out = calculate_foe_ransac(inter_in_roi, roi)
                    rel_vel = calculate_rel_vel(flow, p1, roi)
                    if rel_vel == None:
                        rel_vel_all.append(-1)
                    else:
                        rel_vel_all.append(rel_vel)
                    if foe == 0:
                        ttc = 22222 # sets ttc to -2 when flow is 0
                        TTC_all.append(ttc)
                    
                    else:
                        if rel_vel == -1:
                            ttc = 11111 # sets ttc to -1 when flow is 0

                        else:
                            ttc = round(calculate_ttc(foe, rel_vel, roi), 4)
                            section_out_all.append(section_out)

                        TTC_all.append(ttc)
                        foe_all.append(foe)
                    

                else:
                    TTC_all.append(22222)
                    sectionNum += 1

            p0 = prev_point
            # prev_point = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
            points_of_inter1 = points_of_inter.tolist()

            # display(frame, rois, foe_all, TTC_all, p0, p1, points_of_inter1)

            return TTC_all, foe_all, prev_point, rel_vel_all
        
        start = False




           


