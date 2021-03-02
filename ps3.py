"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import time
from scipy import ndimage


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    d = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
    return d


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    top_left = (0, 0)
    bottom_left = (0, image.shape[0] - 1)
    top_right = (image.shape[1]-1, 0)
    bottom_right = (image.shape[1] - 1, image.shape[0] - 1)

    return [top_left, bottom_left, top_right, bottom_right]
    

# def matchTemplates(listTemplates, image, method=cv2.TM_CCOEFF_NORMED, score_threshold=0.5, maxOverlap=0):
#     xOffset=yOffset=0
#     listHit = []
#     for tempTuple in listTemplates:
#         templateName, template = tempTuple[:2]
#         corrMap = cv2.matchTemplate(image, template, method)
#         if (-corrMap).shape == (1,1): 
#             if (-corrMap)[0,0]>=-score_threshold:
#                 Peaks = np.array([[0,0]])
#             else:
#                 Peaks = []
#         elif (-corrMap).shape[0] == 1: 
#             Peaks = find_peaks((-corrMap)[0], height=-score_threshold)
#             Peaks = [[0,i] for i in Peaks[0]]
#         elif (-corrMap).shape[1] == 1: 
#             Peaks = find_peaks((-corrMap)[:,0], height=-score_threshold)
#             Peaks = [[i,0] for i in Peaks[0]]
#         else: 
#             Peaks = peak_local_max(-corrMap, threshold_abs=-score_threshold, exclude_border=False).tolist()

#         height, width = template.shape[0:2]
#         for peak in Peaks :
#             coeff  = corrMap[tuple(peak)]
#             newHit = {'TemplateName':templateName, 'BBox': ( int(peak[1])+xOffset, int(peak[0])+yOffset, width, height ) , 'Score':coeff}
#             listHit.append(newHit)
#     if listHit:
#         tableHit = pd.DataFrame(listHit)
#     else:
#         tableHit = pd.DataFrame(columns=["TemplateName", "BBox", "Score"])

#     listBoxes  = tableHit["BBox"].to_list()
#     listScores = tableHit["Score"].to_list()
#     listScores = [1-score for score in listScores] # NMS expect high-score for good predictions
#     scoreThreshold = 1-scoreThreshold
#     indexes = cv2.dnn.NMSBoxes(listBoxes, listScores, scoreThreshold, maxOverlap)
#     indexes  = [ index[0] for index in indexes ]
#     outtable = tableHit.iloc[indexes]
    
#     return outtable


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    image_c = image.copy()
    image_blur = cv2.GaussianBlur(image_c, (5,5), 0)
    image_blur = cv2.medianBlur(image_blur,5)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('image',image_gray)
    # cv2.waitKey(0)

    # _,thresh = cv2.threshold(image_gray, 200, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV)
    # circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=20, minRadius=5, maxRadius=100)
    # circles = np.uint16(np.around(circles))   
    # for i in circles[0,:]:
    #     cv2.circle(image_c,(i[0],i[1]),i[2],(0,255,0),2)
    #     cv2.circle(image_c,(i[0],i[1]),2,(0,0,255),3)
    # cv2.imshow('detected circles',image_c)
    # cv2.waitKey(0)
    # pt1 = []
    # for i in circles[0,:]:
    #     pt1.append((i[0], i[1]))

    # canny=cv2.Canny(template_gray,50,80)
    # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # close=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
    # contours2,_ = cv2.findContours(close,cv2.RETR_CCOMP,1)

    # canny=cv2.Canny(image_blur,50,80)
    # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # close=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
    # contours1,_ = cv2.findContours(close,cv2.RETR_CCOMP,1)
    
    # pt1 = []
    # for i in contours1:
    #     similarity = cv2.matchShapes(i, contours2[0], 1, 0)
    #     if similarity < 0.2:
    #         M = cv2.moments(i)
    #         x = int(M['m10']/M['m00'])
    #         y = int(M['m01']/M['m00'])
    #         pt1.append((x,y))

    cor = cv2.cornerHarris(image_gray, 10, 3, 0.001)
    cor = cv2.normalize(cor,None,0,255,cv2.NORM_MINMAX,cv2.CV_32FC1,None)
    loc = np.where( cor >= 0.2*np.max(cor))
    points = zip(*loc[::-1])
    cors=get_corners_list(image)
    for i in cors:
        points=[pt for pt in points if euclidean_distance(pt, i)>10]
    points = np.asarray(points)
    points = np.float32(points)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    _,label,centers=cv2.kmeans(points,K=4,bestLabels=None, criteria=criteria,attempts=100,flags =cv2.KMEANS_PP_CENTERS)
    centers=np.uint(centers)
    pts=[]
    for center in centers:
        pts.append((int(center[0]),int(center[1])))
    
    # pt1 = np.int16(possible_corners)
    # image_c[pt1[:,1],pt1[:,0]]=[0,0,255]
    # cv2.imshow('image',image_c)
    # cv2.waitKey(0)

    pts = sorted(pts, key=lambda x:x[0])
    left2 = pts[:2]
    right2 = pts[-2:]
    left2 = sorted(left2, key=lambda x:x[1])
    right2 = sorted(right2, key=lambda x:x[1])
    top_left = tuple(left2[0])
    top_right =  tuple(right2[0])
    bottom_left = tuple(left2[-1])
    bottom_right = tuple(right2[-1])

    return [top_left, bottom_left, top_right, bottom_right]



def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    image_c = image.copy()
    cv2.line(image_c, markers[0], markers[2], (0, 0, 255), thickness=thickness)
    cv2.line(image_c, markers[0], markers[1], (0, 0, 255), thickness=thickness)
    cv2.line(image_c, markers[2], markers[3], (0, 0, 255), thickness=thickness)
    cv2.line(image_c, markers[1], markers[3], (0, 0, 255), thickness=thickness)

    return image_c

    


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    src = imageA.copy()
    dst_true = imageB.copy()
    h, w = dst_true.shape[:2]
    mks = np.int32(find_markers(dst_true))
    pts = [mks[0], mks[2], mks[3], mks[1]]
    pts = [[np.int32(pt[0]), np.int32(pt[1])] for pt in pts]
    cv2.fillPoly(dst_true, np.array([pts]), (0, 0, 0), 0, 0)

    # inverse wrap
    H = np.linalg.inv(homography)
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1]
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
    blended = cv2.addWeighted(dst_true, 1, dst, 1, 0)

    # bitwise
    # gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    # dst_ = cv2.bitwise_and(dst,dst,mask = mask_inv)
    # blended = cv2.add(dst_true,dst_)

    # cv2.imshow('image', blended)
    # cv2.waitKey(0)

    return blended



def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    u = src_points
    v = dst_points
    U = np.array([[u[0][0], u[0][1], 1, 0, 0, 0, -u[0][0] * v[0][0], -u[0][1] * v[0][0]],
                  [0, 0, 0, u[0][0], u[0][1], 1, -u[0][0] * v[0][1], -u[0][1] * v[0][1]],
                  [u[1][0], u[1][1], 1, 0, 0, 0, -u[1][0] * v[1][0], -u[1][1] * v[1][0]],
                  [0, 0, 0, u[1][0], u[1][1], 1, -u[1][0] * v[1][1], -u[1][1] * v[1][1]],
                  [u[2][0], u[2][1], 1, 0, 0, 0, -u[2][0] * v[2][0], -u[2][1] * v[2][0]],
                  [0, 0, 0, u[2][0], u[2][1], 1, -u[2][0] * v[2][1], -u[2][1] * v[2][1]],
                  [u[3][0], u[3][1], 1, 0, 0, 0, -u[3][0] * v[3][0], -u[3][1] * v[3][0]],
                  [0, 0, 0, u[3][0], u[3][1], 1, -u[3][0] * v[3][1], -u[3][1] * v[3][1]]])
    V = np.array([[v[0][0]],
                  [v[0][1]],
                  [v[1][0]],
                  [v[1][1]],
                  [v[2][0]],
                  [v[2][1]],
                  [v[3][0]],
                  [v[3][1]]])
    h = np.dot(np.linalg.inv(U), V)
    H = np.array([[h[0, 0], h[1, 0], h[2, 0]],
                  [h[3, 0], h[4, 0], h[5, 0]],
                  [h[6, 0], h[7, 0], 1]])
    return H
    


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None


def find_aruco_markers(image, aruco_dict=cv2.aruco.DICT_5X5_50):
    """Finds all ArUco markers and their ID in a given image.

    Hint: you are free to use cv2.aruco module

    Args:
        image (numpy.array): image array.
        aruco_dict (integer): pre-defined ArUco marker dictionary enum.

        For aruco_dict, use cv2.aruco.DICT_5X5_50 for this assignment.
        To find the IDs of markers, use an appropriate function in cv2.aruco module.

    Returns:
        numpy.array: corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        List: list of detected ArUco marker IDs.
    """
    arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, _) = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)
    if ids is not None:
        corners = np.array([corner[0] for corner in corners])
        ids = [i[0] for i in ids]
        res = list(zip(ids,corners))
        res.sort(key=lambda x:x[0])
        markers = np.array([i[1] for i in res])
        ids = [i[0] for i in res]
    else:
        markers = []
        ids = []

    return markers, ids
    


def find_aruco_center(markers, ids):
    """Draw a bounding box of each marker in image. Also, put a marker ID
        on the top-left of each marker.

    Args:
        image (numpy.array): image array.
        markers (numpy.array): corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        ids (list): list of detected ArUco marker IDs.

    Returns:
        List: list of centers of ArUco markers. Each element needs to be
            (x, y) coordinate tuple.
    """
    centers = []
    for pt in markers:
        x = int(round(pt[:,0].mean()))
        y = int(round(pt[:,1].mean()))
        centers.append((x,y))

    return centers
    
