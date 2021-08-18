import numpy as np
import cv2
import pickle
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from numpy.lib.function_base import append

calib_result_pickle = pickle.load(open("camera_calib_pickle.p", "rb" ))
optimal_camera_matrix = calib_result_pickle["optimal_camera_matrix"]
focal = optimal_camera_matrix[0][0]
pp = (optimal_camera_matrix[0][2], optimal_camera_matrix[1][2])
cap = cv2.VideoCapture(0)
trajectory = [np.array([0, 0, 0])]

x = []
y = []
z = []
fig = plt.figure()
P = np.eye(4)
def extract_feature(image):
    surf = cv2.xfeatures2d.SIFT_create()
    feature_image = np.copy(image)
    keypoints, descriptor = surf.detectAndCompute(feature_image, None)
    return keypoints, descriptor

def filter_match(match, threshold):
    filter_matched = []
    for m, n in match:
        if m.distance<threshold*n.distance:
            filter_matched.append(m)
    return filter_matched

def estimate_trajectory(match, kp1, kp2, k):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
        
    ### START CODE HERE ###
    for m in match:
        query_idx = m.queryIdx
        train_idx = m.trainIdx

        # get first img matched keypoints
        p1_x, p1_y = kp1[query_idx].pt
        image2_points.append([p1_x, p1_y])

        # get second img matched keypoints
        p2_x, p2_y = kp2[train_idx].pt
        image1_points.append([p2_x, p2_y])

    # essential matrix
    E, _ = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), focal, pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, np.array(image1_points), np.array(image2_points))
    
    rmat = R
    tvec = t  
    return rmat, tvec, image1_points, image2_points


if not cap.isOpened():
    print("Cannot open camera")
    exit()

def animate(num):
    global P
    ret, frame1 = cap.read()
    cv2.waitKey(100)
    while True:
        # Capture frame-by-frame
        
        ret, frame2 = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        kp1, des1 = extract_feature(frame1)
        kp2, des2 = extract_feature(frame2)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        match = matcher.knnMatch(des1, des2, 2)
        filter_m = filter_match(match, 0.50)
        rmat, tvec, imgae1_points, image2_points =estimate_trajectory(filter_m, kp1, kp2, optimal_camera_matrix)
        R = rmat
        t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
        P_new = np.eye(4)
        P_new[0:3,0:3] = R.T
        P_new[0:3,3] = (-R.T).dot(t)
        P = P.dot(P_new)
        trajectory.append(P[:3,3])
        frame1 = frame2
        x.append(trajectory[-1][0])
        y.append(trajectory[-1][1])
        z.append(0)
        print(trajectory[-1])
        ax = fig.add_subplot(111, projection='3d')
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('z')
        ax.set_xlim(-100, 200)
        ax.set_ylim(-100, 200)
        graph = ax.plot(x,y,color='orange',marker='o')
        pos = [trajectory[-1][0], trajectory[-1][1], 0]
        with open("/media/marzan/workspace/visual_odometry/position.csv", "a")as output:
            writer = csv.writer(output, delimiter=",")
            writer.writerow(pos)

        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    
    cap.release()
    cv2.destroyAllWindows()
    return graph

ani = animation.FuncAnimation(fig, animate, interval=5)
plt.show()
