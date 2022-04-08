import math
import numpy as np
import csv
import g2o
from optimizer import PoseGraphOptimization
import icp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
    
def main():
    ############################################
    #                                          #
    #          DATA PROCESSING PART            #
    #                                          #
    ############################################
    
    # READ POSE DATA FROM CSV. posedata = [pose0,poes1,...]
    # pose0 = numpy matrix 4x4 (SE3)
    posedata = []
    with open('data/pose.csv', 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in readcsv:
            pose = []
            for i in range(len(row)):
                pose.append(float(row[i]))
            r = R.from_quat([pose[3],pose[4],pose[5],pose[6]]).as_dcm()
            t = np.array([[pose[0]],[pose[1]],[pose[2]]])
            RT = np.vstack((np.hstack((r,t)),np.array([0,0,0,1])))
            posedata.append(RT)

    # READ LIDAR DATA FROM CSV. lidardata = [lidar0,lidar1,...]
    # lidar0 = [[x0,x1,x2,x3,...],[y0,y1,y2,y3,...],[0,0,0,0,...],[1,1,1,1,...]] (2D in 3D! : Numpy matrix 4xn)
    lidardata = []
    with open('data/lidar.csv', 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in readcsv:
            x = []
            y = []
            pointNum = np.int_((len(row)-1) / 2)
            for i in range(pointNum):
                x.append(float(row[2 * i])+0.30)
                y.append(row[2 * i + 1])
            lidardata.append(np.array([x, y, np.zeros(len(x)),np.ones(len(x))]).astype(np.float32))
            
            

    ############################################
    #                                          #
    #      SLAM MAIN PART (ASSIGNMENTS)        #
    #                                          #
    ############################################

    # nodes = [node0, node1, ...]
    # node0 = [pose0,lidar0,posediff with node before]
    nodes = [[posedata[0],lidardata[0],np.eye(4)]]
    
    for i in range(1,len(posedata)):
    ####################################################
    # ASSIGNMENTS 1 : CALCULATE POSE DIFF & MAKE NODE  #
    #                                                  #
    # Create nodes at distance or angle intervals.     #
    #                                                  #
    # POSEDIFF : 4x4 numpy matrix of pose difference   #
    # POSE BEFORE : LAST NODE'S POSE ( nodes[-1][0] )  #
    # POSE NOW    : CURRENT POSE     ( posedata[i]  )  #
    ####################################################
        poseDiff =  np.matmul(np.linalg.inv(nodes[-1][0]), posedata[i]) #Calculate pose diff in 4x4 matrix
        distDiff =  np.linalg.norm(np.transpose(poseDiff)[3][:3], 2) #Calculate euclidean distance between two node (using posediff)
        yawDiff  = R.from_dcm(poseDiff[0:3,0:3]).as_euler('zyx')[0] # Robot is in 2D in this lab, so just use Yaw angle
        # If enough distance(0.1[m]) or angle(30[deg]) difference, create node
        
        # print("Node: ", nodes[-1][0])
        # print("Pose: ", posedata[i])
        # print("Result: ", np.matmul(nodes[-1][0], poseDiff))
        # print("PoseDiff: ", poseDiff)
        # print("DistDiff: ", distDiff)
        # print("YawDiff: ", yawDiff)
        # input("Enter...")
        
        if (distDiff > 0.1 or abs(yawDiff)/3.141592*180 > 30):
            nodes.append([posedata[i],lidardata[i],poseDiff])
            # print(len(nodes))


    #############################################################################
    # ASSIGNMENTS 2 : ADD VERTEX AND ODOMETRY EDGE                              #
    #                                                                           #
    # Add vertex of each node and add odometry constraint edge                  #
    # to optimizer.                                                             #
    # FUNCTION1: optimizer.add_vertex(index, g2o.Isometry3d(pose),fixed)        #
    #            index : int, pose : numpy 4x4 mat, fixed : boolean             #
    # FUNCTION2: optimizer.add_edge([src,dst], g2o.Isometry3d(diff),information)#
    #            src: source index, dst: destination index                      #
    #            diff: diff mat (4x4) between source and destination            #   
    #            information: 6x6 numpy matrix of information                   #
    # TIP : You can set simple identity matrix for information.                 #
    #       It will work but not accurate.                                      #
    #############################################################################
    
    # Define optimizer
    optimizer = PoseGraphOptimization();
    
    #Add first node as a fixed vertex. (True = fixed, False = non-fixed)
    optimizer.add_vertex(0, g2o.Isometry3d(nodes[0][0]), True)

 
    for i in range(1,len(nodes)):
        poseDiff =  nodes[i][2]

        xDiff = abs(poseDiff[0][3])
        yDiff = abs(poseDiff[1][3])
        yawDiff  = abs(R.from_dcm(poseDiff[0:3,0:3]).as_euler('zyx')[0])

        info_matrix = [[1/xDiff, 0, 0, 0, 0, 0],
                       [0, 1/yDiff, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1/yawDiff]]

        # print("x, y, yaw diff: ", xDiff, yDiff, yawDiff)
        
        optimizer.add_vertex(i, g2o.Isometry3d(nodes[i][0]), False)
        optimizer.add_edge([i-1,i],g2o.Isometry3d(nodes[i][2]),
                           information=info_matrix)
        

    #############################################################################
    #                                                                           #
    # VISUALIZE LIDAR POINTS INTO GLOBAL. (BEFORE OPTIMIZATION)                 #
    #                                                                           #
    #############################################################################

    for i in range(0,len(nodes)):
        LiDAR = nodes[i][1][0:4];
        LiDAR = np.dot(nodes[i][0], LiDAR)
        plt.scatter(LiDAR[0], LiDAR[1], c='b', marker='o',s=0.2)

    print("Close the plot window to continue...")
    plt.show()

    for i in range(0,len(nodes)):
        pose = nodes[i][0]
        x = pose[0][3]
        y = pose[1][3]
        plt.scatter(x, y, c='b', marker='o',s=0.2)

    print("Original Poses...")
    plt.show()

    # optimizer.save_g2o('beforeSLAM.g2o')
    
    #############################################################################
    # ASSIGNMENTS 3 : FIND LOOP CLOSURE                                         #
    #                                                                           #
    # Simply, you can put all pair in the matching pair, it will be work.       #
    # How can you reduce pairs for less computation? (option)                   #
    #                                                                           #
    #############################################################################  

    # matchingPair = [[src0,dst0],[src1,dst2]...]

    #####################################################################################
    # ASSIGNMENTS 4 : MATCHING PAIRS, OPTIMIZE!                                         #
    #                                                                                   #
    # FUNCTION1: T,D,I = icp.icp(dstPoints,srcPoints,tolerance,max_iterations)          #
    #            T : Transformation from src node to dst node (3x3 matrix:2D matching!) #
    #            D : Distances between corresponding points in srcPoints and dstPoints  #
    #            I : Total iterations           					                    #
    #											                                        #
    # Apply initial translation to dst point cloud! if not, icp will inaccurate         #
    #											                                        #
    #####################################################################################    

    matchingPair = []
    optimizingPair = []

    search_radius = 1
    search_yaw_diff = 20*math.pi/180

    #for src in range(len(nodes)-10):
    #    for dst in range (src+10, len(nodes)):
    for src in range(len(nodes)):
        for dst in range (len(nodes)):

            if abs(src-dst)<10: continue
            if [min(src, dst), max(src, dst)] in optimizingPair: continue

            pose_src = optimizer.get_pose(src)
            pose_dst = optimizer.get_pose(dst)
            r = np.linalg.norm(pose_src.t - pose_dst.t)
            
            dyaw = abs(R.from_dcm(np.matmul(np.linalg.inv(pose_src.R), pose_dst.R)).as_euler('zyx')[0])
            dyaw = min(2*math.pi-dyaw, dyaw)

            if r < search_radius and dyaw < search_yaw_diff :

                # print(dyaw)

                matchingPair.append([src, dst])

                srcLiDAR = nodes[src][1][0:4]
                dstLiDAR = nodes[dst][1][0:4]
                #GET SOURCE NODE POSITION (FROM VERTEX). srcRT = 4x4 matrix
                rt = optimizer.get_pose(src)
                srcRT = np.insert(rt.R, 3, rt.t, axis=1)
                srcRT = np.insert(srcRT, 3, [0, 0, 0, 1], axis=0)
                #GET DESTINATION NODE POSITION (FROM VERTEX). dstRT = 4x4 matrix
                rt = optimizer.get_pose(dst)
                dstRT = np.insert(rt.R, 3, rt.t, axis=1)
                dstRT = np.insert(dstRT, 3, [0, 0, 0, 1], axis=0)

                #PROCESS POINT CLOUD!
                srcPoint = srcLiDAR
                dstPoint = np.matmul(np.matmul(np.linalg.inv(srcRT),dstRT),dstLiDAR)

                #DON'T HAVE TO CHANGE MATCHING FUNCTION
                T, distances, _ = icp.icp(dstPoint[0:2].T,srcPoint[0:2].T,
                                                    tolerance=0.000001,max_iterations=100)
                #### MAKE 3x3 matrix into 4x4 matrix ####
                T = np.insert(T, 2, [0, 0, 0], axis=1)
                T = np.insert(T, 2, [0, 0, 1, 0], axis=0)

                #DRAWING FUNCTION FOR CHECKING ICP DONE WELL : Source blue, Dest green, Dest after ICP red
                
                # dstTrans = np.dot(T, dstPoint)
                # yawChange = abs(R.from_dcm(T[0:3,0:3]).as_euler('zyx')[0])
                
                if np.mean(distances)<0.05:	# ADD CONDITION OF MATCHING SUCCESS (ex: mean of distances less then 0.05 [m])
                    optimizer.add_edge([src,dst], g2o.Isometry3d(np.matmul(T, np.matmul(np.linalg.inv(srcRT), dstRT))),
                                information=np.identity(6)/np.mean(distances))
                    optimizer.optimize()

                    # plt2 = plt
                    # plt2.scatter(dstPoint[0], dstPoint[1], c='g', marker='o',s=0.2)
                    # plt2.scatter(srcPoint[0], srcPoint[1], c='b', marker='o',s=0.2)
                    # plt2.scatter(dstTrans[0], dstTrans[1], c='r', marker='o',s=0.2)

                    # plt2.show()
                    print("Nodes: ", src, ",", dst, "are matched with", np.mean(distances))
                    optimizingPair.append([src, dst])
    
    print(len(matchingPair), "/", np.int_(len(nodes)*(len(nodes)-1)/2), "pairs are checked!")
    print(len(optimizingPair), "/", len(matchingPair), "pairs are matched!")
    
    

    #############################################################################
    #                                                                           #
    # VISUALIZE LIDAR POINTS INTO GLOBAL (AFTER OPTIMIZATION)                   #
    #                                                                           #
    #############################################################################

    for i in range(0,len(nodes)):
        dstLiDAR = nodes[i][1][0:4];
        rt = optimizer.get_pose(i)
        T = np.insert(rt.R, 3, rt.t, axis=1)
        T = np.insert(T, 3, [0, 0, 0, 1], axis=0)
        dstLiDAR = np.dot(T, dstLiDAR)
        plt.scatter(dstLiDAR[0], dstLiDAR[1], c='b', marker='o',s=0.2)
        
    print("Close the plot window to continue...")
    plt.show()

    for i in range(0,len(nodes)):
        rt = optimizer.get_pose(i)
        T = np.insert(rt.R, 3, rt.t, axis=1)
        T = np.insert(T, 3, [0, 0, 0, 1], axis=0)
        x = T[0][3]
        y = T[1][3]
        plt.scatter(x, y, c='b', marker='o',s=0.2)

    for i,j in optimizingPair:
        rt_i = optimizer.get_pose(i)
        T_i = np.insert(rt_i.R, 3, rt_i.t, axis=1)
        T_i = np.insert(T_i, 3, [0, 0, 0, 1], axis=0)
        x_i = T_i[0][3]
        y_i = T_i[1][3]

        rt_j = optimizer.get_pose(j)
        T_j = np.insert(rt_j.R, 3, rt_j.t, axis=1)
        T_j = np.insert(T_j, 3, [0, 0, 0, 1], axis=0)
        x_j = T_j[0][3]
        y_j = T_j[1][3]
        # print("Node", i, "and", j, "are positioned at", [x_i, x_j], [y_i, y_j])
        plt.plot([x_i, x_j], [y_i, y_j], 'g')

    print("Optimized Poses...")
    plt.show()

    # optimizer.save_g2o('afterSLAM.g2o')



if __name__ == '__main__':
    main()
