import os
import numpy as np
import open3d as o3d
import datetime
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

def icp_core(point_cloud1, point_cloud2):
    """
    Solve transformation from point_cloud2 to point_cloud1, T1_2
    :param point_cloud1: numpy array, size = n x 3, n is num of point
    :param point_cloud2: numpy array, size = n x 3, n is num of point
    :return: transformation matrix T, size = 4x4
    
    Note: point cloud should be in same size. Point with same index are corresponding points.
          For example, point_cloud1[i] and point_cloud2[i] are a pair of cooresponding points.
    
    """
    assert point_cloud1.shape == point_cloud2.shape, 'point cloud size not match'
    
    p_num = point_cloud1.shape[0]
    x_mean = sum(point_cloud1)/p_num
    y_mean = sum(point_cloud2)/p_num

    x_mean = x_mean.reshape(-1, 1)
    y_mean = y_mean.reshape(-1, 1)

    H_matr = np.eye(3)
    for i in range(p_num):
        x = point_cloud1[i].reshape(-1, 1) - x_mean # column vector
        y = point_cloud2[i].reshape(-1, 1) - y_mean
        H_matr += y @ x.T

    Um, _, Vtm = np.linalg.svd(H_matr, full_matrices=False)

    Rm = Vtm.T @ Um.T
    tc = x_mean - Rm @ y_mean

    T1_2 = np.eye(4)
    # TODO: Finish icp based on SVD, you can refer the lecture slides. Please leave comments and explainations for each step.
    T1_2[:3, :3] = Rm
    T1_2[:3,3] = tc.flatten()

    return T1_2


def rough_cp(p1s,p2s):
    assert p1s.shape == p2s.shape, 'rough_cp: point cloud size not match'
    
    p_num = p1s.shape[0]
    x_mean = sum(p1s)/p_num
    y_mean = sum(p2s)/p_num
    x_mean = x_mean.reshape(-1, 1)
    y_mean = y_mean.reshape(-1, 1)

    t = x_mean - y_mean # column vector

    p2s_0 = p2s.T +t
    
    if len(p2s) == len(p2s_0.T):
        return t, p2s_0.T
    else:
        return "NaN", "NaN"


def reject_pairs(p1s, p2s, mean_dist, dist_lst, reject_thed):
    assert p1s.shape == p2s.shape, 'reject_pairs: point cloud size not match'


    
    if mean_dist != 0.:
        dele_id_lst = [i for i, dist in enumerate(dist_lst) if dist[0] > reject_thed* mean_dist]
        
    else:
        print("Beginning")
        m_dist = sum(dist_lst)/len(dist_lst)

        dele_id_lst = [i for i, dist in enumerate(dist_lst) if dist[0] > reject_thed* m_dist]

    return np.delete(p1s, dele_id_lst, axis=0), np.delete(p2s, dele_id_lst, axis=0)

def draw_curve(res_lst, title, x_label, y_label):
    iter = list(range(1,len(res_lst)+1))
    plt.figure(figsize=(10,8))
    plt.plot(iter, res_lst, marker='o', linestyle='-', markersize=4)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(True)

    file_name = "_".join(title.split(' '))
    plt.savefig(f"{file_name}.png",dpi=800 )
    plt.show()

def solve_icp_without_known_correspondence(point_cloud1, point_cloud2, n_iter, threshold):
    point_cloud2_temp = point_cloud2.copy()
    T_1_2accumulated = np.eye(4)

    # viz
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
    pcd1.paint_uniform_color([0, 0, 1]) # Blue
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pcd1)


    total_time_cost = 0

    # Rough move p2 to p2_0
    t0, point_cloud2_temp = rough_cp(point_cloud1, point_cloud2)
    T_1_2accumulated[:3,3] = t0.flatten()
    print("finish rough move")

    tree = KDTree(point_cloud1)

    error_lst = []
    save_path = "icp_result.txt"
    if os.path.exists(save_path):
        os.remove(save_path)
    res_f = open(save_path,"a")
    # Beginning of ICP
    for i in range(n_iter):
        start_time = datetime.datetime.now()
        
        # TODO: Try to estimate correspondence of points between 2 point clouds, 
        #       and reindex point_cloud2 based on your estimated correspondence


        distances, indices = tree.query(point_cloud2_temp, k=1)

        id1s = np.arange(indices.shape[0]).reshape(-1,1)
        point_pairs = np.hstack((id1s, indices))
        _, uni_id = np.unique(point_pairs[:,1], return_index=True)
        id_pairs = point_pairs[np.sort(uni_id)]

        point_cloud1_reorder = point_cloud1[id_pairs[:, 0]]
        point_cloud2_reorder = point_cloud2_temp[id_pairs[:, 1]]
        distances = distances[id_pairs[:, 1]]

        
        mean_distance = mean_dist(point_cloud1, point_cloud2_temp)
        reject_thed = 3 
        if mean_distance > 1.5:
            pass
        else:
            reject_thed = 1
        
        point_cloud1_reorg, point_cloud2_reorg = reject_pairs(point_cloud1_reorder,
                                                              point_cloud2_reorder,
                                                              mean_distance,
                                                              distances, reject_thed)

        
        # Solve ICP for current iteration
        T1_2_cur = icp_core(point_cloud1_reorg, point_cloud2_reorg)
        
        
        # TODO: Update point cloud2 using transform from current iteration
        R1_2 = T1_2_cur[:3,:3]
        t1_2 = T1_2_cur[:3,3].reshape(-1,1)
        point_cloud2_ta = R1_2 @ point_cloud2_temp.T + t1_2
        point_cloud2_ta = point_cloud2_ta.T
        
        mean_distance = mean_dist(point_cloud1, point_cloud2_ta)
        error_lst.append(mean_distance)
        res_f.write(f"{mean_distance}\n")
        # if i > 0:
        #     diff_lst = np.diff(error_lst) / np.array(error_lst[:-1])
        #     if diff_lst[-1] > 0. and error_lst[-2] <= 1.8 and error_lst[-1] > 1.5* error_lst[-2]:
        #         print("! Remain!")
        #         error_lst[-1] = error_lst[-2]
        #         T1_2_cur = np.eye(4)
        #         point_cloud2_ta = point_cloud2_temp
        #     else:
        #         pass
        point_cloud2_temp = point_cloud2_ta

        # TODO: Update accumulated transformation
        T_1_2accumulated = T1_2_cur @ T_1_2accumulated


        end_time = datetime.datetime.now()
        time_difference = (end_time - start_time).total_seconds()
        total_time_cost += time_difference

        print('-----------------------------------------')
        print('iteration = ' + str(i+1))
        print('time cost = ' + str(time_difference) + 's')
        print('total time cost = ' + str(total_time_cost) + 's')  

        print(f"reject threshold = {reject_thed}")
        print(f"number of point pairs after rejection: {len(point_cloud2_reorg)}")
        
        print('T1_2_cur = ')
        print(T1_2_cur)
        print('accumulated T = ')
        print(T_1_2accumulated)
        print('mean_error= ' + str(mean_distance))

        # Update visualization
        pcd2_transed = o3d.geometry.PointCloud()
        pcd2_transed.points = o3d.utility.Vector3dVector(point_cloud2_temp)
        pcd2_transed.paint_uniform_color([1, 0, 0]) # Red
        vis.add_geometry(pcd2_transed)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(pcd2_transed)

        

        if mean_distance < 0.00001 or mean_distance < threshold:
            print('------- fully converged! -------')
            break
        
        if i == n_iter - 1:
            print('------- reach iteration limit -------')

    print('time cost: ' + str(total_time_cost) + ' s')
    
    res_f.close()

    draw_curve(error_lst, 
               "ICP Convergence Illustration",
               "Iteration",
               "Mean Error")
    
    vis.destroy_window()
    
    # Final visualization
    pcd2_final = o3d.geometry.PointCloud()
    pcd2_final.points = o3d.utility.Vector3dVector(point_cloud2_temp)
    pcd2_final.paint_uniform_color([1, 0, 0]) # Red
    o3d.visualization.draw_geometries([axis_pcd, pcd1, pcd2_final])

  

def mean_dist(point_cloud1, point_cloud2):
    dis_array = []
    for i in range(point_cloud1.shape[0]):
        dif = point_cloud1[i] - point_cloud2[i]
        dis = np.linalg.norm(dif)
        dis_array.append(dis)
        
    return np.mean(np.array(dis_array))

def main():
    print('start program')
    data_base_path = 'data/'
    pcd1 = o3d.io.read_point_cloud(data_base_path+'bunny1.ply') # change to your file path
    pcd2 = o3d.io.read_point_cloud(data_base_path+'bunny2.ply') # change to your file path
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)

    solve_icp_without_known_correspondence(points1, points2, n_iter=150, threshold=0.0159)

if __name__ == '__main__':
    main()