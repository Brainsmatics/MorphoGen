import os
import numpy as np
import torch
import open3d as o3d
import networkx as nx
from scipy.spatial.distance import cdist
import utils.L1_skeleton as L1
import random
import time

def knnPointSwcIdx(x, k):
    inner = -2 * torch.matmul(x, x.transpose(0, 1))  ## -2xy

    xx = torch.sum(x ** 2, dim=-1, keepdim=True)  ## p = xx + yy + zz
    pairwise_distance = -xx - inner - xx.transpose(0, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knnPointSwc(swcdata, k):
    knn_idx = knnPointSwcIdx(swcdata, k)
    knn_points = swcdata[knn_idx[:, 1:]]
    knn_swc = torch.sum(knn_points, dim=1) / (k - 1)
    return knn_swc


def pointVis(data, lines=None, data_t=None, lines_t=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:])
    print(pcd)
    pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd])
    if data_t is not None:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(data_t[:])
        print(pcd1)
        pcd1.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd1])
        o3d.visualization.draw_geometries([pcd, pcd1])

    if lines is not None:
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.points = o3d.utility.Vector3dVector(data)
        lines_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, lines_pcd])

    if lines_t is not None:
        lines_pcd1 = o3d.geometry.LineSet()
        lines_pcd1.lines = o3d.utility.Vector2iVector(lines_t)
        lines_pcd1.points = o3d.utility.Vector3dVector(data_t)
        lines_pcd1.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd1, lines_pcd1])

def pointVis_o3d(data, lines=None, data_t=None, lines_t=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window("原始数据")
    opt = vis.get_render_option()
    opt.line_width = 10
    opt.point_size = 5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:])
    print(pcd)
    pcd.paint_uniform_color([1, 0, 0])
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()

    if data_t is not None:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(data_t[:])
        print(pcd1)
        pcd1.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd1], "reconstru")
        o3d.visualization.draw_geometries([pcd, pcd1], "result")


    if lines is not None:
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.points = o3d.utility.Vector3dVector(data)
        lines_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, lines_pcd])

    if lines_t is not None:
        lines_pcd1 = o3d.geometry.LineSet()
        lines_pcd1.lines = o3d.utility.Vector2iVector(lines_t)
        lines_pcd1.points = o3d.utility.Vector3dVector(data_t)
        lines_pcd1.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd1, lines_pcd1])


def minTree(swcpath, swcdata=None):
    if swcdata is not None:
        pointdata = swcdata
    else:
        if swcpath.split('.')[-1] == 'swc':
            pointdata = np.loadtxt(swcpath)[:, 2:5]
        elif swcpath.split('.')[-1] == 'txt':
            pointdata = np.loadtxt(swcpath)
        else:
            raise TypeError('need swc or txt')
    pointcount = pointdata.shape[0]

    G = nx.Graph()

    node_list = range(pointcount)
    G.add_nodes_from(node_list)
    e = [[i, j, np.linalg.norm(pointdata[i] - pointdata[j])] for i in node_list for j in node_list]
    for k in e:
        G.add_edge(k[0], k[1], weight=k[2])
    T = nx.minimum_spanning_tree(G)

    connect = []
    return T.edges, connect


def fps(pc, npoint):
    N = pc.shape[0]
    centroids = np.zeros(npoint, dtype=np.int64)
    distance_ = np.ones(N) * 1e10
    farthest = np.random.randint(0, npoint, dtype=int)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = pc[farthest, :]
        dist = np.sum((pc - centroid) ** 2, -1)
        mask = dist < distance_
        distance_[mask] = dist[mask]
        farthest = np.argmax(distance_, -1)

    return centroids


def farthest_point_sample_faster(swc: np.array, num: int, file_name=None, save_dir=None) -> np.array:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    pts = swc
    pc1 = np.expand_dims(pts, axis=0)  # 1, N, 3
    batchsize, npts, dim = pc1.shape
    centroids = np.zeros((batchsize, num), dtype=np.compat.long)
    distance = np.ones((batchsize, npts)) * 1e10
    farthest_id = np.random.randint(0, npts, (batchsize,), dtype=np.compat.long)
    batch_index = np.arange(batchsize)
    for i in range(num):
        centroids[:, i] = farthest_id
        centro_pt = pc1[batch_index, farthest_id, :].reshape(batchsize, 1, 3)
        dist = np.sum((pc1 - centro_pt) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest_id = np.argmax(distance[batch_index])

    save_path = 'FPS_' + file_name.split('\\')[-1].split('.')[0] + '.txt'
    save_path = os.path.join(save_dir, save_path)
    with open(save_path, 'w+') as f:
        for i in centroids[0]:
            np.savetxt(f, swc[i], fmt='%.2f', newline=' ')
            f.write('\n')
    return centroids, save_path


def cal_inter_sphere(r1, r2, xyz1, xyz2):
    d = np.float32(cdist(xyz1.reshape(-1, 3), xyz2.reshape(-1, 3)).reshape(-1))

    # to check: r1 + r2 => d. If not, set r2=0 and d=r1
    indx = ~np.logical_and(abs(r1 - r2) < d, d < (r1 + r2))
    r2[indx] = 0
    d[indx] = r1

    inter_vol = (np.pi * (r1 + r2 - d) ** 2 * (d ** 2 + 2 * d * (r1 + r2) - 3 * (r1 ** 2 + r2 ** 2) + 6 * r1 * r2)) / (
                12 * d)

    return inter_vol, (~indx).sum()


def nms_swc_sphere(radius, obj_score, xyz, overlap_threshold=0.25):
    # cal volume of each swc ball
    vol = (4.0 / 3.0) * np.pi * (radius ** 3)

    I = np.argsort(obj_score.squeeze())
    pick = []

    dict_num_pts_deleted = {}
    while (I.size != 0):
        last = I.size
        # pick the point with the largest score
        i = I[-1]
        r1, xyz1 = radius[i], xyz[i, :]
        r2, xyz2 = radius[I[:last - 1]], xyz[I[:last - 1], :]
        inter, numInteract = cal_inter_sphere(r1, r2, xyz1, xyz2)
        o = inter / (vol[i] + vol[I[:last - 1]] - inter)

        pts_deleted = np.concatenate(([last - 1], np.where(o > overlap_threshold)[0]))
        dict_num_pts_deleted[obj_score[i]] = numInteract  # pts_deleted.size
        I = np.delete(I, pts_deleted)

        if numInteract > 1:
            pick.append(i)

    return np.array(pick)


def L1_medial(points, NCenters=1000, iters=4):

    maxPoints = 5000
    try_make_skeleton = False

    if len(points) > maxPoints:
        random_indices = random.sample(range(0, len(points)), maxPoints)
        points = points[random_indices, :]

    h0 = L1.get_h0(points) / 8  #
    h = h0

    random.seed(16)
    random_centers = np.random.choice(range(0, len(points)), NCenters)
    centers = points[random_centers, :]

    myCenters = L1.MyCenters(centers, h, maxPoints = NCenters)
    density_weights = L1.get_density_weights(points, h0)
    p = L1.plot3dClass(points, centers)

    time1 = time.time()

    for i in range(iters):

        bridge_points = 0
        non_branch_points = 0
        for center in myCenters.myCenters:
            if center.label == 'bridge_point':
                bridge_points += 1
            if center.label == 'non_branch_point':
                non_branch_points += 1

        centers = myCenters.centers

        t1 = time.perf_counter()

        last_error = 0
        for j in range(2):
            local_indices = L1.get_local_points(points, centers, h)
            error = myCenters.contract(points, local_indices, h, density_weights)
            myCenters.update_properties()

        if try_make_skeleton:
            myCenters.find_connections()

        t1_d = time.perf_counter()
        p.drawCenters(myCenters.myCenters, h)
        t2_d = time.perf_counter()

        draw_time = round(t2_d - t1_d, 3)

        t2 = time.perf_counter()

        tt = round(t2 - t1, 3)
        if non_branch_points == 0:
            print("Found WHOLE skeleton!")
            break

        h = h + h0

    points_s = p.drawCenters(myCenters.myCenters, h)


    time2 = time.time()

    return points_s

def LL1_medial(points, NCenters=5000, iters=4):

    maxPoints = 5000
    try_make_skeleton = False

    if len(points) > maxPoints:
        random_indices = random.sample(range(0, len(points)), maxPoints)
        points = points[random_indices, :]

    h0 = L1.get_h0(points) / 4
    h = h0

    random_centers = fps(points, NCenters)
    centers = points[random_centers, :]

    myCenters = L1.MyCenters(centers, h, maxPoints=NCenters)
    density_weights = L1.get_density_weights(points, h0)
    p = L1.plot3dClass(points, centers)

    for i in range(iters):
        bridge_points = 0
        non_branch_points = 0
        for center in myCenters.myCenters:
            if center.label == 'bridge_point':
                bridge_points += 1
            if center.label == 'non_branch_point':
                non_branch_points += 1

        centers = myCenters.centers

        for j in range(2):
            local_indices = L1.get_local_points(points, centers, h)
            error = myCenters.contract(points, local_indices, h, density_weights)
            myCenters.update_properties()

        if try_make_skeleton:
            myCenters.find_connections()

        p.drawCenters(myCenters.myCenters, h)

    return myCenters.centers


if __name__ == '__main__':
    sign = "L1-Test"

    if sign == "L1-Test":
        swc_path = r''
        real_path = r''
        swc_data = np.loadtxt(swc_path)
        real_data = np.loadtxt(real_path)
        print(swc_data.shape)
        NCenters = int(swc_data.shape[0] / 4 * 3)
        point_swc_L1 = L1_medial(points=swc_data, NCenters=NCenters, iters=1)


        point_swc_L1 = point_swc_L1
        L1.make_plot(point_swc_L1, compare_points=real_data)

    if sign == 'Laplacian-Test':
        swc_path = r''
        swc_data = np.load(swc_path).T
        skeleton, sceleton = Laplacian_ske(swc_data)
        L1.make_plot(sceleton.points, compare_points=swc_data)