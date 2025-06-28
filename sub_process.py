import numpy as np
import os
import pandas as pd
from plyfile import PlyData, PlyElement
import open3d as o3d
import shutil
def pc_normlize(pc):
    centiroid = np.mean(pc, axis=0)
    pc = pc - centiroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_norm = pc / m
    return pc_norm, centiroid, m

def uni_sampling(swc, num):
    num_array = np.linspace(0, len(swc) - 1, num)
    num_array = np.round(num_array)
    num_list = num_array.tolist()
    num_list = [int(i) for i in num_list]
    new_swc = swc[:][num_list][:]
    new_swc = new_swc.tolist()
    return new_swc


def zeroDataClean(swc):
    swcnum = len(swc)
    zeroArr = swc[:,4]==0
    zeronum = np.sum(zeroArr==True)
    # print(swcnum, zeronum)
    if (np.double(zeronum)/swcnum>=0.1):
        return 0

def zDataClean(swc):
    xData = swc[:, 0]
    xMax = max(xData)
    xMin = min(xData)
    yData = swc[:, 1]
    yMax = max(yData)
    yMin = min(yData)
    zData = swc[:,2]
    zMax = max(zData)
    zMin = min(zData)

    if np.absolute(zMax) <= 0.2 or np.absolute(zMin) <=0.2:
        print(xMax, xMin)
        print(yMax, yMin)
        print(zMax, zMin)
        return 0



def getBranch(swc):
    branch_list = []
    branch = []
    for j in range(len(swc)):
        if j == 0:
            branch.append(j)
            continue
        if swc[j, 6] == swc[j - 1, 0]:
            # print(swc[j,6])
            branch.append(j)
            if j == len(swc) - 1:
                branch_list.append(branch)
        else:
            branch_list.append(branch)
            branch = []
            branch.append(int(swc[j, 6]) - 1)
            branch.append(int(swc[j, 0]) - 1)
            continue
    return branch_list

def uni_sampling_up(swc, num, linkbool=False):
    if linkbool==False:
        if num > len(swc):
            upsamp_k = len(swc) - 1
            # print(upsamp_k)
            need_pnum = int((num - len(swc)) / upsamp_k) + 1
            # print(need_pnum)
            swc_branch = getBranch(swc)
            # print(swc_branch)
            upsamp_data = []
            for i in range(len(swc_branch)):
                for j in range(len(swc_branch[i]) - 1):
                    swc_arr = swc[swc_branch[i][j]]
                    next_swc_arr = swc[swc_branch[i][j + 1]]
                    distance = np.linalg.norm(swc_arr - next_swc_arr)
                    if distance > 0.5:
                        for k in range(need_pnum):

                            alpha = (k + 1) / (need_pnum + 1)
                            upsamp_arr = swc_arr[2:5] - alpha * (swc_arr[2:5] - next_swc_arr[2:5])
                            upsamp_list = upsamp_arr.tolist()
                            upsamp_data.append(upsamp_list)
            upsamp_swc = swc[:, 2:5].tolist()
            upsamp_swc = upsamp_swc + upsamp_data
            upsamp_swc = np.array(upsamp_swc)
            swc_data = uni_sampling(upsamp_swc, num)
            swc_data = np.array(swc_data)
        else:
            swc_data = uni_sampling(swc[:, 2:5], num)
        return swc_data
    else:
        if num > len(swc):
            pass
        else:
            swc_data = uni_sampling(swc[:, 2:5], num)
        return 0

def Normalization(data):
    data_normalized = np.zeros(data.shape)
    for i in range(0,data.shape[0]):
        temp = data
        origin = np.zeros((1,3))
        origin[0][0] = (temp[:,0].max() + temp[:,0].min()) / 2
        origin[0][1] = (temp[:,1].max() + temp[:,1].min()) / 2
        origin[0][2] = (temp[:,2].max() + temp[:,2].min()) / 2
        temp[:,0] = temp[:,0] - origin[0][0]
        temp[:,1] = temp[:,1] - origin[0][1]
        temp[:,2] = temp[:,2] - origin[0][2]
        if temp[:,0].max()>1:
            temp[:,0] = temp[:,0] / temp[:,0].max()
        if temp[:, 1].max() > 1:
            temp[:,1] = temp[:,1] / temp[:,1].max()
        if temp[:, 2].max() > 1:
            temp[:,2] = temp[:,2] / temp[:,2].max()
        data_normalized = temp
        break
    return data_normalized

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample_faster(pts: np.array, num: int) -> np.array:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    pc1 = np.expand_dims(pts, axis=0)   # 1, N, 3
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
    return centroids,pts[centroids][0]





def samplingSWC(swc, file, ratio=0.15):
    num = len(swc)
    branch = getBranch(swc)
    ignorePoint_begin = [i[0] for i in branch]
    ignorePoint_end = [i[-1] for i in branch]
    ignorePoint = ignorePoint_begin + ignorePoint_end
    print(ignorePoint)
    range_list = list(range(0, num-1))
    new_range = [x for x in range_list if x not in ignorePoint]

    print(range_list)
    print(new_range)

    np.random.shuffle(new_range)
    if num>=2048:
        del_list = new_range[:num-2048]
    else:
        del_list = []

    re_branch = []
    for i in branch:
        re_branch.append([x for x in i if x not in del_list])
    print( re_branch)
    for i in sorted(del_list):
        swc[i+1][6] = swc[i][6]
    file_path = "re_" + file
    with open(file_path, 'a+') as f:
        for i in re_branch:
            for j in i:
                if i.index(j) == 0 and re_branch.index(i) != 0:
                    continue
                np.savetxt(f, swc[j], fmt = '%.2f', newline=' ')
                f.write('\n')


def noiseSWC(swc, file, sigma = 1):

    noise = np.random.normal(0, sigma, len(swc)-1) / 100
    swc[1:, 2] = swc[1:, 2] + noise
    swc[1:, 3] = swc[1:, 3] + noise
    swc[1:, 4] = swc[1:, 4] + noise
    file_path = "noise_" + file
    np.savetxt(file_path, swc, fmt='%.2f')

def readPly(fileName='test.ply'):
    plydata = PlyData.read(fileName)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float)
    property = data[0].dtype.names
    for i, name in enumerate(property):
        print(name)
        data_np[:, i] = data_pd[name]
    print(data_np)
    return data_np


from plyfile import PlyData, PlyElement


def visual(file_name,data=None):
    if file_name.split('.')[-1] == 'swc':
        pcd = o3d.geometry.PointCloud()
        print('swc')
        swc = np.loadtxt(file_name)[:,2:5]
        # pcd = o3d.io.read_point_cloud(file_name)
        pcd.points = o3d.utility.Vector3dVector(swc)
    elif file_name.split('.')[-1] == 'ply':
        pcd = o3d.io.read_point_cloud(file_name)
    elif file_name.split('.')[-1] == 'txt':
        pcd = o3d.geometry.PointCloud()
        swc = np.loadtxt(file_name)[:]
        pcd.points = o3d.utility.Vector3dVector(swc)
    else:
        pcd = o3d.geometry.PointCloud()
        swc = data
        pcd.points = o3d.utility.Vector3dVector(swc)
    pcd.paint_uniform_color([0,0,0.7])
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd])

# Press the green button in the gutter to run the script.

def makePly(points, ply_fileName):

    vertex = np.array([tuple(point) for point in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertex, 'vertex')

    PlyData([el], text=True).write(ply_fileName)


def makeNpy(points, npyfile,npyname):

    npypath = os.path.join(npyfile, npyname.rsplit('.', 1)[0])

    np.save(npypath, points)
if __name__ == '__main__':

    rootpath = r''
    threshold = 15000
    npyfile = r''
    filelist = os.listdir(rootpath)

    for j in filelist:
        filepath = os.path.join(rootpath,j)
        swc_data = np.loadtxt(filepath)
        if len(swc_data) > threshold:
            print('fps')
            _, swc_data_list = farthest_point_sample_faster(np.array(swc_data[:,2:5]),threshold)

        else:

            print('uni')
            swc_data_list = uni_sampling_up(swc_data, len(swc_data)+3*(threshold-len(swc_data)))
            _, swc_data_list = farthest_point_sample_faster(np.array(swc_data_list), threshold)
        swc_data_list,_,_ = pc_normlize(swc_data_list)
        swc_data_arr = np.array(swc_data_list)
        makeNpy(swc_data_arr,npyfile,j)
