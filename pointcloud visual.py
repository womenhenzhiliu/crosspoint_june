import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")

# 从文件中读取点云
pcd = o3d.io.read_point_cloud('/home/june/PycharmProjects/CrossPoint_June1.0/data/ShapeNet/02828884/338fc00ee5b182181faebbdea6bd9be.ply')
print(pcd)
print(np.asarray(pcd.points))
# 可视化点云。使用鼠标/触控板从不同的视点查看几何图形。
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
