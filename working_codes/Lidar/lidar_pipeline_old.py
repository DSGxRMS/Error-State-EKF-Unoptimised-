import numpy as np
import cv2
from sklearn.neighbors import KDTree


def euclidean_clustering(P, radius):
    tree = KDTree(P)
    n_points = P.shape[0]
    visited = np.zeros(n_points, dtype=bool)

    clusters = []

    for idx in range(n_points):
        if visited[idx]:
            continue

        cluster = []
        queue = [idx]
        visited[idx] = True

        while queue:
            current_idx = queue.pop(0)
            cluster.append(current_idx)

            indices = tree.query_radius(P[current_idx].reshape(1, -1), r=radius)[0]

            for neighbor_idx in indices:
                if not visited[neighbor_idx]:
                    queue.append(neighbor_idx)
                    visited[neighbor_idx] = True

        clusters.append(cluster)

    return clusters


def visualize_bev_clusters(points, clusters, title="BEV Clusters", size=(800, 800), range_x=(-5, 15), range_y=(-5, 5)):

    bev_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    scale_x = size[1] / (range_x[1] - range_x[0])
    scale_y = size[0] / (range_y[1] - range_y[0])

    color_map = np.random.randint(30, 255, (len(clusters), 3), dtype=np.uint8)

    for cluster_id, cluster in enumerate(clusters):
        color = color_map[cluster_id]

        for idx in cluster:
            x, y = points[idx][0], points[idx][1]
            u = int((x - range_x[0]) * scale_x)
            v = int((y - range_y[0]) * scale_y)
            if 0 <= u < size[1] and 0 <= v < size[0]:
                bev_image[v, u] = color 

    cv2.imshow(title, bev_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bev_image


def filter_clusters_by_size(clusters, min_size=None, max_size=None):

    filtered = []
    for cluster in clusters:
        if len(cluster) >= min_size and (max_size is None or len(cluster) <= max_size):
            filtered.append(cluster)
    return filtered


def points_in_cylinder(points, center, axis, radius=0.2, height=1.8):

    d = points - center

    norm = np.linalg.norm(axis)
    if norm == 0:
        norm = 0.001
    axis = axis / norm

    proj = np.dot(d, axis)

    within_height = (np.abs(proj) <= height / 2)

    proj_vec = np.outer(proj, axis)
    d_perp = (points-center) - proj_vec
    dist_perp = np.linalg.norm(d_perp, axis=1)

    within_radius = dist_perp <= radius

    return within_height & within_radius


def cone_reconstruction(clusters, ground_points, all_points, height=0.8):

    better_clusters = []
    cone_positions = []

    for act_cluster in clusters:
        cluster = ground_points[act_cluster]
        centroid = np.mean(cluster, axis=0)

        cluster = np.array(cluster)
#        print(cluster)
#        print(cluster.shape)
        k = np.argmax(cluster[:, 2])
        top_point = cluster[k]

        axis = top_point-centroid
        if np.linalg.norm(axis)!=0:
            axis_vec = axis / np.linalg.norm(axis)
        else:
            continue

        center = top_point + (axis_vec * height/2)

        mask = points_in_cylinder(all_points, center, axis)
        better_cluster = all_points[mask]

        better_clusters.append(better_cluster)
        position = np.mean(better_cluster, axis=0)
        cone_positions.append(position)

#       better_clusters.append(centroid)

    better_clusters = np.vstack(better_clusters)

    return better_clusters, cone_positions
