import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
def delaunay(points):
    points_3d = np.hstack([points, (points[:, 0]**2 + points[:, 1]**2).reshape(-1, 1)]) #projecting points on a paraboloid
    hull = ConvexHull(points_3d)
    triangles = set()
    for side in hull.simplices:
        pts = points_3d[side]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if normal[2] < 0:
            triangles.add(tuple(sorted(side)))
    return list(triangles)

def delaunay_reference(points):
    tri = Delaunay(points)
    return tri.simplices

def plot_delaunay(points, triangles):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='red')

    for tri in triangles:
        triangle_points = points[list(tri) + [tri[0]]]
        plt.plot(triangle_points[:, 0], triangle_points[:, 1], 'b-')

    plt.gca().set_aspect('equal')
    plt.title("Delaunay Triangulation Visualization")
    plt.show()

def circle_center(A, B, C):
        Q = 2 * (B[0] - A[0])
        W = 2 * (B[1] - A[1])
        E = 2 * (C[0] - A[0])
        R = 2 * (C[1] - A[1])
        T = B[0] ** 2 - A[0] ** 2 + B[1] ** 2 - A[1] ** 2
        Y = C[0] ** 2 - A[0] ** 2 + C[1] ** 2 - A[1] ** 2
        denom = Q * R - W * E
        if abs(denom) < 1e-12:
            return [np.inf, np.inf]

        cx = (R * T - W * Y) / denom
        cy = (Q * Y - T * E) / denom
        return [cx, cy]


def centers(pts, triangles):
    centers = []
    for tri in triangles:
        triangle_points = pts[list(tri)]
        center = circle_center(triangle_points[0], triangle_points[1], triangle_points[2])
        if center[0] != np.inf:
            centers.append(center)
    return centers

def triangles_to_edges(triangles):
    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[2], tri[0]))))
    return list(edges)

def filter_edges(edges, num_good_pts):
    return [e for e in edges if e[0] < num_good_pts and e[1] < num_good_pts]

def save_edges(edges, filename):
    with open(filename, 'w') as f:
        for i, j in edges:
            f.write(f"{i} {j}\n")

def visualize_edges(points, edges):
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    for i, j in edges:
        p1, p2 = points[i], points[j]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
    plt.gca().set_aspect('equal')
    plt.title("Filtered Crust Edges")
    plt.show()

if __name__ == "__main__":
    pts = np.loadtxt("points.txt", delimiter=" ")
    triangles = delaunay(pts)
    cen = np.array(centers(pts, triangles))
    new_pts = np.vstack([pts, cen])
    pts = new_pts
    triangles = delaunay(pts)
    edges = triangles_to_edges(triangles)
    edges_filtered = filter_edges(edges, len(pts) - len(cen))
    save_edges(edges_filtered, "crust_edges.txt")
    visualize_edges(pts[:len(pts) - len(cen)], edges_filtered)
