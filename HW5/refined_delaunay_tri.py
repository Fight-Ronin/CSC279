import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict
import sys

sys.setrecursionlimit(10000)  # Increase recursion limit

@dataclass
class Point2D:
    x: float
    y: float
    
    def __eq__(self, other):
        if not isinstance(other, Point2D):
            return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
    
    def __str__(self):
        return f"Point2D({self.x:.2f}, {self.y:.2f})"

class Triangle2D:
    def __init__(self, points=None, p1=None, p2=None, p3=None):
        if points is not None:
            self.vertices = list(points)
        else:
            self.vertices = [p1, p2, p3]
    
    def get_points(self):
        return self.vertices
    
    def get_point(self, index):
        return self.vertices[index]
    
    def set_points(self, points=None, p1=None, p2=None, p3=None):
        if points is not None:
            self.vertices = list(points)
        else:
            self.vertices = [p1, p2, p3]

class DagNode:
    def __init__(self, triangle_index: int):
        self.triangle = triangle_index
        self.children = []
    
    def append_child(self, new_node):
        self.children.append(new_node)
    
    def get_index(self):
        return self.triangle
    
    def get_children(self):
        return self.children

class TriangulationMember(Triangle2D):
    def __init__(self, points, adj_list, dag_node, is_active=True):
        super().__init__(points)
        self.adj_list = list(adj_list)
        self.dag_node = dag_node
        self.active = is_active
    
    def set_active(self):
        self.active = True
    
    def set_inactive(self):
        self.active = False
    
    def is_active(self):
        return self.active
    
    def get_neighbour(self, index):
        return self.adj_list[index]
    
    def get_neighbours(self):
        return self.adj_list
    
    def get_dag_node(self):
        return self.dag_node
    
    def set_neighbour(self, neighbour, new_index):
        self.adj_list[neighbour] = new_index

class Triangulation:
    def __init__(self, init_triangle: Triangle2D, dag_node: DagNode):
        adj_list = [0, 0, 0]
        self.triangles = [TriangulationMember(init_triangle.get_points(), adj_list, dag_node)]
    
    def get_triangle(self, index):
        if 0 <= index < len(self.triangles):
            return self.triangles[index]
        return None
    
    def get_triangles(self):
        return self.triangles
    
    def size(self):
        return len(self.triangles)
    
    def add_triangle(self, triangle):
        self.triangles.append(triangle)
    
    def set_triangle_active(self, index):
        if 0 <= index < len(self.triangles):
            self.triangles[index].set_active()
    
    def set_triangle_inactive(self, index):
        if 0 <= index < len(self.triangles):
            self.triangles[index].set_inactive()
    
    def set_triangle_neighbour(self, triangle, neighbour, new_index):
        if 0 <= triangle < len(self.triangles):
            self.triangles[triangle].set_neighbour(neighbour, new_index)

class GeometryUtils:
    @staticmethod
    def point_in_circle(p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D, include_edges: bool) -> bool:
        try:
            matrix = np.array([
                [p1.x - p4.x, p1.y - p4.y, (p1.x - p4.x)**2 + (p1.y - p4.y)**2],
                [p2.x - p4.x, p2.y - p4.y, (p2.x - p4.x)**2 + (p2.y - p4.y)**2],
                [p3.x - p4.x, p3.y - p4.y, (p3.x - p4.x)**2 + (p3.y - p4.y)**2]
            ])
            det = np.linalg.det(matrix)
            return det > 0 if include_edges else det >= 0
        except:
            return False

    @staticmethod
    def point_position_to_segment(p1: Point2D, p2: Point2D, p: Point2D) -> float:
        return (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x)

    @staticmethod
    def point_in_triangle(p1: Point2D, p2: Point2D, p3: Point2D, p: Point2D, include_edges: bool) -> bool:
        pos1 = GeometryUtils.point_position_to_segment(p1, p2, p)
        pos2 = GeometryUtils.point_position_to_segment(p2, p3, p)
        pos3 = GeometryUtils.point_position_to_segment(p3, p1, p)
        
        if include_edges:
            return (pos1 >= 0 and pos2 >= 0 and pos3 >= 0) or (pos1 <= 0 and pos2 <= 0 and pos3 <= 0)
        else:
            return (pos1 > 0 and pos2 > 0 and pos3 > 0) or (pos1 < 0 and pos2 < 0 and pos3 < 0)

class DelaunayTriangulation:
    MAX_RECURSION_DEPTH = 100
    
    @staticmethod
    def update_index_in_neighbour(triangulation: Triangulation, triangle_index: int, 
                                neighbour_index: int, new_index: int):
        if neighbour_index == 0:
            return
            
        neighbour = triangulation.get_triangle(neighbour_index)
        if neighbour is None:
            return
            
        for i in range(3):
            if neighbour.get_neighbour(i) == triangle_index:
                triangulation.set_triangle_neighbour(neighbour_index, i, new_index)
                break

    @staticmethod
    def locate_point(triangulation: Triangulation, dag: DagNode, point: Point2D) -> DagNode:
        for child in dag.get_children():
            triangle = triangulation.get_triangle(child.get_index())
            if triangle and triangle.is_active():
                if GeometryUtils.point_in_triangle(
                    triangle.get_point(0), triangle.get_point(1),
                    triangle.get_point(2), point, True
                ):
                    return DelaunayTriangulation.locate_point(triangulation, child, point)
        return dag

    @staticmethod
    def flip_edge(triangulation: Triangulation, triangle_index: int, point_index: int, depth: int = 0):
        if depth >= DelaunayTriangulation.MAX_RECURSION_DEPTH:
            return
            
        triangle = triangulation.get_triangle(triangle_index)
        if not triangle or not triangle.is_active():
            return
            
        neighbour_index = triangle.get_neighbour((point_index + 1) % 3)
        if neighbour_index == 0:
            return
            
        adj_triangle = triangulation.get_triangle(neighbour_index)
        if not adj_triangle or not adj_triangle.is_active():
            return

        try:
            adj_point_index = (DelaunayTriangulation.find_index_in_neighbour(
                triangulation, triangle_index, neighbour_index) + 2) % 3

            if GeometryUtils.point_in_circle(
                triangle.get_point(0), triangle.get_point(1),
                triangle.get_point(2), adj_triangle.get_point(adj_point_index), False):

                # Create new triangles
                new_points1 = [
                    triangle.get_point(point_index),
                    triangle.get_point((point_index + 1) % 3),
                    adj_triangle.get_point(adj_point_index)
                ]
                new_points2 = [
                    triangle.get_point(point_index),
                    adj_triangle.get_point(adj_point_index),
                    triangle.get_point((point_index + 2) % 3)
                ]

                # Set up indices
                current_index = triangulation.size()
                new_triangle_index1 = current_index
                new_triangle_index2 = current_index + 1

                # Create adjacency lists
                adj_list1 = [
                    triangle.get_neighbour(point_index),
                    adj_triangle.get_neighbour((adj_point_index + 2) % 3),
                    new_triangle_index2
                ]
                adj_list2 = [
                    new_triangle_index1,
                    adj_triangle.get_neighbour(adj_point_index),
                    triangle.get_neighbour((point_index + 2) % 3)
                ]

                # Create DAG nodes
                dag1 = DagNode(new_triangle_index1)
                dag2 = DagNode(new_triangle_index2)

                # Update triangulation
                triangulation.set_triangle_inactive(triangle_index)
                triangulation.set_triangle_inactive(neighbour_index)

                triangulation.add_triangle(TriangulationMember(new_points1, adj_list1, dag1))
                triangulation.add_triangle(TriangulationMember(new_points2, adj_list2, dag2))

                # Update DAG
                triangle.get_dag_node().append_child(dag1)
                adj_triangle.get_dag_node().append_child(dag1)
                triangle.get_dag_node().append_child(dag2)
                adj_triangle.get_dag_node().append_child(dag2)

                # Update neighbors
                DelaunayTriangulation.update_index_in_neighbour(
                    triangulation, triangle_index,
                    triangle.get_neighbour(point_index), new_triangle_index1)
                DelaunayTriangulation.update_index_in_neighbour(
                    triangulation, neighbour_index,
                    adj_triangle.get_neighbour((adj_point_index + 2) % 3),
                    new_triangle_index1)
                DelaunayTriangulation.update_index_in_neighbour(
                    triangulation, triangle_index,
                    triangle.get_neighbour((point_index + 2) % 3),
                    new_triangle_index2)
                DelaunayTriangulation.update_index_in_neighbour(
                    triangulation, neighbour_index,
                    adj_triangle.get_neighbour(adj_point_index),
                    new_triangle_index2)

                # Recursive flips
                DelaunayTriangulation.flip_edge(triangulation, new_triangle_index1, 0, depth + 1)
                DelaunayTriangulation.flip_edge(triangulation, new_triangle_index2, 0, depth + 1)

        except Exception as e:
            print(f"Error in flip_edge: {e}")
            return

    @staticmethod
    def find_index_in_neighbour(triangulation: Triangulation, triangle_index: int, 
                              neighbour_index: int) -> int:
        neighbour = triangulation.get_triangle(neighbour_index)
        if not neighbour:
            return 3
        for i in range(3):
            if neighbour.get_neighbour(i) == triangle_index:
                return i
        return 3

    @staticmethod
    def incremental_step(triangulation: Triangulation, dag: DagNode, point: Point2D):
        try:
            current_node = DelaunayTriangulation.locate_point(triangulation, dag, point)
            if not current_node:
                return

            triangle_index = current_node.get_index()
            current_triangle = triangulation.get_triangle(triangle_index)
            if not current_triangle:
                return

            # Check if point already exists
            if any(point == current_triangle.get_point(i) for i in range(3)):
                return

            triangulation.set_triangle_inactive(triangle_index)
            current_index = triangulation.size()

            # Create three new triangles
            for i in range(3):
                new_points = [
                    point,
                    current_triangle.get_point(i),
                    current_triangle.get_point((i + 1) % 3)
                ]
                adj_list = [
                    current_index + ((i + 2) % 3),
                    current_triangle.get_neighbour(i),
                    current_index + ((i + 1) % 3)
                ]
                new_dag = DagNode(current_index + i)
                
                triangulation.add_triangle(TriangulationMember(new_points, adj_list, new_dag))
                current_node.append_child(new_dag)
                
                DelaunayTriangulation.update_index_in_neighbour(
                    triangulation,
                    triangle_index,
                    current_triangle.get_neighbour(i),
                    current_index + i
                )

            # Flip edges if necessary
            for i in range(3):
                DelaunayTriangulation.flip_edge(triangulation, current_index + i, 0)

        except Exception as e:
            print(f"Error in incremental_step: {e}")
            return

    @staticmethod
    def get_triangulation(triangulation: Triangulation, dag: DagNode, points: List[Point2D]):
        shuffled_points = points.copy()
        random.shuffle(shuffled_points)
        
        for point in shuffled_points:
            DelaunayTriangulation.incremental_step(triangulation, dag, point)

def create_bounding_triangle(points: List[Point2D]) -> Triangle2D:
    """Create a bounding triangle that contains all points."""
    min_x = min(p.x for p in points) - 0.1
    max_x = max(p.x for p in points) + 0.1
    min_y = min(p.y for p in points) - 0.1
    max_y = max(p.y for p in points) + 0.1
    
    dx = max_x - min_x
    dy = max_y - min_y
    size = max(dx, dy) * 3  # Increased margin
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    p1 = Point2D(center_x - size, center_y - size)
    p2 = Point2D(center_x + size, center_y - size)
    p3 = Point2D(center_x, center_y + size)
    
    return Triangle2D(p1=p1, p2=p2, p3=p3)

def generate_grid_points(n: int) -> List[Point2D]:
    """Generate an n×n grid of points."""
    points = []
    for i in np.linspace(0, 1, n):
        for j in np.linspace(0, 1, n):
            points.append(Point2D(i, j))
    return points

def plot_triangulation(triangulation: Triangulation, points: List[Point2D], title: str = "Delaunay Triangulation"):
    """Plot the triangulation with points and edges."""
    plt.figure(figsize=(12, 12))
    
    # Plot active triangles
    for tri in triangulation.get_triangles():
        if tri.is_active():
            vertices = tri.get_points()
            xs = [v.x for v in vertices + [vertices[0]]]
            ys = [v.y for v in vertices + [vertices[0]]]
            plt.plot(xs, ys, 'b-', alpha=0.5)
    
    # Plot points
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.scatter(xs, ys, c='red', s=20, zorder=3)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_dag(root: DagNode) -> dict:
    """Analyze the DAG structure and return statistics."""
    visited = set()
    depths = defaultdict(int)
    max_depth = 0
    
    def traverse(node, depth=0):
        nonlocal max_depth
        if node in visited:
            return
        
        visited.add(node)
        depths[depth] += 1
        max_depth = max(max_depth, depth)
        
        for child in node.get_children():
            traverse(child, depth + 1)
    
    traverse(root)
    
    total_nodes = len(visited)
    avg_depth = sum(depth * count for depth, count in depths.items()) / total_nodes if total_nodes > 0 else 0
    
    return {
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'total_nodes': total_nodes,
        'depth_distribution': dict(depths)
    }

def run_test(grid_size: int, plot: bool = True):
    """Run Delaunay triangulation test for a given grid size."""
    print(f"\nTesting {grid_size}×{grid_size} grid...")
    
    # Generate test points
    points = generate_grid_points(grid_size)
    print(f"Generated {len(points)} points")
    
    # Create initial triangulation
    bounding_tri = create_bounding_triangle(points)
    root_node = DagNode(0)
    triangulation = Triangulation(bounding_tri, root_node)
    
    # Run triangulation
    try:
        DelaunayTriangulation.get_triangulation(triangulation, root_node, points)
        print("Triangulation completed successfully")
        
        # Analyze DAG
        stats = analyze_dag(root_node)
        print("\nDAG Statistics:")
        print(f"Maximum depth: {stats['max_depth']}")
        print(f"Average depth: {stats['avg_depth']:.2f}")
        print(f"Total nodes: {stats['total_nodes']}")
        
        # Plot if requested
        if plot:
            plot_triangulation(triangulation, points, 
                             f"Delaunay Triangulation ({grid_size}×{grid_size} grid)")
        
        return triangulation, stats
        
    except Exception as e:
        print(f"Error during triangulation: {e}")
        return None, None

def run_comprehensive_test():
    """Run tests for multiple grid sizes and analyze results."""
    grid_sizes = [5, 10, 15, 20, 25]  # Increased sizes
    results = []
    
    for size in grid_sizes:
        triangulation, stats = run_test(size, plot=(size <= 20))  # Only plot smaller grids
        if stats:
            results.append({
                'size': size,
                'stats': stats
            })
    
    # Plot statistics
    if results:
        plt.figure(figsize=(15, 5))
        
        # Plot depths
        plt.subplot(121)
        sizes = [r['size'] for r in results]
        max_depths = [r['stats']['max_depth'] for r in results]
        avg_depths = [r['stats']['avg_depth'] for r in results]
        
        plt.plot(sizes, max_depths, 'ro-', label='Maximum Depth')
        plt.plot(sizes, avg_depths, 'bo-', label='Average Depth')
        plt.xlabel('Grid Size')
        plt.ylabel('Depth')
        plt.title('DAG Depth Analysis')
        plt.legend()
        plt.grid(True)
        
        # Plot total nodes
        plt.subplot(122)
        total_nodes = [r['stats']['total_nodes'] for r in results]
        plt.plot(sizes, total_nodes, 'go-', label='Total Nodes')
        plt.xlabel('Grid Size')
        plt.ylabel('Number of Nodes')
        plt.title('DAG Size Analysis')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run the comprehensive test
    run_comprehensive_test()
