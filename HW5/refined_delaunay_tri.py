import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
        
    def __hash__(self):
        return hash((self.x, self.y))
        
    def __str__(self):
        return f"({self.x}, {self.y})"

def orientation(p, q, r):
    """Returns orientation of ordered triplet (p, q, r).
    Returns:
     0 --> p, q and r are collinear
     1 --> Clockwise
    -1 --> Counterclockwise"""
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if abs(val) < 1e-10:
        return 0
    return 1 if val > 0 else -1

def in_circle(a, b, c, d):
    """Returns True if point d lies inside the circumcircle of triangle abc"""
    # Matrix determinant test for in-circle predicate
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y
    dx, dy = d.x, d.y
    
    # Compute the determinant
    det = np.linalg.det([
        [ax - dx, ay - dy, (ax - dx)**2 + (ay - dy)**2],
        [bx - dx, by - dy, (bx - dx)**2 + (by - dy)**2],
        [cx - dx, cy - dy, (cx - dx)**2 + (cy - dy)**2]
    ])
    
    # Positive determinant means d is inside the circumcircle
    return det > 1e-10

class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.triangles = []
        
    def __eq__(self, other):
        return (self.p1 == other.p1 and self.p2 == other.p2) or \
               (self.p1 == other.p2 and self.p2 == other.p1)
               
    def __hash__(self):
        return hash(frozenset([self.p1, self.p2]))
        
    def __str__(self):
        return f"Edge({self.p1}, {self.p2})"

class Triangle:
    def __init__(self, p1, p2, p3):
        # Ensure counterclockwise orientation
        if orientation(p1, p2, p3) < 0:
            self.points = [p1, p2, p3]
        else:
            self.points = [p1, p3, p2]
        self.edges = []
        self.dag_children = []
        self.depth = 0
        
    def __str__(self):
        return f"Triangle({self.points[0]}, {self.points[1]}, {self.points[2]})"
        
    def circumcircle_contains(self, point):
        """Check if point lies inside this triangle's circumcircle"""
        return in_circle(self.points[0], self.points[1], self.points[2], point)

class DelaunayTriangulation:
    def __init__(self):
        self.triangles = []
        self.edges = set()
        self.bounding_triangle_points = []
        
    def initialize_bounding_triangle(self, points):
        # Calculate bounding box
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        
        # Add margin
        margin = max(max_x - min_x, max_y - min_y)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Create super-triangle vertices
        p1 = Point(center_x - 2*margin, center_y - margin)
        p2 = Point(center_x + 2*margin, center_y - margin)
        p3 = Point(center_x, center_y + 2*margin)
        
        self.bounding_triangle_points = [p1, p2, p3]
        
        # Create initial triangle
        triangle = Triangle(p1, p2, p3)
        e1 = Edge(p1, p2)
        e2 = Edge(p2, p3)
        e3 = Edge(p3, p1)
        
        triangle.edges = [e1, e2, e3]
        for e in triangle.edges:
            e.triangles.append(triangle)
        
        self.triangles.append(triangle)
        self.edges.update(triangle.edges)

    def locate_containing_triangle(self, point):
        """Find the triangle containing the given point"""
        for triangle in self.triangles:
            # Check if point is inside or on the triangle
            inside = True
            for i in range(3):
                p1 = triangle.points[i]
                p2 = triangle.points[(i + 1) % 3]
                if orientation(p1, p2, point) > 0:
                    inside = False
                    break
            if inside:
                return triangle
        return None

    def build(self, points):
        if len(points) < 3:
            return
            
        self.initialize_bounding_triangle(points)
        
        # Randomize points
        points = points.copy()
        random.shuffle(points)
        
        # Insert points one by one
        for point in points:
            bad_triangles = []
            
            # Find all triangles whose circumcircle contains the point
            for triangle in self.triangles:
                if triangle.circumcircle_contains(point):
                    bad_triangles.append(triangle)
            
            # Find boundary of the hole
            edge_count = defaultdict(int)
            for triangle in bad_triangles:
                for edge in triangle.edges:
                    edge_count[edge] += 1
            
            boundary = [edge for edge, count in edge_count.items() if count == 1]
            
            # Remove bad triangles
            for triangle in bad_triangles:
                self.triangles.remove(triangle)
                for edge in triangle.edges:
                    if edge in self.edges:
                        self.edges.remove(edge)
            
            # Create new triangles
            new_triangles = []
            for edge in boundary:
                new_triangle = Triangle(edge.p1, edge.p2, point)
                e1 = Edge(edge.p1, point)
                e2 = Edge(edge.p2, point)
                
                new_triangle.edges = [edge, e1, e2]
                edge.triangles = [new_triangle]
                e1.triangles.append(new_triangle)
                e2.triangles.append(new_triangle)
                
                # Update DAG
                new_triangle.dag_children.extend(bad_triangles)
                new_triangle.depth = max([t.depth for t in bad_triangles], default=0) + 1
                
                new_triangles.append(new_triangle)
                self.edges.update([edge, e1, e2])
            
            self.triangles.extend(new_triangles)
            
        # Remove triangles connected to the bounding triangle
        final_triangles = []
        final_edges = set()
        for triangle in self.triangles:
            if not any(p in self.bounding_triangle_points for p in triangle.points):
                final_triangles.append(triangle)
                final_edges.update(triangle.edges)
        
        self.triangles = final_triangles
        self.edges = final_edges

    def validate_delaunay(self):
        """Validate that the triangulation satisfies the Delaunay property"""
        for triangle in self.triangles:
            for other_triangle in self.triangles:
                if triangle == other_triangle:
                    continue
                # Check if any point from other triangle is inside this triangle's circumcircle
                for point in other_triangle.points:
                    if triangle.circumcircle_contains(point):
                        return False
        return True

    def plot_triangulation(self, title="Delaunay Triangulation"):
        plt.figure(figsize=(10, 10))
        
        # Plot edges
        for edge in self.edges:
            plt.plot([edge.p1.x, edge.p2.x], 
                     [edge.p1.y, edge.p2.y], 'b-', linewidth=0.5)
        
        # Plot points
        points_x = []
        points_y = []
        for triangle in self.triangles:
            for point in triangle.points:
                points_x.append(point.x)
                points_y.append(point.y)
        
        plt.scatter(points_x, points_y, color='red', s=20)
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def get_dag_statistics(self):
        depths = [t.depth for t in self.triangles]
        return max(depths), sum(depths) / len(depths) if depths else 0

def generate_grid_points(n):
    points = []
    for i in range(n):
        for j in range(n):
            points.append(Point(i, j))
    return points

def run_experiment():
    grid_sizes = [2, 4, 8, 16]
    max_depths = []
    avg_depths = []
    
    for n in grid_sizes:
        print(f"Processing {n}x{n} grid...")
        points = generate_grid_points(n)
        triangulation = DelaunayTriangulation()
        triangulation.build(points)
        
        # Validate Delaunay property
        is_valid = triangulation.validate_delaunay()
        print(f"Delaunay property validated: {is_valid}")
        
        triangulation.plot_triangulation(f"{n}x{n} Grid Triangulation")
        
        max_depth, avg_depth = triangulation.get_dag_statistics()
        max_depths.append(max_depth)
        avg_depths.append(avg_depth)
        print(f"Max depth: {max_depth}, Avg depth: {avg_depth}")
    
    # Plot DAG statistics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(grid_sizes, max_depths, 'b-o')
    plt.xlabel('Grid Size (n)')
    plt.ylabel('Maximum DAG Depth')
    plt.title('Maximum Depth vs Grid Size')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(grid_sizes, avg_depths, 'r-o')
    plt.xlabel('Grid Size (n)')
    plt.ylabel('Average DAG Depth')
    plt.title('Average Depth vs Grid Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with small example first
    points = generate_grid_points(5)
    triangulation = DelaunayTriangulation()
    triangulation.build(points)
    
    # Validate and show results
    is_valid = triangulation.validate_delaunay()
    print(f"Delaunay property validated: {is_valid}")
    triangulation.plot_triangulation("5x5 Grid Example")
    
    # Run full experiment
    run_experiment()
