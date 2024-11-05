import random
import matplotlib.pyplot as plt
from collections import deque

# ----------------------------
# Data Structures Definitions
# ----------------------------

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1  # Point
        self.p2 = p2  # Point
        self.triangles = []  # Adjacent triangles (up to 2)

    def __repr__(self):
        return f"Segment(({self.p1.x}, {self.p1.y}) - ({self.p2.x}, {self.p2.y}))"

class Triangle:
    def __init__(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]  # Points
        self.edges = [
            SegmentWrapper(p1, p2),
            SegmentWrapper(p2, p3),
            SegmentWrapper(p3, p1)
        ]
        self.children = []  # For history DAG

    def __repr__(self):
        verts = ', '.join([f"({p.x}, {p.y})" for p in self.vertices])
        return f"Triangle({verts})"

class SegmentWrapper:
    """
    Helper class to uniquely identify a segment irrespective of point order.
    """
    def __init__(self, p1, p2):
        # Ensure consistent ordering
        if (p1.x, p1.y) < (p2.x, p2.y):
            self.p1, self.p2 = p1, p2
        else:
            self.p1, self.p2 = p2, p1

    def __eq__(self, other):
        return (self.p1.x, self.p1.y) == (other.p1.x, other.p1.y) and \
               (self.p2.x, self.p2.y) == (other.p2.x, other.p2.y)

    def __hash__(self):
        return hash(((self.p1.x, self.p1.y), (self.p2.x, self.p2.y)))

class HistoryDAGNode:
    def __init__(self, triangle):
        self.triangle = triangle  # Triangle object
        self.children = []        # List of HistoryDAGNode

    def add_child(self, child_node):
        self.children.append(child_node)

# ----------------------------
# Delaunay Triangulation Class
# ----------------------------

class DelaunayTriangulation:
    def __init__(self, points):
        self.points = points  # List of Point objects
        self.segments = set() # Set of SegmentWrapper
        self.triangles = []   # List of Triangle objects
        self.history_root = None

    def create_super_triangle(self):
        """
        Create a super-triangle that encompasses all the points.
        """
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)

        dx = max_x - min_x
        dy = max_y - min_y
        delta_max = max(dx, dy) * 10  # Make it large enough

        # Create three points that form a super-triangle
        p1 = Point(min_x - delta_max, min_y - delta_max)
        p2 = Point(min_x + delta_max, min_y - delta_max)
        p3 = Point(min_x, max_y + delta_max)

        super_triangle = Triangle(p1, p2, p3)
        self.triangles.append(super_triangle)
        self.history_root = HistoryDAGNode(super_triangle)

        # Add segments of the super-triangle
        for edge in super_triangle.edges:
            self.segments.add(edge)

    def locate_containing_triangle(self, point):
        """
        Traverse the history DAG to locate the triangle containing the point.
        This is a simplified approach and may not be optimized.
        """
        node = self.history_root
        while node.children:
            found = False
            for child in node.children:
                if self.point_in_triangle(point, child.triangle):
                    node = child
                    found = True
                    break
            if not found:
                break  # Point not found in any child; fallback
        # Now, node.triangle should contain the point
        return node.triangle, node

    @staticmethod
    def point_in_triangle(p, triangle):
        """
        Check if point p is inside the given triangle using barycentric coordinates.
        """
        def sign(p1, p2, p3):
            return (p1.x - p3.x) * (p2.y - p3.y) - \
                   (p2.x - p3.x) * (p1.y - p3.y)

        b1 = sign(p, triangle.vertices[0], triangle.vertices[1]) < 0.0
        b2 = sign(p, triangle.vertices[1], triangle.vertices[2]) < 0.0
        b3 = sign(p, triangle.vertices[2], triangle.vertices[0]) < 0.0

        return ((b1 == b2) and (b2 == b3))

    def insert_point(self, point):
        """
        Insert a single point into the triangulation.
        """
        containing_triangle, containing_node = self.locate_containing_triangle(point)

        # Subdivide the containing triangle into three new triangles
        t1 = Triangle(containing_triangle.vertices[0], containing_triangle.vertices[1], point)
        t2 = Triangle(containing_triangle.vertices[1], containing_triangle.vertices[2], point)
        t3 = Triangle(containing_triangle.vertices[2], containing_triangle.vertices[0], point)

        self.triangles.extend([t1, t2, t3])

        # Update segments
        self.segments.remove(SegmentWrapper(containing_triangle.vertices[0], containing_triangle.vertices[1]))
        self.segments.remove(SegmentWrapper(containing_triangle.vertices[1], containing_triangle.vertices[2]))
        self.segments.remove(SegmentWrapper(containing_triangle.vertices[2], containing_triangle.vertices[0]))

        for t in [t1, t2, t3]:
            for edge in t.edges:
                self.segments.add(edge)

        # Update history DAG
        child_nodes = [
            HistoryDAGNode(t1),
            HistoryDAGNode(t2),
            HistoryDAGNode(t3)
        ]
        for child in child_nodes:
            containing_node.add_child(child)

        # Edge Legalization would go here (not implemented for brevity)

    def build_triangulation(self):
        """
        Build the triangulation by inserting all points.
        """
        self.create_super_triangle()
        for point in self.points:
            self.insert_point(point)
        self.remove_super_triangle()

    def remove_super_triangle(self):
        """
        Remove any triangles that share a vertex with the super-triangle.
        """
        # Assuming the first triangle is the super-triangle
        super_vertices = set(self.history_root.triangle.vertices)
        self.triangles = [t for t in self.triangles if not any(v in super_vertices for v in t.vertices)]

    def calculate_dag_depths(self):
        """
        Calculate maximum and average depths of the history DAG.
        """
        depths = []
        queue = deque([(self.history_root, 0)])

        while queue:
            node, depth = queue.popleft()
            if not node.children:
                depths.append(depth)
            else:
                for child in node.children:
                    queue.append((child, depth + 1))

        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0
        return max_depth, avg_depth

# ----------------------------
# Utility Functions
# ----------------------------

def generate_grid_points(n):
    points = [Point(i, j) for i in range(n) for j in range(n)]
    random.shuffle(points)
    return points

# ----------------------------
# Main Execution and Plotting
# ----------------------------

def main():
    n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_depths = []
    avg_depths = []

    for n in n_values:
        print(f"Processing n = {n}")
        points = generate_grid_points(n)
        dt = DelaunayTriangulation(points)
        dt.build_triangulation()
        max_depth, avg_depth = dt.calculate_dag_depths()
        max_depths.append(max_depth)
        avg_depths.append(avg_depth)
        print(f"n = {n}: Max Depth = {max_depth}, Avg Depth = {avg_depth}")

    # Plotting the Results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(n_values, max_depths, marker='o', color='blue')
    plt.title('Maximum DAG Depth vs n')
    plt.xlabel('n')
    plt.ylabel('Maximum Depth')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_values, avg_depths, marker='o', color='orange')
    plt.title('Average DAG Depth vs n')
    plt.xlabel('n')
    plt.ylabel('Average Depth')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
