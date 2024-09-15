# Axis-Aligned Segment Intersection Counting Algorithm
# Time Complexity: O(n log n)

class Segment:
    def __init__(self, x1, y1, x2, y2):
        # Ensure (x1, y1) is the lower-left point and (x2, y2) is the upper-right point
        if x1 > x2 or y1 > y2:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        # Determine if the segment is horizontal or vertical
        self.is_horizontal = y1 == y2
        self.is_vertical = x1 == x2

def coordinate_compress(coordinates):
    unique_coords = sorted(set(coordinates))
    coord_dict = {coord: idx for idx, coord in enumerate(unique_coords)}
    return coord_dict

class BIT:
    def __init__(self, size):
        self.size = size + 2  # +2 to avoid index issues
        self.tree = [0] * self.size

    def update(self, idx, val):
        idx += 1  # BIT uses 1-based indexing
        while idx < self.size:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        idx += 1
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx
        return result

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

def count_intersections(segments):
    horizontal_segments = []
    vertical_segments = []
    y_coords = []

    for seg in segments:
        if seg.is_horizontal:
            horizontal_segments.append(seg)
            y_coords.append(seg.y1)
        elif seg.is_vertical:
            vertical_segments.append(seg)
            y_coords.append(seg.y1)
            y_coords.append(seg.y2)

    # Coordinate compression for y-coordinates
    y_coord_map = coordinate_compress(y_coords)
    max_y_idx = len(y_coord_map)

    events = []

    # Create add and remove events for horizontal segments
    for seg in horizontal_segments:
        y_idx = y_coord_map[seg.y1]
        events.append((seg.x1, 0, y_idx))  # Add event
        events.append((seg.x2, 2, y_idx))  # Remove event

    # Create query events for vertical segments
    for seg in vertical_segments:
        y1_idx = y_coord_map[seg.y1]
        y2_idx = y_coord_map[seg.y2]
        events.append((seg.x1, 1, min(y1_idx, y2_idx), max(y1_idx, y2_idx)))  # Query event

    # Sort events by x-coordinate and event type
    events.sort(key=lambda x: (x[0], x[1]))

    bit = BIT(max_y_idx)
    intersection_count = 0

    for event in events:
        if event[1] == 0:
            # Add event: add horizontal segment's y-coordinate to BIT
            y_idx = event[2]
            bit.update(y_idx, 1)
        elif event[1] == 2:
            # Remove event: remove horizontal segment's y-coordinate from BIT
            y_idx = event[2]
            bit.update(y_idx, -1)
        else:
            # Query event: count overlapping horizontal segments
            y1_idx, y2_idx = event[2], event[3]
            count = bit.range_query(y1_idx, y2_idx)
            intersection_count += count

    return intersection_count

# Example usage
if __name__ == "__main__":
    # Define some axis-aligned segments
    segments = [
        Segment(1, 2, 5, 2),  # Horizontal segment from (1,2) to (5,2)
        Segment(3, 1, 3, 4),  # Vertical segment from (3,1) to (3,4)
        Segment(2, 3, 6, 3),  # Horizontal segment from (2,3) to (6,3)
        Segment(4, 0, 4, 5),  # Vertical segment from (4,0) to (4,5)
        Segment(0, 1, 7, 1),  # Horizontal segment from (0,1) to (7,1)
        Segment(5, 1, 5, 3),  # Vertical segment from (5,1) to (5,3)
    ]
    count = count_intersections(segments)
    print(f"Total number of intersections: {count}")
