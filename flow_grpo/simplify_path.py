import re
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional

# ------------------- SVG compression related functions -------------------

def optimize_path_commands(d_str: str) -> str:
    """
    Optimize path command strings using shorter representations
    
    Args:
        d_str: Original path command string
        
    Returns:
        Optimized path command string
    """
    # Replace multiple spaces with single space
    d_str = re.sub(r'\s+', ' ', d_str)
    
    # Remove unnecessary 0, e.g.: 0.5 -> .5
    d_str = re.sub(r'([^0-9])0\.', r'\1.', d_str)
    
    # Remove numbers with decimal point followed by 0, e.g.: 10.0 -> 10
    d_str = re.sub(r'(\d+)\.0([^0-9]|$)', r'\1\2', d_str)
    
    # Negative numbers can use - instead of 0-
    d_str = re.sub(r' 0-', ' -', d_str)
    
    return d_str.strip()

def shorten_color(match) -> str:
    """Shorten hexadecimal color codes, e.g., #AABBCC -> #ABC"""
    color = match.group(1)
    if len(color) == 7 and color[1] == color[2] and color[3] == color[4] and color[5] == color[6]:
        return f'"#{color[1]}{color[3]}{color[5]}"'
    return match.group(0)

def process_numeric_attributes(svg_str: str) -> str:
    """
    Process numeric attributes in SVG strings, removing unnecessary decimal points and zeros
    
    Args:
        svg_str: Original SVG string
        
    Returns:
        Processed SVG string
    """
    # Process decimal points in d attributes
    pattern_d = r'd="([^"]*)"'
    for match in re.finditer(pattern_d, svg_str):
        d_attr = match.group(1)
        # Optimize floating point representation: if decimal part is 0, remove decimal part
        new_d_attr = re.sub(r'(\d+)\.0+([^0-9]|$)', r'\1\2', d_attr)
        # For floating point numbers close to integers (decimal part < 0.001), convert to integer
        new_d_attr = re.sub(r'(\d+\.\d{1,3})([^0-9]|$)', 
                           lambda m: str(int(float(m.group(1)))) + m.group(2) 
                           if abs(float(m.group(1)) - int(float(m.group(1)))) < 0.001 
                           else m.group(0), new_d_attr)
        # Handle negative numbers after parentheses
        new_d_attr = re.sub(r'([,\(])0-', r'\1-', new_d_attr)
        svg_str = svg_str.replace(f'd="{d_attr}"', f'd="{new_d_attr}"')
    
    # Process other possible floating point numbers (such as width, height, etc.)
    # 1. Replace numbers ending with .0
    svg_str = re.sub(r'(\d+)\.0+(?=[\s"\'])', r'\1', svg_str)
    # 2. For floating point numbers in important attributes, only convert to integer when close to integer
    svg_str = re.sub(r'(width|height|x|y|cx|cy|r|rx|ry)="(\d+\.\d+)"', 
                    lambda m: f'{m.group(1)}="{int(float(m.group(2)))}"' 
                    if abs(float(m.group(2)) - int(float(m.group(2)))) < 0.001 
                    else m.group(0), svg_str)
    
    return svg_str

def compress_whitespace(svg_str: str) -> str:
    """
    Compress whitespace characters in SVG string
    
    Args:
        svg_str: Original SVG string
        
    Returns:
        Compressed SVG string
    """
    # Remove redundant spaces
    compressed = re.sub(r'\s+', ' ', svg_str)
    compressed = re.sub(r'>\s+<', '><', compressed)
    compressed = re.sub(r'\s+/>', '/>', compressed)
    
    return compressed.strip()

def remove_commas_from_path(d_str: str) -> str:
    """
    Remove unnecessary commas from paths, replace with spaces
    e.g.: M0,0 L23,0 -> M0 0L23 0
    
    Args:
        d_str: Original path command string
        
    Returns:
        Optimized path command string
    """
    # Replace commas with spaces
    d_str = re.sub(r',', ' ', d_str)
    
    # Remove spaces after command letters (e.g., "M 0" -> "M0")
    d_str = re.sub(r'([MLHVCSQTAZmlhvcsqtaz]) ', r'\1', d_str)
    
    return d_str

def optimize_consecutive_commands(d_str: str) -> str:
    """
    Optimize consecutive identical commands by omitting repeated command letters
    e.g.: L10 0 L20 10 -> L10 0 20 10
    
    Args:
        d_str: Original path command string
        
    Returns:
        Optimized path command string
    """
    # Match all SVG path commands and their parameters
    pattern = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
    matches = re.findall(pattern, d_str)
    
    if not matches:
        return d_str
        
    result = []
    prev_cmd = None
    
    for cmd, params in matches:
        # If command is same as previous, only add parameters
        if cmd == prev_cmd:
            result.append(params)
        else:
            result.append(cmd + params)
            prev_cmd = cmd
    
    return ''.join(result)

def apply_transform_to_path(d_str: str, transform: str) -> str:
    """
    Apply transform directly to path coordinates, properly handling multiple subpaths
    Currently only supports translate transform
    
    Args:
        d_str: Original path command string
        transform: SVG transform string
        
    Returns:
        Path command string with transform applied
    """
    # Parse translate transform
    translate_match = re.search(r'translate\(([^,]+),([^)]+)\)', transform)
    if not translate_match:
        return d_str
        
    tx = int(float(translate_match.group(1)))
    ty = int(float(translate_match.group(2)))
    
    # Check if contains multiple subpaths (multiple M commands)
    m_count = len(re.findall(r'[Mm]', d_str))
    
    if m_count <= 1:
        # Single subpath, use original logic
        points = extract_polygon_points(d_str)
        if not points:
            return d_str
            
        # Apply translate to all points
        points = [(x + tx, y + ty) for x, y in points]
        
        # Determine if path is closed
        close_path = d_str.strip().upper().endswith('Z')
        
        # Rebuild path using relative coordinates
        return rebuild_polygon_path(points, close_path, use_relative=True)
    else:
        # Multiple subpaths, process each subpath separately
        # Use regex to split subpaths
        subpaths = re.split(r'(?=[Mm])', d_str)
        subpaths = [sp.strip() for sp in subpaths if sp.strip()]
        
        transformed_subpaths = []
        
        for subpath in subpaths:
            # Apply transform to each subpath
            points = extract_polygon_points(subpath)
            if not points:
                continue
                
            # Apply translate to all points
            points = [(x + tx, y + ty) for x, y in points]
            
            # Determine if subpath is closed
            close_path = subpath.strip().upper().endswith('Z')
            
            # Rebuild subpath using relative coordinates
            transformed_subpath = rebuild_polygon_path(points, close_path, use_relative=True)
            if transformed_subpath:
                transformed_subpaths.append(transformed_subpath)
        
        # Recombine all subpaths
        return ' '.join(transformed_subpaths)

def compress_svg(svg_string: str) -> str:
    """
    Main function for compressing SVG string size, can be used independently
    
    Args:
        svg_string: Original SVG string
        
    Returns:
        Compressed SVG string
    """
    # Compress whitespace characters
    compressed = compress_whitespace(svg_string)
    
    # Compress color codes
    compressed = re.sub(r'"(#[0-9A-Fa-f]{6})"', shorten_color, compressed)
    
    # Process numeric attributes
    compressed = process_numeric_attributes(compressed)
    
    # Compress viewBox attribute
    compressed = re.sub(r'viewBox="([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)"', r'viewBox="\1 \2 \3 \4"', compressed)
    
    # Process SVG paths
    try:
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        ET.register_namespace('', namespaces['svg'])
        
        root = ET.fromstring(compressed)
        path_elements = root.findall('.//svg:path', namespaces=namespaces)
        
        for path_elem in path_elements:
            d_str = path_elem.get('d')
            if not d_str:
                continue
                
            # Get transform attribute
            transform = path_elem.get('transform')
            
            # If there's transform, apply it to the path
            if transform:
                d_str = apply_transform_to_path(d_str, transform)
                # Remove transform attribute
                path_elem.attrib.pop('transform', None)
            
            # Remove commas
            d_str = remove_commas_from_path(d_str)
            
            # Optimize consecutive commands
            d_str = optimize_consecutive_commands(d_str)
            
            # General path command optimization
            d_str = optimize_path_commands(d_str)
            
            # Update path
            path_elem.set('d', d_str)
        
        # Convert XML to string
        compressed = ET.tostring(root, encoding='unicode')
        
    except Exception as e:
        print(f"Path compression error: {e}")
    
    return compressed

# ------------------- Path simplification related functions -------------------

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def perpendicular_distance(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
    """Calculate perpendicular distance from point to line"""
    if line_start == line_end:
        return calculate_distance(point, line_start)
    
    line_length_squared = (line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2
    
    t = max(0, min(1, (
        (point[0] - line_start[0]) * (line_end[0] - line_start[0]) +
        (point[1] - line_start[1]) * (line_end[1] - line_start[1])
    ) / line_length_squared))
    
    projection = (
        line_start[0] + t * (line_end[0] - line_start[0]),
        line_start[1] + t * (line_end[1] - line_start[1])
    )
    
    return calculate_distance(point, projection)

def ramer_douglas_peucker(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    """Simplify polygon using Ramer-Douglas-Peucker algorithm"""
    if len(points) <= 2:
        return points
    
    dmax = 0
    index = 0
    start, end = points[0], points[-1]
    
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], start, end)
        if d > dmax:
            index = i
            dmax = d
    
    if dmax > epsilon:
        rec_results1 = ramer_douglas_peucker(points[:index + 1], epsilon)
        rec_results2 = ramer_douglas_peucker(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2
    else:
        return [start, end]

def extract_polygon_points(d_str: str) -> List[Tuple[int, int]]:
    """Extract point coordinates from polygon path, convert to integers"""
    # Match commands and coordinates
    pattern = r'([MLZmlz])|([-+]?\d*\.\d+|[-+]?\d+)'
    tokens = re.findall(pattern, d_str)
    
    points = []
    current_cmd = None
    current_x, current_y = 0, 0
    coords = []
    
    for cmd, param in tokens:
        if cmd:  # Found command
            current_cmd = cmd
            coords = []
        elif param and current_cmd in 'MLml':  # Found coordinate
            coords.append(int(float(param)))  # Ensure conversion to integer
            if len(coords) == 2:
                if current_cmd in 'ml':  # Relative coordinates
                    current_x += coords[0]
                    current_y += coords[1]
                else:  # Absolute coordinates
                    current_x, current_y = coords
                points.append((current_x, current_y))
                coords = []
    
    return points

def rebuild_polygon_path(points: List[Tuple[int, int]], close_path: bool = True, use_relative: bool = True) -> str:
    """Rebuild polygon path from point list, using relative coordinates to reduce SVG size"""
    if not points:
        return ""
    
    # First point must use absolute coordinates
    path_str = f"M{int(points[0][0])},{int(points[0][1])}"
    
    if use_relative:
        # Use relative coordinates (lowercase l command)
        prev_x, prev_y = points[0]
        for x, y in points[1:]:
            dx, dy = x - prev_x, y - prev_y
            path_str += f" l{int(dx)},{int(dy)}"
            prev_x, prev_y = x, y
    else:
        # Use absolute coordinates (uppercase L command)
        for x, y in points[1:]:
            path_str += f" L{int(x)},{int(y)}"
    
    if close_path:
        path_str += " z"  # Use lowercase z, shorter
    
    return path_str

def simplify_path(d_str: str, epsilon: float = 1.0, use_relative: bool = True) -> str:
    """
    Simplify single SVG path, properly handling multiple subpaths
    
    Args:
        d_str: Original path string
        epsilon: Simplification threshold
        use_relative: Whether to use relative coordinates
        
    Returns:
        Simplified path string
    """
    # Check if contains multiple subpaths (multiple M commands)
    m_count = len(re.findall(r'[Mm]', d_str))
    
    if m_count <= 1:
        # Single subpath, use original logic
        close_path = d_str.strip().upper().endswith('Z')
        points = extract_polygon_points(d_str)
        if len(points) <= 3:
            return d_str
        
        if close_path and points[0] != points[-1]:
            points.append(points[0])
        
        simplified_points = ramer_douglas_peucker(points, epsilon)
        
        if close_path and len(simplified_points) > 1 and simplified_points[0] == simplified_points[-1]:
            simplified_points = simplified_points[:-1]
        
        new_d_str = rebuild_polygon_path(simplified_points, close_path, use_relative)
        new_d_str = optimize_path_commands(new_d_str)
        return new_d_str
    
    else:
        # Multiple subpaths, process each subpath separately
        # Use regex to split subpaths
        subpaths = re.split(r'(?=[Mm])', d_str)
        subpaths = [sp.strip() for sp in subpaths if sp.strip()]
        
        simplified_subpaths = []
        
        for subpath in subpaths:
            # Recursively simplify each subpath
            simplified_subpath = simplify_path(subpath, epsilon, use_relative)
            if simplified_subpath:
                simplified_subpaths.append(simplified_subpath)
        
        # Recombine all subpaths
        result = ' '.join(simplified_subpaths)
        return optimize_path_commands(result)

def simplify_svg_polygons(svg_string: str, epsilon: float = 1.0, remove_transforms: bool = True, use_relative: bool = True, size_threshold: int = 0) -> str:
    """
    Simplify polygon paths in SVG
    
    Args:
        svg_string: Original SVG string
        epsilon: Simplification threshold
        remove_transforms: Whether to remove transforms
        use_relative: Whether to use relative coordinates
        size_threshold: SVG UTF-8 encoded length threshold, stop processing when reached, 0 means no threshold
        
    Returns:
        Simplified SVG string
    """
    try:
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        ET.register_namespace('', namespaces['svg'])
        
        root = ET.fromstring(svg_string)
        path_elements = root.findall('.//svg:path', namespaces=namespaces)
        
        # If size threshold is set, sort paths by length in descending order for processing
        if size_threshold > 0:
            # Get length information for each path
            path_info = []
            for i, path_elem in enumerate(path_elements):
                d_str = path_elem.get('d', '')
                path_info.append((i, len(d_str), path_elem))
            
            # Sort by path length in descending order
            path_info.sort(key=lambda x: x[1], reverse=True)
            
            # Process sorted paths
            for _, _, path_elem in path_info:
                d_str = path_elem.get('d')
                if not d_str:
                    continue

                transform = path_elem.get('transform')
                
                # Handle transform attribute
                if remove_transforms and transform:
                    points = extract_polygon_points(d_str)
                    # Parse translate transform
                    translate_match = re.search(r'translate\((\d+),(\d+)\)', transform)
                    if translate_match:
                        tx, ty = int(translate_match.group(1)), int(translate_match.group(2))
                        # Apply translate to all points
                        points = [(x + tx, y + ty) for x, y in points]
                        # Remove transform attribute
                        path_elem.attrib.pop('transform')
                        
                        # Rebuild path
                        close_path = d_str.strip().upper().endswith('Z')
                        d_str = rebuild_polygon_path(points, close_path, use_relative)
                
                # Simplify path
                new_d_str = simplify_path(d_str, epsilon, use_relative)
                
                # Update path's 'd' attribute
                path_elem.set('d', new_d_str)
                
                # Convert to compressed SVG string, check size
                temp_svg_str = compress_svg(ET.tostring(root, encoding='unicode'))
                current_size = len(temp_svg_str.encode('utf-8'))
                
                if current_size <= size_threshold:
                    return temp_svg_str
        else:
            # Original logic: process all paths
            for path_elem in path_elements:
                d_str = path_elem.get('d')
                if not d_str:
                    continue

                transform = path_elem.get('transform')
                
                # Handle transform attribute
                if remove_transforms and transform:
                    points = extract_polygon_points(d_str)
                    # Parse translate transform
                    translate_match = re.search(r'translate\((\d+),(\d+)\)', transform)
                    if translate_match:
                        tx, ty = int(translate_match.group(1)), int(translate_match.group(2))
                        # Apply translate to all points
                        points = [(x + tx, y + ty) for x, y in points]
                        # Remove transform attribute
                        path_elem.attrib.pop('transform')
                        
                        # Rebuild path
                        close_path = d_str.strip().upper().endswith('Z')
                        d_str = rebuild_polygon_path(points, close_path, use_relative)
                
                # Simplify path
                new_d_str = simplify_path(d_str, epsilon, use_relative)
                
                # Update path's 'd' attribute
                path_elem.set('d', new_d_str)
        
        # Convert XML to string
        svg_str = ET.tostring(root, encoding='unicode')
        
        # Use general SVG compression function to process result
        svg_str = compress_svg(svg_str)
        # print(f"Final SVG size: {len(svg_str.encode('utf-8'))} bytes")
        return svg_str
    
    except Exception as e:
        print(f"SVG simplification error: {e}")
        return svg_string

# ------------------- Main function and test functions -------------------

def simplify_svg_string(svg_string: str, threshold: float = 1.0, use_relative: bool = True, size_threshold: int = 0) -> str:
    """
    Main function for simplifying SVG polygons
    
    Args:
        svg_string: Original SVG string
        threshold: Simplification threshold
        use_relative: Whether to use relative coordinates
        size_threshold: SVG UTF-8 encoded length threshold, stop processing when reached, 0 means no threshold
        
    Returns:
        Simplified SVG string
    """
    # Get original SVG size
    original_size = len(svg_string.encode('utf-8'))
    print(f"Original SVG size: {original_size} bytes")
    
    # If already smaller than threshold, return directly
    if size_threshold > 0 and original_size <= size_threshold:
        return svg_string
    
    return simplify_svg_polygons(svg_string, epsilon=threshold, use_relative=use_relative, size_threshold=size_threshold)

def compress_svg_string(svg_string: str) -> str:
    """
    Only compress SVG string without path simplification
    
    Args:
        svg_string: Original SVG string
        
    Returns:
        Compressed SVG string
    """
    return compress_svg(svg_string)

def test_simplify_svg():
    """Test SVG path simplification functionality"""
    # Test case: An SVG string with polygon paths
    test_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
        <path d="M10,10 L20,10 L30,20 L40,20 L50,30 L60,30 L70,20 L80,20 L90,10 L100,10 Z" fill="blue" />
    </svg>'''
    
    # Test simplification effects with different thresholds and relative coordinates
    simplified_svg1 = simplify_svg_string(test_svg, threshold=1.0, use_relative=True)
    simplified_svg2 = simplify_svg_string(test_svg, threshold=5.0, use_relative=True)
    simplified_svg3 = simplify_svg_string(test_svg, threshold=10.0, use_relative=True)
    
    # Only use compression functionality
    compressed_svg = compress_svg_string(test_svg)
    
    print("Original SVG:")
    print(test_svg)
    print("\nCompression only result:")
    print(compressed_svg)
    print("\nSimplification result with threshold = 1.0 (relative coordinates):")
    print(simplified_svg1)
    print("\nSimplification result with threshold = 5.0 (relative coordinates):")
    print(simplified_svg2)
    print("\nSimplification result with threshold = 10.0 (relative coordinates):")
    print(simplified_svg3)
    
    # Test specific relative coordinate conversion example
    test_svg2 = '''<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 384 384">
        <path d="M0,0 L384,0 L384,384 L0,384 Z" fill="#DED48A"/>
    </svg>'''
    simplified_svg4 = simplify_svg_string(test_svg2, threshold=1.0, use_relative=True)
    print("\nRectangle relative coordinate test:")
    print(simplified_svg4)
    
    # Test decimal processing functionality
    test_svg4 = '''<svg xmlns="http://www.w3.org/2000/svg" width="100.0" height="100.5">
        <path d="M10.0,10.0 L20.5,10.3 L30.2,20.7" stroke-width="1.5" />
        <circle cx="50.0" cy="50.0" r="30.0" fill="#FF0000" />
        <ellipse cx="70.123" cy="70.999" rx="10.501" ry="5.499" fill="#0000FF" />
    </svg>'''
    
    print("\nTest decimal processing functionality:")
    print("Original SVG:")
    print(test_svg4)
    print("\nCompressed SVG:")
    compressed_svg3 = compress_svg_string(test_svg4)
    print(compressed_svg3)
    
    # Test size threshold functionality
    test_svg_large = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
        <path d="M10,10 L20,10 L30,20 L40,20 L50,30 L60,30 L70,20 L80,20 L90,10 L100,10 L110,20 L120,20 L130,30 L140,30 L150,20 L160,20 L170,10 L180,10 Z" fill="red" />
        <path d="M200,200 L220,200 L240,220 L260,220 L280,240 L300,240 L320,220 L340,220 L360,200 L380,200 L400,220 L420,220 L440,240 L460,240 L480,220 L500,220 L520,200 L540,200 Z" fill="green" />
        <path d="M100,300 L120,300 L140,320 L160,320 L180,340 L200,340 L220,320 L240,320 L260,300 L280,300 L300,320 L320,320 L340,340 L360,340 L380,320 L400,320 L420,300 L440,300 Z" fill="blue" />
    </svg>'''
    
    # Get original and compressed sizes
    original_size = len(test_svg_large.encode('utf-8'))
    compressed_size = len(compress_svg_string(test_svg_large).encode('utf-8'))
    
    # Calculate a threshold between the two
    middle_threshold = (original_size + compressed_size) // 2
    
    print("\nTest size threshold functionality:")
    print(f"Original SVG size: {original_size} bytes")
    print(f"Compressed SVG size: {compressed_size} bytes")
    print(f"Set threshold: {middle_threshold} bytes")
    
    # Use size threshold
    result = simplify_svg_string(test_svg_large, threshold=1.0, use_relative=True, size_threshold=middle_threshold)
    print("\nSimplification result using size threshold:")
    print(result)

# If this file is run directly, execute test
if __name__ == "__main__":
    test_simplify_svg()
