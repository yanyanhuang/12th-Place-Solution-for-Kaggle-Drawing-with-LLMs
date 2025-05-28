import re
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional

# ------------------- SVG压缩相关函数 -------------------

def optimize_path_commands(d_str: str) -> str:
    """
    优化路径命令字符串，使用更短的表示方式
    
    Args:
        d_str: 原始路径命令字符串
        
    Returns:
        优化后的路径命令字符串
    """
    # 将多个空格替换为单个空格
    d_str = re.sub(r'\s+', ' ', d_str)
    
    # 删除不必要的0，例如: 0.5 -> .5
    d_str = re.sub(r'([^0-9])0\.', r'\1.', d_str)
    
    # 删除小数点后为0的数字，例如: 10.0 -> 10
    d_str = re.sub(r'(\d+)\.0([^0-9]|$)', r'\1\2', d_str)
    
    # 负数前面可以用-而不是0-
    d_str = re.sub(r' 0-', ' -', d_str)
    
    return d_str.strip()

def shorten_color(match) -> str:
    """缩短十六进制颜色代码，例如 #AABBCC -> #ABC"""
    color = match.group(1)
    if len(color) == 7 and color[1] == color[2] and color[3] == color[4] and color[5] == color[6]:
        return f'"#{color[1]}{color[3]}{color[5]}"'
    return match.group(0)

def process_numeric_attributes(svg_str: str) -> str:
    """
    处理SVG字符串中的数值属性，移除不必要的小数点和零
    
    Args:
        svg_str: 原始SVG字符串
        
    Returns:
        处理后的SVG字符串
    """
    # 处理d属性中的小数点
    pattern_d = r'd="([^"]*)"'
    for match in re.finditer(pattern_d, svg_str):
        d_attr = match.group(1)
        # 优化浮点数表示：如果小数部分为0，则移除小数点后部分
        new_d_attr = re.sub(r'(\d+)\.0+([^0-9]|$)', r'\1\2', d_attr)
        # 对于接近整数的浮点数（小数部分<0.001），转换为整数
        new_d_attr = re.sub(r'(\d+\.\d{1,3})([^0-9]|$)', 
                           lambda m: str(int(float(m.group(1)))) + m.group(2) 
                           if abs(float(m.group(1)) - int(float(m.group(1)))) < 0.001 
                           else m.group(0), new_d_attr)
        # 处理括号后的负数
        new_d_attr = re.sub(r'([,\(])0-', r'\1-', new_d_attr)
        svg_str = svg_str.replace(f'd="{d_attr}"', f'd="{new_d_attr}"')
    
    # 处理其他可能的浮点数（如宽度、高度等）
    # 1. 替换.0结尾的数字
    svg_str = re.sub(r'(\d+)\.0+(?=[\s"\'])', r'\1', svg_str)
    # 2. 对于重要属性中的浮点数，只有在接近整数时才转换为整数
    svg_str = re.sub(r'(width|height|x|y|cx|cy|r|rx|ry)="(\d+\.\d+)"', 
                    lambda m: f'{m.group(1)}="{int(float(m.group(2)))}"' 
                    if abs(float(m.group(2)) - int(float(m.group(2)))) < 0.001 
                    else m.group(0), svg_str)
    
    return svg_str

def compress_whitespace(svg_str: str) -> str:
    """
    压缩SVG字符串中的空白字符
    
    Args:
        svg_str: 原始SVG字符串
        
    Returns:
        压缩后的SVG字符串
    """
    # 删除多余的空格
    compressed = re.sub(r'\s+', ' ', svg_str)
    compressed = re.sub(r'>\s+<', '><', compressed)
    compressed = re.sub(r'\s+/>', '/>', compressed)
    
    return compressed.strip()

def remove_commas_from_path(d_str: str) -> str:
    """
    移除路径中不必要的逗号，使用空格替代
    例如: M0,0 L23,0 -> M0 0L23 0
    
    Args:
        d_str: 原始路径命令字符串
        
    Returns:
        优化后的路径命令字符串
    """
    # 将逗号替换为空格
    d_str = re.sub(r',', ' ', d_str)
    
    # 移除命令字母后的空格 (如 "M 0" -> "M0")
    d_str = re.sub(r'([MLHVCSQTAZmlhvcsqtaz]) ', r'\1', d_str)
    
    return d_str

def optimize_consecutive_commands(d_str: str) -> str:
    """
    优化连续相同的命令，省略重复命令字母
    例如: L10 0 L20 10 -> L10 0 20 10
    
    Args:
        d_str: 原始路径命令字符串
        
    Returns:
        优化后的路径命令字符串
    """
    # 匹配所有SVG路径命令和它们的参数
    pattern = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
    matches = re.findall(pattern, d_str)
    
    if not matches:
        return d_str
        
    result = []
    prev_cmd = None
    
    for cmd, params in matches:
        # 如果命令与前一个相同，只添加参数
        if cmd == prev_cmd:
            result.append(params)
        else:
            result.append(cmd + params)
            prev_cmd = cmd
    
    return ''.join(result)

def apply_transform_to_path(d_str: str, transform: str) -> str:
    """
    将变换直接应用到路径坐标上，正确处理多个子路径
    目前仅支持translate变换
    
    Args:
        d_str: 原始路径命令字符串
        transform: SVG变换字符串
        
    Returns:
        应用变换后的路径命令字符串
    """
    # 解析translate transform
    translate_match = re.search(r'translate\(([^,]+),([^)]+)\)', transform)
    if not translate_match:
        return d_str
        
    tx = int(float(translate_match.group(1)))
    ty = int(float(translate_match.group(2)))
    
    # 检查是否包含多个子路径（多个M命令）
    m_count = len(re.findall(r'[Mm]', d_str))
    
    if m_count <= 1:
        # 单一子路径，使用原有逻辑
        points = extract_polygon_points(d_str)
        if not points:
            return d_str
            
        # 将translate应用到所有点
        points = [(x + tx, y + ty) for x, y in points]
        
        # 判断路径是否闭合
        close_path = d_str.strip().upper().endswith('Z')
        
        # 重建路径，使用相对坐标
        return rebuild_polygon_path(points, close_path, use_relative=True)
    else:
        # 多个子路径，分别处理每个子路径
        # 使用正则表达式分割子路径
        subpaths = re.split(r'(?=[Mm])', d_str)
        subpaths = [sp.strip() for sp in subpaths if sp.strip()]
        
        transformed_subpaths = []
        
        for subpath in subpaths:
            # 对每个子路径应用变换
            points = extract_polygon_points(subpath)
            if not points:
                continue
                
            # 将translate应用到所有点
            points = [(x + tx, y + ty) for x, y in points]
            
            # 判断子路径是否闭合
            close_path = subpath.strip().upper().endswith('Z')
            
            # 重建子路径，使用相对坐标
            transformed_subpath = rebuild_polygon_path(points, close_path, use_relative=True)
            if transformed_subpath:
                transformed_subpaths.append(transformed_subpath)
        
        # 重新组合所有子路径
        return ' '.join(transformed_subpaths)

def compress_svg(svg_string: str) -> str:
    """
    压缩SVG字符串大小的主函数，可独立使用
    
    Args:
        svg_string: 原始SVG字符串
        
    Returns:
        压缩后的SVG字符串
    """
    # 压缩空白字符
    compressed = compress_whitespace(svg_string)
    
    # 压缩颜色代码
    compressed = re.sub(r'"(#[0-9A-Fa-f]{6})"', shorten_color, compressed)
    
    # 处理数值属性
    compressed = process_numeric_attributes(compressed)
    
    # 压缩viewBox属性
    compressed = re.sub(r'viewBox="([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)"', r'viewBox="\1 \2 \3 \4"', compressed)
    
    # 处理SVG路径
    try:
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        ET.register_namespace('', namespaces['svg'])
        
        root = ET.fromstring(compressed)
        path_elements = root.findall('.//svg:path', namespaces=namespaces)
        
        for path_elem in path_elements:
            d_str = path_elem.get('d')
            if not d_str:
                continue
                
            # 获取transform属性
            transform = path_elem.get('transform')
            
            # 如果有transform，将其应用到路径中
            if transform:
                d_str = apply_transform_to_path(d_str, transform)
                # 移除transform属性
                path_elem.attrib.pop('transform', None)
            
            # 移除逗号
            d_str = remove_commas_from_path(d_str)
            
            # 优化连续命令
            d_str = optimize_consecutive_commands(d_str)
            
            # 通用路径命令优化
            d_str = optimize_path_commands(d_str)
            
            # 更新路径
            path_elem.set('d', d_str)
        
        # 将XML转换为字符串
        compressed = ET.tostring(root, encoding='unicode')
        
    except Exception as e:
        print(f"路径压缩错误: {e}")
    
    return compressed

# ------------------- 路径简化相关函数 -------------------

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """计算两点之间的欧几里得距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def perpendicular_distance(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
    """计算点到直线的垂直距离"""
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
    """使用Ramer-Douglas-Peucker算法简化多边形"""
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
    """从多边形路径中提取点坐标，转换为整数"""
    # 匹配命令和坐标
    pattern = r'([MLZmlz])|([-+]?\d*\.\d+|[-+]?\d+)'
    tokens = re.findall(pattern, d_str)
    
    points = []
    current_cmd = None
    current_x, current_y = 0, 0
    coords = []
    
    for cmd, param in tokens:
        if cmd:  # 找到命令
            current_cmd = cmd
            coords = []
        elif param and current_cmd in 'MLml':  # 找到坐标
            coords.append(int(float(param)))  # 确保转换为整数
            if len(coords) == 2:
                if current_cmd in 'ml':  # 相对坐标
                    current_x += coords[0]
                    current_y += coords[1]
                else:  # 绝对坐标
                    current_x, current_y = coords
                points.append((current_x, current_y))
                coords = []
    
    return points

def rebuild_polygon_path(points: List[Tuple[int, int]], close_path: bool = True, use_relative: bool = True) -> str:
    """从点列表重建多边形路径，使用相对坐标减小SVG大小"""
    if not points:
        return ""
    
    # 第一个点必须使用绝对坐标
    path_str = f"M{int(points[0][0])},{int(points[0][1])}"
    
    if use_relative:
        # 使用相对坐标(小写l命令)
        prev_x, prev_y = points[0]
        for x, y in points[1:]:
            dx, dy = x - prev_x, y - prev_y
            path_str += f" l{int(dx)},{int(dy)}"
            prev_x, prev_y = x, y
    else:
        # 使用绝对坐标(大写L命令)
        for x, y in points[1:]:
            path_str += f" L{int(x)},{int(y)}"
    
    if close_path:
        path_str += " z"  # 使用小写z，更短
    
    return path_str

def simplify_path(d_str: str, epsilon: float = 1.0, use_relative: bool = True) -> str:
    """
    简化单个SVG路径，正确处理多个子路径
    
    Args:
        d_str: 原始路径字符串
        epsilon: 简化阈值
        use_relative: 是否使用相对坐标
        
    Returns:
        简化后的路径字符串
    """
    # 检查是否包含多个子路径（多个M命令）
    m_count = len(re.findall(r'[Mm]', d_str))
    
    if m_count <= 1:
        # 单一子路径，使用原有逻辑
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
        # 多个子路径，分别处理每个子路径
        # 使用正则表达式分割子路径
        subpaths = re.split(r'(?=[Mm])', d_str)
        subpaths = [sp.strip() for sp in subpaths if sp.strip()]
        
        simplified_subpaths = []
        
        for subpath in subpaths:
            # 递归简化每个子路径
            simplified_subpath = simplify_path(subpath, epsilon, use_relative)
            if simplified_subpath:
                simplified_subpaths.append(simplified_subpath)
        
        # 重新组合所有子路径
        result = ' '.join(simplified_subpaths)
        return optimize_path_commands(result)

def simplify_svg_polygons(svg_string: str, epsilon: float = 1.0, remove_transforms: bool = True, use_relative: bool = True, size_threshold: int = 0) -> str:
    """
    简化SVG中的多边形路径
    
    Args:
        svg_string: 原始SVG字符串
        epsilon: 简化阈值
        remove_transforms: 是否移除变换
        use_relative: 是否使用相对坐标
        size_threshold: SVG UTF-8编码长度阈值，达到此阈值时停止处理，0表示不使用阈值
        
    Returns:
        简化后的SVG字符串
    """
    try:
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        ET.register_namespace('', namespaces['svg'])
        
        root = ET.fromstring(svg_string)
        path_elements = root.findall('.//svg:path', namespaces=namespaces)
        
        # 如果设置了大小阈值，按路径长度降序排序处理
        if size_threshold > 0:
            # 获取每条路径的长度信息
            path_info = []
            for i, path_elem in enumerate(path_elements):
                d_str = path_elem.get('d', '')
                path_info.append((i, len(d_str), path_elem))
            
            # 按路径长度降序排序
            path_info.sort(key=lambda x: x[1], reverse=True)
            
            # 处理排序后的路径
            for _, _, path_elem in path_info:
                d_str = path_elem.get('d')
                if not d_str:
                    continue

                transform = path_elem.get('transform')
                
                # 处理transform属性
                if remove_transforms and transform:
                    points = extract_polygon_points(d_str)
                    # 解析translate transform
                    translate_match = re.search(r'translate\((\d+),(\d+)\)', transform)
                    if translate_match:
                        tx, ty = int(translate_match.group(1)), int(translate_match.group(2))
                        # 将translate应用到所有点
                        points = [(x + tx, y + ty) for x, y in points]
                        # 移除transform属性
                        path_elem.attrib.pop('transform')
                        
                        # 重建路径
                        close_path = d_str.strip().upper().endswith('Z')
                        d_str = rebuild_polygon_path(points, close_path, use_relative)
                
                # 简化路径
                new_d_str = simplify_path(d_str, epsilon, use_relative)
                
                # 更新路径的'd'属性
                path_elem.set('d', new_d_str)
                
                # 转换为压缩后的SVG字符串，检查大小
                temp_svg_str = compress_svg(ET.tostring(root, encoding='unicode'))
                current_size = len(temp_svg_str.encode('utf-8'))
                
                if current_size <= size_threshold:
                    return temp_svg_str
        else:
            # 原始逻辑：处理所有路径
            for path_elem in path_elements:
                d_str = path_elem.get('d')
                if not d_str:
                    continue

                transform = path_elem.get('transform')
                
                # 处理transform属性
                if remove_transforms and transform:
                    points = extract_polygon_points(d_str)
                    # 解析translate transform
                    translate_match = re.search(r'translate\((\d+),(\d+)\)', transform)
                    if translate_match:
                        tx, ty = int(translate_match.group(1)), int(translate_match.group(2))
                        # 将translate应用到所有点
                        points = [(x + tx, y + ty) for x, y in points]
                        # 移除transform属性
                        path_elem.attrib.pop('transform')
                        
                        # 重建路径
                        close_path = d_str.strip().upper().endswith('Z')
                        d_str = rebuild_polygon_path(points, close_path, use_relative)
                
                # 简化路径
                new_d_str = simplify_path(d_str, epsilon, use_relative)
                
                # 更新路径的'd'属性
                path_elem.set('d', new_d_str)
        
        # 将XML转换为字符串
        svg_str = ET.tostring(root, encoding='unicode')
        
        # 使用通用SVG压缩函数处理结果
        svg_str = compress_svg(svg_str)
        # print(f"最终SVG大小: {len(svg_str.encode('utf-8'))} 字节")
        return svg_str
    
    except Exception as e:
        print(f"SVG简化错误: {e}")
        return svg_string

# ------------------- 主函数与测试函数 -------------------

def simplify_svg_string(svg_string: str, threshold: float = 1.0, use_relative: bool = True, size_threshold: int = 0) -> str:
    """
    简化SVG多边形的主函数
    
    Args:
        svg_string: 原始SVG字符串
        threshold: 简化阈值
        use_relative: 是否使用相对坐标
        size_threshold: SVG UTF-8编码长度阈值，达到此阈值时停止处理，0表示不使用阈值
        
    Returns:
        简化后的SVG字符串
    """
    # 获取原始SVG大小
    original_size = len(svg_string.encode('utf-8'))
    print(f"原始SVG大小: {original_size} 字节")
    
    # 如果已经小于阈值，直接返回
    if size_threshold > 0 and original_size <= size_threshold:
        return svg_string
    
    return simplify_svg_polygons(svg_string, epsilon=threshold, use_relative=use_relative, size_threshold=size_threshold)

def compress_svg_string(svg_string: str) -> str:
    """
    仅压缩SVG字符串，不进行路径简化
    
    Args:
        svg_string: 原始SVG字符串
        
    Returns:
        压缩后的SVG字符串
    """
    return compress_svg(svg_string)

def test_simplify_svg():
    """测试SVG路径简化功能"""
    # 测试用例：一个带有多边形路径的SVG字符串
    test_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
        <path d="M10,10 L20,10 L30,20 L40,20 L50,30 L60,30 L70,20 L80,20 L90,10 L100,10 Z" fill="blue" />
    </svg>'''
    
    # 使用不同阈值和相对坐标测试简化效果
    simplified_svg1 = simplify_svg_string(test_svg, threshold=1.0, use_relative=True)
    simplified_svg2 = simplify_svg_string(test_svg, threshold=5.0, use_relative=True)
    simplified_svg3 = simplify_svg_string(test_svg, threshold=10.0, use_relative=True)
    
    # 仅使用压缩功能
    compressed_svg = compress_svg_string(test_svg)
    
    print("原始SVG:")
    print(test_svg)
    print("\n仅压缩结果:")
    print(compressed_svg)
    print("\n阈值 = 1.0 的简化结果 (相对坐标):")
    print(simplified_svg1)
    print("\n阈值 = 5.0 的简化结果 (相对坐标):")
    print(simplified_svg2)
    print("\n阈值 = 10.0 的简化结果 (相对坐标):")
    print(simplified_svg3)
    
    # 测试具体的相对坐标转换示例
    test_svg2 = '''<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 384 384">
        <path d="M0,0 L384,0 L384,384 L0,384 Z" fill="#DED48A"/>
    </svg>'''
    simplified_svg4 = simplify_svg_string(test_svg2, threshold=1.0, use_relative=True)
    print("\n矩形相对坐标测试:")
    print(simplified_svg4)
    
    # 测试小数处理功能
    test_svg4 = '''<svg xmlns="http://www.w3.org/2000/svg" width="100.0" height="100.5">
        <path d="M10.0,10.0 L20.5,10.3 L30.2,20.7" stroke-width="1.5" />
        <circle cx="50.0" cy="50.0" r="30.0" fill="#FF0000" />
        <ellipse cx="70.123" cy="70.999" rx="10.501" ry="5.499" fill="#0000FF" />
    </svg>'''
    
    print("\n测试小数处理功能:")
    print("原始SVG:")
    print(test_svg4)
    print("\n压缩后的SVG:")
    compressed_svg3 = compress_svg_string(test_svg4)
    print(compressed_svg3)
    
    # 测试大小阈值功能
    test_svg_large = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
        <path d="M10,10 L20,10 L30,20 L40,20 L50,30 L60,30 L70,20 L80,20 L90,10 L100,10 L110,20 L120,20 L130,30 L140,30 L150,20 L160,20 L170,10 L180,10 Z" fill="red" />
        <path d="M200,200 L220,200 L240,220 L260,220 L280,240 L300,240 L320,220 L340,220 L360,200 L380,200 L400,220 L420,220 L440,240 L460,240 L480,220 L500,220 L520,200 L540,200 Z" fill="green" />
        <path d="M100,300 L120,300 L140,320 L160,320 L180,340 L200,340 L220,320 L240,320 L260,300 L280,300 L300,320 L320,320 L340,340 L360,340 L380,320 L400,320 L420,300 L440,300 Z" fill="blue" />
    </svg>'''
    
    # 获取原始和压缩大小
    original_size = len(test_svg_large.encode('utf-8'))
    compressed_size = len(compress_svg_string(test_svg_large).encode('utf-8'))
    
    # 计算一个介于两者之间的阈值
    middle_threshold = (original_size + compressed_size) // 2
    
    print("\n测试大小阈值功能:")
    print(f"原始SVG大小: {original_size} 字节")
    print(f"压缩后SVG大小: {compressed_size} 字节")
    print(f"设置阈值: {middle_threshold} 字节")
    
    # 使用大小阈值
    result = simplify_svg_string(test_svg_large, threshold=1.0, use_relative=True, size_threshold=middle_threshold)
    print("\n使用大小阈值的简化结果:")
    print(result)

# 如果直接运行此文件，则执行测试
if __name__ == "__main__":
    test_simplify_svg()
