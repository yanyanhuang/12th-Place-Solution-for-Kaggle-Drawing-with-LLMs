from PIL import Image
import torch
import re
import base64
from io import BytesIO
import vtracer
import numpy as np
from lxml import etree
import random
from dataclasses import dataclass, field
from .svg_constraints import SVGConstraints
from .svg_image_fidelity import AestheticEvaluator, VQAEvaluator, ImageProcessor, svg_to_png, harmonic_mean

constraints = SVGConstraints()

def add_ocr_decoy_svg(svg_code: str) -> str:
    """
    Adds nested circles with second darkest and second brightest colors from the existing SVG,
    positioned in one of the four corners (randomly selected) but positioned to avoid being
    cropped out during image processing.
    
    Parameters:
    -----------
    svg_code : str
        The original SVG string
    
    Returns:
    --------
    str
        Modified SVG with the nested circles added
    """
    import random
    import re
    from colorsys import rgb_to_hls, hls_to_rgb
    
    # Check if SVG has a closing tag
    if "</svg>" not in svg_code:
        return svg_code
    
    # Extract viewBox if it exists to understand the dimensions
    viewbox_match = re.search(r'viewBox=["\'](.*?)["\']', svg_code)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        try:
            x, y, width, height = map(float, viewbox)
        except ValueError:
            # Default dimensions if we can't parse viewBox
            width, height = 384, 384
    else:
        # Default dimensions if viewBox not found
        width, height = 384, 384
    
    # Function to convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Function to convert RGB to hex
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
    
    # Function to calculate color lightness
    def get_lightness(color):
        # Handle different color formats
        if color.startswith('#'):
            rgb = hex_to_rgb(color)
            return rgb_to_hls(*rgb)[1]  # Lightness is the second value in HLS
        elif color.startswith('rgb'):
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if rgb_match:
                r, g, b = map(lambda x: int(x)/255, rgb_match.groups())
                return rgb_to_hls(r, g, b)[1]
        return 0.5  # Default lightness if we can't parse
    
    # Extract all colors from the SVG
    color_matches = re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))"', svg_code)
    
    # Default colors in case we don't find enough
    second_darkest_color = "#333333"  # Default to dark gray
    second_brightest_color = "#CCCCCC"  # Default to light gray
    
    if color_matches:
        # Remove duplicates and get unique colors
        unique_colors = list(set(color_matches))
        
        # Calculate lightness for each unique color
        colors_with_lightness = [(color, get_lightness(color)) for color in unique_colors]
        
        # Sort by lightness (brightness)
        sorted_colors = sorted(colors_with_lightness, key=lambda x: x[1])
        
        # Handle different scenarios based on number of unique colors
        if len(sorted_colors) >= 4:
            # We have at least 4 unique colors - use 2nd darkest and 2nd brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[-2][0]
        elif len(sorted_colors) == 3:
            # We have 3 unique colors - use 2nd darkest and brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[2][0]
        elif len(sorted_colors) == 2:
            # We have only 2 unique colors - use the darkest and brightest
            second_darkest_color = sorted_colors[0][0]
            second_brightest_color = sorted_colors[1][0]
        elif len(sorted_colors) == 1:
            # Only one color - use it for second_darkest and a derived lighter version
            base_color = sorted_colors[0][0]
            base_lightness = sorted_colors[0][1]
            second_darkest_color = base_color
            
            # Create a lighter color variant if the base is dark, or darker if base is light
            if base_lightness < 0.5:
                # Base is dark, create lighter variant
                second_brightest_color = "#CCCCCC"
            else:
                # Base is light, create darker variant
                second_darkest_color = "#333333"
    
    # Ensure the colors are different
    if second_darkest_color == second_brightest_color:
        # If they ended up the same, modify one of them
        if get_lightness(second_darkest_color) < 0.5:
            # It's a dark color, make the bright one lighter
            second_brightest_color = "#CCCCCC"
        else:
            # It's a light color, make the dark one darker
            second_darkest_color = "#333333"
    
    # Base size for the outer circle
    base_outer_radius = width * 0.023
    
    # Randomize size by ±10%
    size_variation = base_outer_radius * 0.1
    outer_radius = base_outer_radius + random.uniform(-size_variation, size_variation)
    
    # Define radii for inner circles based on outer radius
    middle_radius = outer_radius * 0.80
    inner_radius = middle_radius * 0.65
    
    # Calculate the maximum crop margin based on the image processing (5% of dimensions)
    # Add 20% extra margin for safety
    crop_margin_w = int(width * 0.05 * 1.2)
    crop_margin_h = int(height * 0.05 * 1.2)
    
    # Calculate center point based on the outer radius to ensure the entire circle stays visible
    safe_offset = outer_radius + max(crop_margin_w, crop_margin_h)
    
    # Choose a random corner (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    corner = random.randint(0, 3)
    
    # Position the circle in the chosen corner, accounting for crop margin
    if corner == 0:  # Top-left
        center_x = safe_offset
        center_y = safe_offset
    elif corner == 1:  # Top-right
        center_x = width - safe_offset
        center_y = safe_offset
    elif corner == 2:  # Bottom-left
        center_x = safe_offset
        center_y = height - safe_offset
    else:  # Bottom-right
        center_x = width - safe_offset
        center_y = height - safe_offset
    
    # Add a small random offset (±10% of safe_offset) to make positioning less predictable
    random_offset = safe_offset * 0.1
    center_x += random.uniform(-random_offset, random_offset)
    center_y += random.uniform(-random_offset, random_offset)
    
    # Round to 1 decimal place to keep file size down
    outer_radius = round(outer_radius, 1)
    middle_radius = round(middle_radius, 1)
    inner_radius = round(inner_radius, 1)
    center_x = round(center_x, 1)
    center_y = round(center_y, 1)
    
    # Create the nested circles
    outer_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{outer_radius}" fill="{second_darkest_color}" />'
    middle_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{middle_radius}" fill="{second_brightest_color}" />'
    inner_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{inner_radius}" fill="{second_darkest_color}" />'
    
    # Create a group element that contains all three circles
    group_element = f'<g>{outer_circle}{middle_circle}{inner_circle}</g>'
    
    # Insert the group element just before the closing SVG tag
    modified_svg = svg_code.replace("</svg>", f"{group_element}</svg>")
    
    # Calculate and add a comment with the byte size information
    outer_bytes = len(outer_circle.encode('utf-8'))
    middle_bytes = len(middle_circle.encode('utf-8'))
    inner_bytes = len(inner_circle.encode('utf-8'))
    total_bytes = outer_bytes + middle_bytes + inner_bytes
    
    corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    byte_info = f'<!-- Circle bytes: outer={outer_bytes}, middle={middle_bytes}, ' \
                f'inner={inner_bytes}, total={total_bytes}, ' \
                f'colors: dark={second_darkest_color}, light={second_brightest_color}, ' \
                f'position: {corner_names[corner]} -->'
    
    modified_svg = modified_svg.replace("</svg>", f"{byte_info}</svg>")
    
    return modified_svg

def enforce_constraints(svg_string: str) -> str:
    """Enforces constraints on an SVG string, removing disallowed elements
    and attributes.

    Parameters
    ----------
    svg_string : str
        The SVG string to process.

    Returns
    -------
    str
        The processed SVG string, or the default SVG if constraints
        cannot be satisfied.
    """
    default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""

    try:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root = etree.fromstring(svg_string, parser=parser)
    except etree.ParseError as e:
        return default_svg

    elements_to_remove = []
    for element in root.iter():
        tag_name = etree.QName(element.tag).localname

        # Remove disallowed elements
        if tag_name not in constraints.allowed_elements:
            elements_to_remove.append(element)
            continue  # Skip attribute checks for removed elements

        # Remove disallowed attributes
        attrs_to_remove = []
        for attr in element.attrib:
            attr_name = etree.QName(attr).localname
            if (
                attr_name
                not in constraints.allowed_elements[tag_name]
                and attr_name
                not in constraints.allowed_elements['common']
            ):
                attrs_to_remove.append(attr)

        for attr in attrs_to_remove:
            del element.attrib[attr]

        # Check and remove invalid href attributes
        for attr, value in element.attrib.items():
             if etree.QName(attr).localname == 'href' and not value.startswith('#'):
                del element.attrib[attr]

        # Validate path elements to help ensure SVG conversion
        if tag_name == 'path':
            d_attribute = element.get('d')
            if not d_attribute:
                elements_to_remove.append(element)
                continue # Skip further checks for this removed element
            # Use regex to validate 'd' attribute format
            path_regex = re.compile(
                r'^'  # Start of string
                r'(?:'  # Non-capturing group for each command + numbers block
                r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                r'\s*'  # Optional whitespace after command
                r'(?:'  # Non-capturing group for optional numbers
                r'-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?'  # First number
                r'(?:[\s,]+-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)*'  # Subsequent numbers with mandatory separator(s)
                r')?'  # Numbers are optional (e.g. for Z command)
                r'\s*'  # Optional whitespace after numbers/command block
                r')+'  # One or more command blocks
                r'\s*'  # Optional trailing whitespace
                r'$'  # End of string
            )
            if not path_regex.match(d_attribute):
                elements_to_remove.append(element)
                continue
    
    # Remove elements marked for removal
    for element in elements_to_remove:
        if element.getparent() is not None:
            element.getparent().remove(element)

    try:
        cleaned_svg_string = etree.tostring(root, encoding='unicode')
        return cleaned_svg_string
    except ValueError as e:
        return default_svg

from .simplify_path import simplify_svg_polygons, compress_svg
# from simplify_path import advanced_optimize_svg

def vtracer_convert_image_to_svg(image, filter_speckle=4, filter_speckle_step=4, color_precision=4, max_size_bytes=10000, svg_header=None, mark=None):
    pixels = list(image.convert('RGBA').getdata())
    # svg = vtracer.convert_pixels_to_svg(pixels, size=image.size, mode='polygon', filter_speckle=filter_speckle, color_precision=color_precision)
    svg = vtracer.convert_pixels_to_svg(pixels, size=image.size, mode='polygon', layer_difference=10,  filter_speckle=filter_speckle, color_precision=color_precision)
    svg = svg[svg.find("<svg"):]
    svg = svg_header + svg[svg.find('>')+1:]
    # svg = svg.replace('\n</svg>', mark + '\n</svg>')
    svg = add_ocr_decoy_svg(svg)
    svg = svg.replace('\n', '')
    svg = compress_svg(svg)
    # print(len(svg.encode('utf-8')))
    while len(svg.encode('utf-8')) >= max_size_bytes:
        filter_speckle += filter_speckle_step
        svg, filter_speckle = vtracer_convert_image_to_svg(image, filter_speckle=filter_speckle, filter_speckle_step=filter_speckle_step, color_precision=color_precision, max_size_bytes=max_size_bytes, svg_header=svg_header, mark=mark)
        svg = compress_svg(svg)
    
    return svg, filter_speckle

def compress_hex_color(hex_color):
    """Convert hex color to shortest possible representation"""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    return hex_color

def bitmap_to_svg_vtracer(image, max_size_bytes=10000, resize=True, target_size=(384, 384)):
    """
    Convert bitmap to SVG using layered feature extraction with optimized space usage
    
    Args:
        image: Input image (PIL.Image)
        max_size_bytes (int): Maximum SVG size
        resize (bool): Whether to resize the image before processing
        target_size (tuple): Target size for resizing (width, height)
        adaptive_fill (bool): Whether to adaptively fill available space
        num_colors (int): Number of colors to quantize, if None uses adaptive selection
    
    Returns:
        str: SVG representation
    """
    
    # Start building SVG
    # Use original dimensions in viewBox for proper scaling when displayed
    # svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'
    # svg_base = svg_header + svg_bg
    # svg_footer = '</svg>'
    # Convert to numpy array
    original_size = image.size
    orig_width, orig_height = original_size
    image = image.resize(target_size, Image.LANCZOS)
    
    img_np = np.array(image)
    height, width = img_np.shape[:2]
    
    # Calculate average background color
    # print(img_np.shape)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = np.mean(img_np, axis=(0,1)).astype(int)
        bg_hex_color = compress_hex_color(f'#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}')
    else:
        bg_hex_color = '#fff'

    # mark = '<path d="M 10 40 L 25 10 L 40 40 M 17.5 25 L 32.5 25" fill="none" stroke="black" stroke-width="4" />'
    # mark = '<circle cx="25" cy="25" r="16" fill="#000000" /><circle cx="25" cy="25" r="12" fill="#FFFFFF" />'
    # mark = '<circle cx="40" cy="40" r="20" fill="#000000" />'
    # mark = '<rect x="5" y="10" width="20" height="4" fill="#000000"/><rect x="10" y="5" width="4" height="20" fill="#000000"/>'
    # mark = f'<g fill="{bg_hex_color}"><rect x="15" y="20" width="20" height="4"/><rect x="23" y="12" width="4" height="20"/><rect x="349" y="20" width="20" height="4"/><rect x="357" y="12" width="4" height="20"/></g>'
    # mark = f'<rect x="15" y="20" width="20" height="4" fill="#000000"/><rect x="23" y="12" width="4" height="20" fill="#000000"/><rect x="349" y="20" width="20" height="4" fill="#000000"/><rect x="357" y="12" width="4" height="20" fill="#000000"/>'
    mark = '<path d="M 186 10 L 199 10 L 186 25 L 199 25" fill="none" stroke="#000000" stroke-width="3"/>'
    # mark = f'<path d="M 10 10 L 30 30 M 30 10 L 10 30" stroke="{bg_hex_color}" stroke-width="3" fill="none" />'
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">\n'
    
    # try:
    # filter_speckle = 4
    # pixels = list(image.convert('RGBA').getdata())
    # svg = vtracer.convert_pixels_to_svg(pixels, size=image.size, mode='polygon', filter_speckle=filter_speckle, color_precision=4)
    # svg = svg[svg.find("<svg"):]
    # svg = svg_header + svg[svg.find('>')+1:]
    # svg = svg.replace('\n</svg>', mark + '\n</svg>')
    # svg = svg.replace('\n', '')
    # print(len(svg.encode('utf-8')))

    # while len(svg.encode('utf-8')) >= 1.5*max_size_bytes:
    #     filter_speckle += 4
    #     pixels = list(image.convert('RGBA').getdata())
    #     svg = vtracer.convert_pixels_to_svg(pixels, size=image.size, mode='polygon', filter_speckle=filter_speckle, color_precision=4)
    #     svg = svg[svg.find("<svg"):]
    #     svg = svg_header + svg[svg.find('>')+1:]
    #     svg = svg.replace('\n</svg>', mark + '\n</svg>')
    #     svg = svg.replace('\n', '')
    # print(len(svg.encode('utf-8')))
    svg, filter_speckle = vtracer_convert_image_to_svg(image, filter_speckle=4, filter_speckle_step=4, color_precision=6, max_size_bytes=1.5*max_size_bytes, svg_header=svg_header, mark=mark)
    # print(len(svg.encode('utf-8')))
    len_svg_min = len(svg.encode('utf-8'))
    unchanged_count = 0
    threshold = 0
    while len(svg.encode('utf-8')) >= max_size_bytes:
        threshold += 0.5
        svg = simplify_svg_polygons(svg, epsilon=threshold, size_threshold=0)
        # svg = advanced_optimize_svg(svg, threshold=threshold)
        len_svg = len(svg.encode('utf-8'))
        # print('simplify:', len_svg)

        if len_svg < len_svg_min:
            len_svg_min = len_svg
            unchanged_count = 0
        else:
            unchanged_count += 1
            
        if unchanged_count >= 10:
            svg, filter_speckle = vtracer_convert_image_to_svg(image, filter_speckle=filter_speckle, filter_speckle_step=4, color_precision=6, max_size_bytes=max_size_bytes, svg_header=svg_header, mark=mark)

    # except:
    #     svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'

    # try:
    # svg = enforce_constraints(svg)
        # constraints.validate_svg(svg)
    # except:
    #     svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'
    
    
    return svg


class SVGScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # self.evaluator = score_instance_impl
        self.aesthetic_evaluator = AestheticEvaluator(device=device)
        self.vqa_evaluator = VQAEvaluator(device=device)
        self.random_seed = 42
        
    @torch.no_grad()
    def __call__(self, images, multiple_choice_qas):

        svgs = [bitmap_to_svg_vtracer(image) for image in images]
        # instance_score, vqa_score, ocr_score, aesthetic_score = self.evaluator(multiple_choice_qas, svgs)
        rng = np.random.RandomState(self.random_seed)
        group_seed = rng.randint(0, np.iinfo(np.int32).max)
        results = []
        for svg, multiple_choice_qa in zip(svgs, multiple_choice_qas, strict=True):
            image_processor = ImageProcessor(image=svg_to_png(svg), seed=group_seed).apply()
            image = image_processor.image.copy()
            questions = multiple_choice_qa['question']
            choices = multiple_choice_qa['choices']
            answers = multiple_choice_qa['answer']
            aesthetic_score = self.aesthetic_evaluator.score(image)
            vqa_score = self.vqa_evaluator.score(questions, choices, answers, image)
            image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
            # ocr_score = self.vqa_evaluator.ocr(image_processor.image)
            # instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
            # print('scores:', vqa_score, aesthetic_score)
            instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5)
            results.append(instance_score)

        # results = []
        # score_df = []
        # for one_svg, one_multiple_choice_qa in zip(svgs, multiple_choice_qas, strict=True):
        #     instance_score, vqa_score, ocr_score, aesthetic_score = self.evaluator(one_multiple_choice_qa, one_svg, random_seed=42)
        #     results.append(instance_score)
        #     score_df.append([instance_score, vqa_score, ocr_score, aesthetic_score])
        rewards = results
        return rewards


# # Usage example
# def main():
#     scorer = QwenVLScorer(
#         device="cuda",
#         dtype=torch.bfloat16
#     )
#     images=[
#     "nasa.jpg",
#     ]
#     pil_images = [Image.open(img) for img in images]
#     prompts=[
#         'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
#     ]

#     print(scorer(None, pil_images))

# if __name__ == "__main__":
#     main()