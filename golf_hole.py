import xml.etree.ElementTree as ET
import math
import matplotlib.path as mpath
from pyproj import Geod


class GolfHole:
    def __init__(self, kml_file_path):
        # Load KML content
        with open(kml_file_path, 'r') as file:
            kml_content = file.read()

        # Initialize reference point for translation
        reference_point = self.extract_named_point_coordinates(kml_content, "reference")
        flag_reference_point = self.extract_named_point_coordinates(kml_content, "flag")

        # Extract and translate coordinates for each element
        # Store both Path objects and raw coordinates
        self.fairways_coordinates = [self.translate_coordinates(coords, reference_point, flag_reference_point) for coords in self.extract_multiple_coordinates(kml_content, "fairway")]
        self.fairways_paths = [mpath.Path(coords) for coords in self.fairways_coordinates]

        self.rough_coordinates = self.translate_coordinates(self.extract_polygon_coordinates(kml_content, "rough"), reference_point, flag_reference_point)
        self.rough_path = mpath.Path(self.rough_coordinates)

        self.lakes_coordinates = [self.translate_coordinates(coords, reference_point, flag_reference_point) for coords in self.extract_multiple_coordinates(kml_content, "water")]
        self.lake_paths = [mpath.Path(coords) for coords in self.lakes_coordinates]

        self.teeboxes_coordinates = [self.translate_coordinates(coords, reference_point, flag_reference_point) for coords in self.extract_multiple_coordinates(kml_content, "teebox")]
        self.teebox_paths = [mpath.Path(coords) for coords in self.teeboxes_coordinates]

        self.green_coordinates = self.translate_coordinates(self.extract_polygon_coordinates(kml_content, "green"), reference_point, flag_reference_point)
        self.green_path = mpath.Path(self.green_coordinates)

        self.bunkers_coordinates = [self.translate_coordinates(coords, reference_point, flag_reference_point) for coords in self.extract_multiple_coordinates(kml_content, "bunker")]
        self.bunker_paths = [mpath.Path(coords) for coords in self.bunkers_coordinates]

        self.trees_coordinates = [self.translate_coordinates(coords, reference_point, flag_reference_point) for coords in self.extract_multiple_coordinates(kml_content, "woods")]
        self.trees_paths = [mpath.Path(coords) for coords in self.trees_coordinates]

        self.flag_coordinates = self.translate_single_coordinate(self.extract_named_point_coordinates(kml_content, "flag"), reference_point, flag_reference_point)
        
        self.ball_coordinates = (0,0)

        self.par = kml_file_path[-5:-4]

    # Extract the polygon coordinates from the KML files
    def extract_polygon_coordinates(self,kml_content, placemark_name):
        root = ET.fromstring(kml_content)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        placemark = root.find(".//kml:Placemark[kml:name='{}']".format(placemark_name), namespaces)
        if placemark is not None:
            coordinates = placemark.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
            if coordinates is not None:
                coord_list = coordinates.text.strip().split()
                return [(float(lon), -float(lat)) for lon, lat, _ in 
                        (coord.split(',') for coord in coord_list)]
        return []

    # Extract the single point coordinates from the KML files, like the flag position or the reference
    def extract_named_point_coordinates(self,kml_content, point_name):
        root = ET.fromstring(kml_content)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Find Placemark with the specified name
        placemark = root.find(".//kml:Placemark[kml:name='{}']".format(point_name), namespaces)
         
        if placemark is not None:
            point = placemark.find('.//kml:Point', namespaces)
            if point is not None:
                coordinates = point.find('.//kml:coordinates', namespaces)
                if coordinates is not None:
                    coord_text = coordinates.text.strip()
                    lon, lat, _ = [float(x) for x in coord_text.split(',')]
                    return lon, -lat
        return None

    # Extract the multiple coordinates, if there are more than 1 element (e.g two lakes)
    def extract_multiple_coordinates(self,kml_content, element_name):
        root = ET.fromstring(kml_content)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

        # Use string formatting to incorporate the element_name into the XPath query
        elements = root.findall(".//kml:Placemark[kml:name='{}']".format(element_name), namespaces)
        elements_coordinates = []

        for element in elements:
            coordinates = element.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
            if coordinates is not None:
                coord_list = coordinates.text.strip().split()
                element_coords = [(float(lon), -float(lat)) for lon, lat, _ in (coord.split(',') for coord in coord_list)]
                elements_coordinates.append(element_coords)

        return elements_coordinates


    # Translate multiple coordinates from 3D to 2D
    def translate_coordinates(self,coordinates, reference, flag_reference):
        # Initialize the Geod object with WGS84 ellipsoid
        geod = Geod(ellps="WGS84")
        
        ref_lon, ref_lat = reference
        translated = []
        for lon, lat in coordinates:  # Corrected order of unpacking to (lon, lat)
            # Calculate azimuth and distance between the reference point and the current point
            azimuth1, _, distance = geod.inv(ref_lon, ref_lat, lon, lat)
            
            # Convert distance from meters to yards
            distance_yards = distance / 0.9144
            
            # Convert azimuth to radians
            azimuth_radians = math.radians(azimuth1)
            
            # Calculate the x and y components in yards
            x_component = math.cos(azimuth_radians) * distance_yards
            y_component = math.sin(azimuth_radians) * distance_yards
            
            translated.append((x_component, y_component))
        #return translated
        return self.adjust_orientation(translated,reference,flag_reference)
    

    # Rotatte the points by an angle given in radians
    def rotate_point(self,x, y, theta):
        x_rotated = x * math.cos(theta) - y * math.sin(theta)
        y_rotated = x * math.sin(theta) + y * math.cos(theta)
        return x_rotated, y_rotated
    
    # Adjust the orientation so the hole is always displayed in a vertical line with the reference being on the bottom 
    # of the image and the flag being on the top of the figure
    def adjust_orientation(self,coordinates, reference, flag):
        # Calculate azimuth from reference to flag
        geod = Geod(ellps="WGS84")
        azimuth1, azimuth2, distance = geod.inv(reference[0], reference[1], flag[0], flag[1])
        
        # Convert azimuth to radians and determine rotation angle
        # Azimuth is from north, so subtract from 90 degrees (or π/2 radians) for alignment
        rotation_angle = math.radians(90) - math.radians(azimuth1)
        
        # Rotate all points
        rotated_coordinates = [self.rotate_point(x, y, rotation_angle) for x, y in coordinates]
        
        return rotated_coordinates
    

    # Translate single coordinates from 3D to 2D
    def translate_single_coordinate(self,coordinates, reference, flag_reference):
        # Initialize the Geod object with WGS84 ellipsoid
        geod = Geod(ellps="WGS84")
        ref_lon, ref_lat = reference
        lon, lat = coordinates

        # Calculate azimuth and distance between the reference point and the current point
        azimuth1, _, distance = geod.inv(ref_lon, ref_lat, lon, lat)
        
       # Convert distance from meters to yards
        distance_yards = distance / 0.9144
        
        # Convert azimuth to radians
        azimuth_radians = math.radians(azimuth1)
        
        # Calculate the x and y components in yards
        x_component = math.cos(azimuth_radians) * distance_yards
        y_component = math.sin(azimuth_radians) * distance_yards

        ## rotate the point
        geod = Geod(ellps="WGS84")
        azimuth1, _, distance = geod.inv(reference[0], reference[1], flag_reference[0], flag_reference[1])
        
        # Convert azimuth to radians and determine rotation angle
        # Azimuth is from north, so subtract from 90 degrees (or π/2 radians) for alignment
        rotation_angle = math.radians(90) - math.radians(azimuth1)

        #return (x_component, y_component)
        return self.rotate_point(x_component, y_component, rotation_angle)

    # Return the flag coordinates
    def get_flag_coordinates(self):
        return self.flag_coordinates
    
    # Check if there are ant obstacles between any ball position and the flag
    def check_obstacles_in_line_of_sight(self, start_point, end_point):

        flag_tree = 0
        flag_lake = 0
        # Create a line path from start_point to end_point
        line_path = mpath.Path([start_point, end_point])

        # Check intersection with tree obstacles
        for tree_path in self.trees_paths:
            if line_path.intersects_path(tree_path, filled=True):
                flag_tree = 1
                break  # Exit loop once an obstruction is found

        # Check intersection with lake obstacles
        for lake_path in self.lake_paths:
            if line_path.intersects_path(lake_path, filled=True):
                flag_lake = 1
                break  # Exit loop once an obstruction is found

        return flag_lake, flag_tree




