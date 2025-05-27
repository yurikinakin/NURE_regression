import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import rasterio as rio

from rasterio.features import dataset_features
from geopy import distance
from io import StringIO
from osgeo import gdal
from tqdm import tqdm
from whitebox_workflows import WbEnvironment, show, WbPalette, AttributeField, VectorGeometryType, FieldDataType, FieldData


class generate_global_watersheds:
    def __init__(self, epsg = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'):       
        self.sample_points = None 
        self.file_list = None
        self.file_dict = None
        self.distance_array = None
        self.adjacent_file_list = None
        self.clipped_sample_points = None
        self.dem = None
        self.streams = None
        self.d8_pntr = None
        self.dem_filled = None
        self.snapped_clipped_sample_points = None
        
        self.out_att_fields = [
            AttributeField("FID", FieldDataType.Int, 6, 0),
            AttributeField("SRC_FID", FieldDataType.Int, 6, 0),
            AttributeField("SAMPLEID", FieldDataType.Int, 8, 0)
        ]
        self.watershed_vectors = wbe.new_vector(VectorGeometryType.Polygon, self.out_att_fields, proj=epsg)


    def load_samples(self, filename, x_field_num=0, y_field_num=1, epsg = 4326):
        self.sample_points = wbe.csv_points_to_vector(filename, x_field_num=x_field_num, y_field_num=y_field_num, epsg = epsg)
        
    # Utilize this function to convert all of the .tif files downloaded from the USGS main portal into a compressed format that can be read by WhiteBoxGeo.
    # The savings in compression is approximately 30%. This will grab all of the files in the directory in which this is run. The existsing .tif files will not be deleted,
    # so in order to realize the space savings a user will need to manually delete them after the tool has been run. 
    def convert_files(self):
        topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
        path = rf'*.tif'
        file_list = glob.glob(path)
        for file in file_list:
            dataset = gdal.Open(file, gdal.GA_ReadOnly)
            gdal.Translate(file[:-3] + 'tiff', file, creationOptions=["COMPRESS=LZW", "TILED=YES", "DISCARD_LSB=10"])

    # Creates a dictionary of {filename: center, upper left, lower right} corner coordinates for each of the DEM raster files in the current directory. This is used by
    # later functions to determine tile adjacency. This function assumes that all files have already been converted via the 'convert_files' function. 
    def create_DEM_dict(self, rerun = True):
        if rerun == True:
            print('create_DEM_dict reran!')
            path = rf'USGS*.tiff'
            self.file_list = glob.glob(path)
            self.file_dict = {}
            for file in self.file_list:
                dataset = gdal.Open(file, gdal.GA_ReadOnly)
                pixels = (dataset.RasterXSize, dataset.RasterYSize)
                corner_coords = [dataset.GetGeoTransform()[0], dataset.GetGeoTransform()[3], dataset.GetGeoTransform()[0] + (pixels[0]-1)*dataset.GetGeoTransform()[1], dataset.GetGeoTransform()[3] + (pixels[1]-1)*dataset.GetGeoTransform()[5]]
                center_coordinate = [dataset.GetGeoTransform()[3] + (pixels[1]*dataset.GetGeoTransform()[5])/2, dataset.GetGeoTransform()[0] + (pixels[0]*dataset.GetGeoTransform()[1])/2]
                self.file_dict[file] = [center_coordinate + corner_coords]         
            with open('file_list.pkl', 'wb') as handle:
                pickle.dump(self.file_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('file_dict.pkl', 'wb') as handle:
                pickle.dump(self.file_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('file_dict.pkl', 'rb') as handle:
                self.file_dict = pickle.load(handle)
            with open('file_list.pkl', 'rb') as handle:
                self.file_list = pickle.load(handle)
        
        return self.file_list
                
    # This calculates the distance in kms between centrepoints of DEM files utlizing the dictionary of file adjacencies created by 'create_DEM_dict', returning an array of
    # disstances. 
    def calc_DEM_dist(self, rerun = True):
        def cal_coords(eval_array):
            eval_coord = (eval_array[0], eval_array[1])
            center_coord = (eval_array[2], eval_array[3])
            return distance.distance(eval_coord, center_coord).km
        
        if rerun == True:
            print('calc_DEM_dist reran!')
            distance_list = []
            coord_array = np.array(list(self.file_dict.values()))[:,0,:2] # Create an array of the center coordinates to determine distances.
            for file, value in self.file_dict.items():
                centerpoint = np.array([[np.float32(value[0][0])], [np.float32(value[0][1])]])
                eval_array = np.hstack((coord_array, np.repeat(centerpoint, coord_array.shape[0], axis = 1).T))
                dist_vector = np.apply_along_axis(cal_coords, axis = 1, arr=eval_array)
                distance_list.append(dist_vector)
            self.distance_array = np.asarray(distance_list)
            with open('distance_array.pkl', 'wb') as handle:
                pickle.dump(self.distance_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('distance_array.pkl', 'rb') as handle:
                self.distance_array = pickle.load(handle)

    # Create a list of lists that associates each filename with an associated list of adjacent tiles. Tiles are assumed to be all the same size, so the adjacency value is just an integer that specifies
    # the number of tiles to search outwards from the center of the active DEM tile.
    def find_adjacent_tiles(self, adjacent_tiles, rerun = True):
        i = 0
        self.adjacent_tile_list = []
        if rerun == True:
            print('find_adjacent_tiles reran!')
            for file in self.file_list:
                corner_ul = (self.file_dict[file][0][3], self.file_dict[file][0][2])
                corner_br = (self.file_dict[file][0][5], self.file_dict[file][0][4])
                c2c_distance = distance.distance(corner_ul, corner_br).km
                self.adjacent_tile_list.append(np.flatnonzero(self.distance_array[:,i] <= c2c_distance*1.1*adjacent_tiles).tolist())
                i+=1
                with open('adjacent_tile_list.pkl', 'wb') as handle:
                    pickle.dump(self.adjacent_tile_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('adjacent_tile_list.pkl', 'rb') as handle:
                self.adjacent_tile_list = pickle.load(handle)

    # Take a single DEM and find the points that are wholly contained within the outline polygon of the raster. 
    def get_points_in_DEM(self, filename):       
        mask_polygon = wbe.layer_footprint_raster(wbe.read_raster(filename))
        self.clipped_sample_points = wbe.clip(self.sample_points, mask_polygon)
        return self.clipped_sample_points.num_records
        
    # This takes an individual DEM tile and returns a mosaic of adjacent tiles.
    def create_DEM_mosaic(self, filename):
        file_index = self.file_list.index(filename)
        files = [self.file_list[i] for i in self.adjacent_tile_list[file_index]]
    
        def create_rasters(files):
            raster_list = []
            for file in files:
                raster_list.append(wbe.read_raster(file))
            return raster_list
    
        raster_list = create_rasters(files)
        self.dem = wbe.mosaic(images = raster_list)

    # Create the filled dem by filling all sink depressions. Determine the stream network from flow accumulation. The default value is 25000, but this is something that can be optimized 
    # by trial and error. A d8 pointer is created to associate a direction with each value in the flow accumulation raster. 
    def create_d8_pntr(self, channel_threshold = 25000.0):
        self.dem_filled = wbe.fill_depressions(dem = self.dem)
        flow_accum = wbe.qin_flow_accumulation(self.dem_filled, out_type='cells', log_transform = True)
        self.streams = flow_accum > math.log(channel_threshold)
        self.d8_pntr = wbe.d8_pointer(self.dem_filled)

    # Use Jenson's method to snap sample points to the closest stream.
    def snap_points_to_stream(self, max_snap_distance = 5): 
        self.snapped_clipped_sample_points = wbe.jenson_snap_pour_points(self.clipped_sample_points, self.streams, max_snap_distance)

    # Create a full watershed at each point in the sample set. This is done using the wbe.watersheds function call only. The remaining code in this section allows the watershed to be
    # associated with the correct sample point and written to a shapefile. The shapefile is overwritten for each DEM time that is prduced. This was done to allow for faster debugging and 
    # just output at the end of the full call. 
    def create_global_watersheds(self):
        i = 0
        for point in self.snapped_clipped_sample_points:  
            vector_point = wbe.new_vector(VectorGeometryType.Point, self.out_att_fields)
            vector_point.add_record(point)
            watershed = wbe.watershed(d8_pointer=self.d8_pntr, pour_points=vector_point)
            watershed_vec = wbe.raster_to_vector_polygons(watershed)
            sample_rec = self.snapped_clipped_sample_points.get_attribute_record(i)
            sample_data = [
                FieldData.new_int(i), 
                FieldData.new_int(i+1),
                sample_rec[2]
            ]
            self.watershed_vectors.add_record(watershed_vec[0])
            self.watershed_vectors.add_attribute_record(sample_data, deleted = True)
            i+=1
        wbe.write_vector(self.watershed_vectors, 'watershed.shp')

wbe = WbEnvironment()
watersheds = generate_global_watersheds()
watersheds.load_samples('NURE_input_ss.csv')
file_list = watersheds.create_DEM_dict(rerun = True)
watersheds.calc_DEM_dist(rerun = True)
watersheds.find_adjacent_tiles(adjacent_tiles = 1, rerun = True)
for file in tqdm(file_list, total = len(file_list)):
    num_points = watersheds.get_points_in_DEM(file)
    if num_points > 0:87cur
        watersheds.create_DEM_mosaic(file)
        watersheds.create_d8_pntr()
        watersheds.snap_points_to_stream()
        watersheds.create_global_watersheds()
