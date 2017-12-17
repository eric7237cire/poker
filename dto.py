from shapely.geometry import Polygon

class BoundingBox(object):

    def __init__(self, min_x=None, max_x=None, min_y=None, max_y=None):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def clip_2d_array(self, image_yx_array):

        return image_yx_array[
               int(self.min_y):int(self.max_y)+1,
               int(self.min_x):int(self.max_x)+1]


class Contour(object):

    def __init__(self):
        self.bounding_box = None

        # 2d array in y/x order
        self.points_array = None

        self.polygon = None

    def set_points_array(self, points_array):
        self.points_array = points_array
        self.polygon = Polygon(self.get_contour_xy())

    def get_contour_xy(self):
        """
        Contours are in y,x
        :param contour:
        :return: same points x,y
        """
        # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
        return self.points_array[:, [1, 0]]