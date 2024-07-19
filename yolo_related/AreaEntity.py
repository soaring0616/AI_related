from typing import List
import numpy as np

class AreaEntity:
    def __init__(self, poly_points: List) -> None:
        self._contour = np.array(poly_points, np.int32).reshape((-1, 1, 2))
        self._max_x = np.max(self._contour, axis=0).flatten()[0]
        self._max_y = np.max(self._contour, axis=0).flatten()[1]
        self._min_x = np.min(self._contour, axis=0).flatten()[0]
        self._min_y = np.min(self._contour, axis=0).flatten()[1]

    @property
    def contour(self):
        return self._contour

    @property
    def max_x(self):        
        return self._max_x
    
    @property
    def max_y(self):        
        return self._max_y
    
    @property
    def min_x(self):        
        return self._min_x
    
    @property
    def min_y(self):        
        return self._min_y

if __name__ == '__main__':
    import json
    with open("danger_area_cood.json") as f:
        danger_zone = json.load(f)
            
    danger1 = AreaEntity(danger_zone[2]['annotations'][0]['result'][0]['value']['points'])
    danger2 = AreaEntity(danger_zone[2]['annotations'][0]['result'][1]['value']['points'])
    
    print(danger1)
    print(danger2)