import xml.etree.ElementTree as etree
import math
from sys import argv, exit

import numpy as np

def calc_sphere(x, y, z):
	"""Calculate spherical coordinates for axial data."""
	return np.degrees(np.arctan2(*(np.array((x,y))*np.sign(z)))) % 360, np.degrees(np.arccos(np.abs(z)))

#From http://docs.python.org/2/library/itertools.html#recipes
def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

if __name__=="__main__":
    if len(argv) != 2 or argv[1]=="--help":
        print """\
Virtual scanline analysis system, for use with meshlab point picking tool.
Usage:

python proc_sline.py file_name

Prints to stdout the resultant data.
"""    
        exit()
    root = etree.parse(argv[1]).getroot()
    
    scanline_start, scanline_end = [np.array(((float(x.attrib['x']), float(x.attrib['y']), float(x.attrib['z'])))) for x in root[1:3]]
    scanline_orientation =  scanline_end - scanline_start
    scanline_orientation /= np.linalg.norm(scanline_orientation)
    
    points = sorted([int(x) for x in set([x.attrib['name'] for x in root[3:]])])
    
    distances = []
    #last_d = 0.0
    
    print "dipdir,dip,point,position,length,angle"
    for point in points:
        a, b, c = [np.array(((float(x.attrib['x']), float(x.attrib['y']), float(x.attrib['z'])))) for x in root.findall("point[@name='%i']" % point)]
        ab = b - a
        ac = c - a
        bc = c - b
        length = max([np.linalg.norm(x) for x in (ab, ac, bc)])
        centroid = np.mean((a, b, c), axis=0)
        d = np.dot(centroid - scanline_start, scanline_orientation)
        distances.append(d)
        #last_d = d
        n = np.cross(ab, ac)
        n /= np.linalg.norm(n)
        theta = math.degrees(math.acos(abs(np.dot(n, scanline_orientation))))
        print "{0[0]},{0[1]},{1},{2},{3},{4}".format(calc_sphere(*n), point, d, length, theta)
