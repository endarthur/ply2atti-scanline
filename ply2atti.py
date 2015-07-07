#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sin, radians, degrees, atan2, copysign, sqrt
from colorsys import hsv_to_rgb
from itertools import izip, combinations
from tempfile import TemporaryFile as temp

import numpy as np
from networkx import Graph, connected_components
from IPython import embed

#From Eelco Hoogendoorn answer to http://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
#Kudos!
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def calc_sphere(x, y, z):
	"""Calculate spherical coordinates for axial data."""
	return np.degrees(np.arctan2(*(np.array((x,y))*np.sign(z)))) % 360, 90. - np.degrees(np.arcsin(np.abs(z)))

def general_axis(data, order=0):
	"""Calculates the Nth eigenvector dataset tensor, first one by default."""
	direction_tensor = np.cov(data.T[:3, :])
	# print direction_tensor
	eigen_values, eigen_vectors = np.linalg.eigh(direction_tensor, UPLO='U')
	eigen_values_order = eigen_values.argsort()[::-1]
	cone_axis = eigen_vectors[:,eigen_values_order[order]]
	return cone_axis/np.linalg.norm(cone_axis)

def calibrate_azimuth(data, target_color, target_azimuth):
	calibrate_data = np.mean(data[target_color], axis=0)
	d_az = target_azimuth - calibrate_data[0]
	# print calibrate_data
	# print d_az
	for color in data.keys():
		data[color] = [((az + d_az) % 360, dip) for az, dip in data[color]]
	return data

def parse_ply(f, colors, eig=False):
	output = {color:[] for color in colors}
	# if not f.readline() == "ply": raise Exception("You must use a .ply (stanford format) file.")
	# if not "ascii" in f.readline(): raise Exception("You must use the ascii .ply specification.")
	header = f.readline()
	print header
	while header != "end_header\n": header = f.readline()
	for line in f:
		data = line.split()
		if len(data) < 10:
			break
		x, y, z,\
		nx, ny, nz,\
		r, g, b, alpha = data
		color = (r, g, b, alpha)
		normal = np.array((float(nx), float(ny), float(nz)))
		if color in colors:
			if eig:
				position = np.array((float(x), float(y), float(z)))
				output[color].append(position)
			else:
				normal = normal/np.linalg.norm(normal)
				output[color].append(calc_sphere(*normal))
			# embed()
		# line = f.readline()
	if eig:
		for color in colors:
			output[color] = (calc_sphere(*general_axis(np.array(output[color]), -1)),)
	return output

def parse_ply_nx(f, colors, eig=False,  network=False):
	output = {color:[] for color in colors}
	output_indices = {color:[] for color in colors}
	output_graphs = {color:Graph() for color in colors}
	# if not f.readline() == "ply": raise Exception("You must use a .ply (stanford format) file.")
	# if not "ascii" in f.readline(): raise Exception("You must use the ascii .ply specification.")
	header_line = f.readline()
	if not "ply" in header_line: 
		raise IOError("can only read text ply files")
	while header_line != "end_header\n":
		print header_line,
	 	if "element" in header_line:
			if "vertex" in header_line:
				vertex_number = int(header_line.split()[-1])
			if "face" in header_line:
				face_number = int(header_line.split()[-1])
		header_line = f.readline()
	vertices = np.memmap(temp(), dtype="float_", mode="w+",
	                   	             shape=(vertex_number, 10))
	for i, line in enumerate(f):
		####YOU CAN DO BETTER THAN THIS... READ THE NUMBER OF ELEMENTS  AND ITERATE ON THAT, LIKE CROWBAR
		data = line.split()
		if len(data) <  10:
			nodes = [int(node_index) for node_index in data[1:]]
			for color in colors:
				colored_indices = output_indices[color]
				colored_nodes = [node for node in nodes if tuple(vertices[node,3:7]) == color]
				if colored_nodes:
					#print 'edge'
					output_graphs[color].add_edges_from(combinations(colored_nodes,  2))
			if not i % 10000: print "processing face %i/%i..." % (i - vertex_number, face_number)
		else:
			x, y, z,\
			nx, ny, nz,\
			r, g, b, alpha = data
			color = tuple((int(r), int(g), int(b), int(alpha)))
			normal = np.array((float(nx), float(ny), float(nz)))
			if color in colors:
				if eig:
					position = np.array((float(x), float(y), float(z)))
					output[color].append(position)
				elif network:
					output_indices[color].append(i)
					vertices[i,:] = np.array((float(x), float(y), float(z),float(r),int(g),int(b),int(alpha), float(nx), float(ny), float(nz)))
				else:
					normal = normal/np.linalg.norm(normal)
					output[color].append(calc_sphere(*normal))
			if not i % 10000: print "processing node %i/%i..." % (i, vertex_number)
	if eig:
		for color in colors:
			output[color] = (calc_sphere(*general_axis(np.array(output[color]), -1)),)
	elif network:
		for color in colors:
			if __debug__: print "processing network for color ", color
			for plane_vertices_indices in connected_components(output_graphs[color]):
				colored_vertices = vertices[plane_vertices_indices,:3]
				dipdir, dip = calc_sphere(*general_axis(colored_vertices, -1))
				X, Y, Z = colored_vertices[:, :3].mean(axis=0)
				highest_vertex = colored_vertices[np.argmax(colored_vertices[:,2]),:]
				lowest_vertex = colored_vertices[np.argmin(colored_vertices[:,2]),:]
				trace = np.linalg.norm(highest_vertex - lowest_vertex)

				direction_cosines = normalized(vertices[plane_vertices_indices, 7:])
				n = direction_cosines.shape[0]
				resultant_vector = np.sum(direction_cosines, axis=0)
				fisher_k = (n - 1)/(n - np.linalg.norm(resultant_vector))

				direction_tensor = np.dot(direction_cosines.T, direction_cosines)
				eigen_values, eigen_vectors = np.linalg.eigh(direction_tensor)
				eigen_values_order = (-eigen_values).argsort()
				
				first_eigenvalue,\
				second_eigenvalue,\
				third_eigenvalue = eigen_values[eigen_values_order]
				
				first_eigenvector,\
				second_eigenvector,\
				third_eigenvector = eigen_vectors[:,eigen_values_order].T
				
				#From Vollmer 1990
				vollmer_P = (first_eigenvalue - second_eigenvalue)/n
				vollmer_G = 2*(second_eigenvalue - third_eigenvalue)/n
				vollmer_R = 3*third_eigenvalue/n
				
				vollmer_B = vollmer_P + vollmer_G

				output[color].append((dipdir, dip, X, Y, Z, trace, n, fisher_k, vollmer_P, vollmer_G, vollmer_R, vollmer_B))
	#embed()
	return output

def color_encode_ply(f, f_out, value=0.7):
	output = {color:[] for color in colors}
	# if not f.readline() == "ply": raise Exception("You must use a .ply (stanford format) file.")
	# if not "ascii" in f.readline(): raise Exception("You must use the ascii .ply specification.")
	header = f.readline()
	#print header,
	f_out.write(header)
	while header != "end_header\n":
		header = f.readline()
		f_out.write(header)
		#print header,
	for line in f:
		data = line.split()
		if len(data) < 10:
			f_out.write(line)
			continue
		x, y, z,\
		nx, ny, nz,\
		r, g, b, alpha = data
		f_nx, f_ny, f_nz = float(nx), float(ny), float(nz)
		norm_n = sqrt(f_nx**2 + f_ny**2 + f_nz**2)
		if norm_n:
			sign_nz = copysign(1, f_nz)
			r, g, b = hsv_to_rgb((degrees(atan2(f_nx*sign_nz, f_ny*sign_nz)) % 360)/360., abs(f_nz/norm_n), value)
			r, g, b = int(r*255), int(g*255), int(b*255)

		f_out.write(" ".join((str(value) for value in (x, y, z,\
							   nx, ny, nz,\
							   r, g, b, alpha))) + "\n")



if __name__ == "__main__":
	from datetime import datetime
	starttime = datetime.now()
	from sys import argv
	from optparse import OptionParser, OptionGroup
	parser = OptionParser(usage="%prog -f input_filename [options] [color1 color2 ... colorN] [-o output_filename]", version="%prog 0.6")
	parser.add_option("-f", "--file", dest="infile", metavar="FILE", help="input painted 3d model")
	parser.add_option("-o", "--outfile", dest="outfile", metavar="FILE", help="output color coded 3d model, for use with --colorencode")
	# parser.add_option("-c", "--colorencode", action="store_true", dest="colorencode", help="Process the model and paints it according to the attitude of each face, based on Assali 2013.", default=False)
	parser.add_option("-j", "--join", action="store_true", dest="join", default=False, help="joins all resultant data in a single file, instead of a file for each color as default. Recomended if using --eigen option.")
	parser.add_option("-n", "--network", action="store_true", dest="network", help="Outputs each different colored plane, through graph analysis.", default=False)
	# parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="outputs detailed information on the data", default=False)
	group = OptionGroup(parser, "Calibration Options", "These are small utilities to aid calibration of your data.")
	group.add_option("-e", "--eigen", action="store_true", dest="eig", help="outputs only the third eigenvector of each color points.", default=False)
	group.add_option("-a", "--azimuth", action="store", dest="calibration_data", metavar="COLOR:AZIMUTH", default=None, help="calibrates your output data by turning its azimuth horizontaly until the given color has the given dipdirection")
	group.add_option("-u", "--value", action="store", dest="value", help="Determines the value used for the color encode option. Defaults to 0.90.", default=.9)
	parser.add_option_group(group)
	(options, args) = parser.parse_args()

	colors = []
	if not options.colorencode:
		for color in args:
			components = tuple(color.split(','))
			if len(components) < 4: components += ('255',)
			colors.append(tuple([int(component) for component in components]))
		filename = options.infile.split()[0]
		with open(options.infile, 'r') as f:
			output = parse_ply_nx(f, colors, options.eig, options.network)
			if options.calibration_data:
				color, az = options.calibration_data.split(":")
				components = tuple(color.split(','))
				if len(components) < 4: components += ('255',)
				color = components
				az = int(az)
				# embed()
				output = calibrate_azimuth(output, color, az)
			# print output
		if not options.join:
			for color in output.keys():
				with open("{0}_{1}.txt".format(filename, color), 'w') as f, open("{0}_{1}_coords.txt".format(filename, color), 'w') as coordf:
					coordf.write("X\tY\tZ\tatti\ttrace\tn_points\tfisher_k\tvoll_P\tvoll_G\tvoll_R\tvoll_B\n")
					for dipdir, dip, X, Y, Z, trace, n, fisher_k, vollmer_P, vollmer_G, vollmer_R, vollmer_B in output[color]:
						f.write("{0}\t{1}\n".format(dipdir, dip))
						coordf.write("{0}\t{1}\t{2}\t{3}/{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n".format(X, Y, Z, int(dipdir), int(dip), trace, n, fisher_k, vollmer_P, vollmer_G, vollmer_R, vollmer_B))
					#np.savetxt(f,  output[color], delimiter="\t",header="dipdir\tdip\tX\tY\tZ")
		else:
			with open("{0}_attitudes.txt".format(filename), 'w') as f:
				for color in output.keys():
					f.write("#{0}\n".format(color))
					for dipdir, dip in output[color]:
						f.write("{0}\t{1}\n".format(dipdir, dip))
	else:
		with open(options.infile, 'r') as f, open(options.outfile, 'wb') as fo:
			color_encode_ply(f, fo, value=float(options.value))
	print "Total time processing ", datetime.now() - starttime,"..."
	print "\a"