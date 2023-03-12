import sys
import pandas as pd 
import json
import numpy as np

# print dataframe df to file specified
def printDFToFile( df, filename):
	if not df.empty:
		f = open(filename, 'w', encoding='utf-8');
		f.write(df.to_csv(index = False, header=None))
		f.close()

def getPortalRoot( portal):
	return portal[ portal.index('module ') + len('module ') : portal.index(')')]

# function to load in a single file
# return a dataframe of the portal, eventname, and project
def get_info_from_file( filename, exclude_paths):
	f = open(filename, "r")
	data = json.loads( f.read())
	f.close()
	# now, read in the relevant info from the json object
	msg_type = [k[0].get("value") for k in data.get("data")] 
	portals = [k[1].get("value") for k in data.get("data")]
	proots = [k[2].get("value") for k in data.get("data")]
	eventnames = [k[3].get("value") for k in data.get("data")]
	proj_names = [data.get("project").get("name")] * len(portals)
	proj_count = [k[4].get("value") for k in data.get("data")]#[1] * len(portals) # we don't remove duplicates yet
	ret_val = pd.DataFrame(np.column_stack([proots, msg_type, portals, eventnames, proj_count, proj_names]), 
							columns = ["proot", "type", "portal", "eventname", "projcount", "path"])
	if exclude_paths:
	    del ret_val["path"]
	# ret_val.insert(loc=0, column="proot", value=ret_val.portal.apply(getPortalRoot))
	return( ret_val)

def merge_info_from_files( specfile, exclude_paths):
	f = open( specfile, "r")
	all_files = f.read().splitlines()
	f.close()
	return( pd.concat([get_info_from_file(fname, exclude_paths) for fname in all_files], ignore_index=True))

# read in a file that has a list of all json files to use 
# specify this file as a command line arg

if( len( sys.argv) < 3):
	print("Usage: python3 parse_query_output_into_csv.py file_spec_list_of_query_output_files output_filename [bool_exclude_paths]")
else:
	exclude_paths = True if len(sys.argv) == 3 else (sys.argv[3] == "true")
	df = merge_info_from_files( sys.argv[1], exclude_paths)
	printDFToFile( df[df.type == "emit"].drop(["type"], axis=1), "emit_" + sys.argv[2])
	printDFToFile( df[df.type == "listen"].drop(["type"], axis=1), "listen_" + sys.argv[2])
	print("Done processing")
