import sys
import pandas as pd 
import numpy as np

def get_preamble():
	return("import javascript\n\nbindingset[portal]\nbindingset[event]\npredicate isExternalDatapoint(string portal, string event) {\n")

def get_predicate_body(df):
	toret = df.apply(lambda row: "\tportal = \"" + str(row.portal) + "\" and event = \"" + str(row.eventname).replace('"', r'\"') + "\"", axis=1)
	return( " or\n".join(toret.to_list()) + "\n}")

def get_AP_class(df):
	to_ret = "\n\nclass ExtAPString extends string {\n\tExtAPString() {\n"
	to_ret_df = df.apply(lambda row: "\t\tthis = \"" + str(row.portal) + "\"", axis=1)
	return( to_ret + " or\n".join(to_ret_df.to_list()) + "\n\t}\n}")

def get_event_class(df):
	to_ret = "\n\nclass ExtEventString extends string {\nExtEventString() {\n"
	to_ret_df = df.apply(lambda row: "\t\tthis = \"" + str(row.eventname).replace('"', r'\"') + "\"", axis=1)
	return( to_ret + " or\n".join(to_ret_df.to_list()) + "\n\t}\n}")

if( len( sys.argv) != 3):
	print("Usage: python3 fake_external_data_allData.py file_with_all_data output_filename")
else:
	df = pd.read_csv( sys.argv[1], sep=",", header=None)
	df.columns = ["proot", "portal", "eventname", "projcount", "path"]
	df["whitespace_ename"] = df.apply(lambda row: str(row.eventname).isspace(), axis=1)
	df = df[df.whitespace_ename == False] # forced boolean comp bc of pandas structure
	df.drop(["proot", "projcount", "path"], inplace=True, axis=1)
	df.drop_duplicates(inplace=True)
	out_file = open( sys.argv[2], "w", encoding='utf-8');
	out_file.write( get_preamble() + get_predicate_body(df) + get_AP_class(df) + get_event_class(df))
	out_file.close()
	print("Done processing")
