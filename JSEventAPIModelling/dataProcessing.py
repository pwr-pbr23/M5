import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import re
from collections import namedtuple
from collections import Counter
import csv
import itertools
import time

pd.options.mode.chained_assignment = None # remove chained assignment warnings (they dont fit my use case)

# suppress runtime warnings to avoid polluting the terminal
# comment this out if debugging
import warnings
warnings.filterwarnings("ignore",category = RuntimeWarning)


# ----------------------------------------------------------------------------------------------------------------------- processing functions
# given the name of a portal, split the string up to find the name of the root
def getPortalRoot( portal):
	return portal[ portal.index('module ') + len('module ') : portal.index(')')]

# get length of an access path (where just a root has length 1)
# to do this just count the number of closing parens, since these only appear at the end of the portal
# and indicate the level of nesting, which is what we're considering length
def getPortalLength( portal):
	return( portal.count(")"))

# given a csv of (root, portal, eventname, projcount) pairs, process it into a dataframe of the correct shape
# named columns, and including the portal root as a column
# we also add the processed columns we want (frequency, etc)
def processFile( fileName, condense_portals = False):
	result = pd.read_csv(fileName, sep = ',', header=None)
	result.columns = ['proot', 'portal', 'eventname', 'projcount']
	if condense_portals:
		result.portal = result.portal.apply(condensePortal)
	result.replace(np.nan, 'NaN', regex=True, inplace=True)
	result['freq'] = result.groupby(['portal','eventname', 'proot'])['projcount'].transform('sum') # add the frequency with which each row appears
	result.drop(['projcount'], inplace=True, axis=1) # this information only makes sense to keep when we also have the project saved, i.e. in the "with files" version
	result.drop_duplicates(inplace=True)
	result['freq_e'] = result.groupby(['proot','eventname'])['freq'].transform('sum') # find frequency with which each event appears for the root
	result['freq_p'] = result.groupby(['proot','portal'])['freq'].transform('sum')	  # same for the portals
	return result

def get_dat_from_dat_with_files( df_with_path):
	result = pd.DataFrame.copy( df_with_path)
	result.drop(["path"], inplace=True, axis=1) # remove unwanted project column
	result.replace(np.nan, 'NaN', regex=True, inplace=True)
	result['freq'] = result.groupby(['portal','eventname', 'proot'])['projcount'].transform('sum') # add the frequency with which each row appears
	result.drop(['projcount'], inplace=True, axis=1) # this information only makes sense to keep when we also have the project saved, i.e. in the "with files" version
	result.drop_duplicates(inplace=True)
	result['freq_e'] = result.groupby(['proot','eventname'])['freq'].transform('sum') # find frequency with which each event appears for the root
	result['freq_p'] = result.groupby(['proot','portal'])['freq'].transform('sum')	  # same for the portals
	return result

# when we have a dataframe which includes the path to the origin project for each 
# (portal, eventname) pair, this function returns the path for a particular pair
def getPathForPortalEname( df_with_path, portal, ename):
	print(ename)
	print(portal)
	return df_with_path[(df_with_path['portal'] == portal) & (df_with_path['eventname'] == ename)]['path'] 

# print dataframe df to file specified
def printDFToFile( df, filename):
	f = open(filename, 'w');
	f.write(df.to_csv(index = False))
	f.close()

# print the results in the "category" column labeledas the category specified 
# out to a file (either appending or replacing the entire contents of the file, as specified by the optional "append" argument)
# we're printing in the format to match the output of the corresponding QL query, so
# as a list of (portal, eventname) pairs, pre/suffixed with "", and without printing the headers
def printCatResultsToFile(df, filename, category, append=True):
	write_mode = 'a' if append else 'w'
	f = open(filename, write_mode)
	toprint = df[df['category'] == category][['freq', 'portal', 'eventname']]
	f.write(toprint.to_csv(index=False, header=False, quoting=csv.QUOTE_ALL))
	f.close()

# ------------------------------------------------------------------------------------------------------------------------------------- graphing

# plot a histogram of all the portals with listeners for a specified eventname, on a specified root
# additional, optional arguments are:
# topToPlot: if a specified positive integer, we plot the "topToPlot" most frequent portals 
# logscale: do we logscale the y axis?
# showTickLabels: the xticklabels are the portal names so they are very unweidly (theyre also shown as a legend and not on the xaxis)
#				  it only makes sense to have them when there are not many portals, or it quickly becomes unreadable 
def plotHistProotEname( proot, eventname, df, topToPlot = -1, logscale = False, showTickLabels = True):
	plotme = df.loc[(df['proot'] == proot) & (df['eventname'] == eventname)][['portal','freq']].sort_values(['freq'])
	if topToPlot > -1:
		plotme = plotme.tail(topToPlot)
	graph = plotme.plot(kind='bar',x='portal',y='freq')
	plt.xlabel('Portal')
	plt.ylabel('Frequency')
	if logscale:
		plt.yscale('log')
	plt.title('Frequency of listeners to "' + eventname + '" on root ' + proot, fontsize=10)
	plt.legend().remove()
	# 
	if showTickLabels:
		plt.gca().axes.xaxis.set_ticklabels(list(range(len(plotme))))
		# make a custom "legend" (i.e. textbox) of the portals actually on this graph
		portals = list( plotme['portal'].values)
		plt.text(0, plt.ylim()[1]*1./4, getLegendList(portals), fontsize=6)
	else:
		plt.gca().axes.xaxis.set_ticklabels([])
	plt.show()

# plot a histogram of the events for which listeners are registered for a specified portal
# note that this is root-specific even though not explicitly encoded, since the portals are root-specific
def plotHistPortalEnames( portal, df):
	plotme = df.loc[df['portal'] == portal][['eventname', 'freq']].sort_values(['freq'])
	graph = plotme.plot(kind='bar',x='eventname',y='freq')
	plt.xlabel('Event name')
	plt.ylabel('Frequency')
	plt.title('Frequency of each listener on portal ' + portal, fontsize=10)
	plt.legend().remove()
	plt.show()

# helper function for getting the legend for the portal histogram (legend is the list of portals)
def getLegendList( pList):
	ret = ""
	for i in range(len(pList)):
		ret += str(i) + ": " + pList[i] + "\n"
	return ret

# gets and returns the optimal solutions for this root, and plots them if specified to
# the optimal solution is defined as the pareto front; we're optimizing to minimize the x variable (FP rate)
# and maximize the y variable (number of results)
def getRootSpecPareto( root_spec_res, root, plotme = False):
	rate = [x[0][0] for x in root_spec_res[root]]
	vals = [x[0][1] for x in root_spec_res[root]]
	return getGenericPareto( rate, vals, root_spec_res[root], plotme, "Number of results vs false positive rate for " + root, "False positive rate", "Number of results")

# optimal solutions for this root, but where we're maximizing the x variable (use for TP rate)
def getRootSpecParetoMax( root_spec_res, root, plotme = False):
	rate = [x[0][0] for x in root_spec_res[root]]
	vals = [x[0][1] for x in root_spec_res[root]]
	return getGenericPareto( rate, vals, root_spec_res[root], plotme, "Number of true positives vs true positive rate for " + root, "True positive rate", "Number of true positives", True)

# optimal solutions overall, plotting if specified to
# same process as getRootSpecPareto but for the whole dataset
def getOverallPareto(resultsum, plotme = False):
	rate = [x[0][0] for x in resultsum]
	vals = [x[0][1] for x in resultsum]
	return getGenericPareto( rate, vals, resultsum, plotme, "Number of results vs false positive rate over all packages", "False positive rate", "Number of results")

# same as getOverallPareto but maximizing the x variable (use for TP rate)
def getOverallParetoMax( resultsum, plotme = False, colourParetoPointAtThresh = -1):
	rate = [x[0][0] for x in resultsum]
	vals = [x[0][1] for x in resultsum]
	return getGenericPareto( rate, vals, resultsum, plotme, "Number of true positives vs true positive rate over all packages", "True positive rate", "Number of true positives", True, colourParetoPointAtThresh)

def getOverallParetoMax_dispRecall( resultsum, recall_denom, plotme = False, colourParetoPointAtThresh = -1):
	rate = [x[0][0] for x in resultsum]
	vals = [x[0][1]/recall_denom for x in resultsum]
	return getGenericPareto( rate, vals, resultsum, plotme, "Recall rate vs precision over all packages", "Precision (true positive rate)", "Recall rate", True, colourParetoPointAtThresh)


# get and plot the pareto for a generic pair of lists
# the non-generic part is we're always minimizing the x and maximizing the y to represent the optimal set of solutions
# unless maximize = True, in which case we're maximizing the x variable too
# the last optional parameter specifies a point to draw as a star on the graph (in the paper, we highlight the point closest to TP threshold 0.9)
def getGenericPareto( rate, vals, rs, plotme = False, title = "", xtitle = "", ytitle = "", maximize = False, colourParetoPointAtThresh = -1):
	# if we're maximizing over x, reverse the lists
	if maximize:
		rate.reverse()
		vals.reverse()
		rs.reverse()
	# can't plot or do anything if there's no data
	if len(rate) == 0:
		if plotme:
			print("Cant plot: no data\n")
		return []
	# compute pareto front
	opt_i = 0
	pareto_list = []
	for i in range(len(rate)):
		if vals[i] > vals[opt_i]:
			if (rate[i] < rate[opt_i] and maximize) or (rate[i] > rate[opt_i] and not maximize):
				pareto_list += [(rate[opt_i], vals[opt_i], rs[opt_i])]
			opt_i = i
	pareto_list += [(rate[opt_i], vals[opt_i], rs[opt_i])]
	if plotme:
		plt.scatter( rate, vals)
		plt.plot( [x[0] for x in pareto_list], [x[1] for x in pareto_list], 'r')
		plt.rc('xtick',labelsize=10)
		plt.rc('ytick',labelsize=10)
		plt.title(title,fontsize=15)
		plt.xlabel(xtitle,fontsize=12)
		plt.ylabel(ytitle,fontsize=12, horizontalalignment='center')
		if not colourParetoPointAtThresh == -1:
			try:
				colour_point = [c for c in pareto_list if c[0] > colourParetoPointAtThresh]
				colour_point = colour_point[-1]
				plt.plot(colour_point[0], colour_point[1], marker='*', markersize=20, markerfacecolor='yellow', mec='black', mew=2)
			except IndexError:
				print("No point above threshold: " + str(colourParetoPointAtThresh)) 
		plt.show()
		plt.savefig("pareto.png",bbox_inches='tight')
	return pareto_list

# get all the results below the thresh (i.e. not just the best result, but the entire set of configurations which match this result condition)
def getOverallLTEThresh( resultsum, thresh):
	return [x for x in resultsum if x[0][0] <= thresh]

# get the last value in the Pareto front of the graph section below the thresh (i.e. get the best result below the thresh)
def getOverallBestLTEThresh( resultsum, thresh):
	return getOverallPareto(getOverallLTEThresh(resultsum, thresh), False)[-1] 

# same as getOverallLTEThresh, but for getting results above the thresh (use for TP rate)
def getOverallGTEThresh( resultsum, thresh):
	return [x for x in resultsum if x[0][0] >= thresh]

# get the last value in the Pareto front of the graph section above the thresh (i.e. get the best above below the thresh)
def getOverallBestGTEThresh( resultsum, thresh):
	return getOverallParetoMax(getOverallGTEThresh(resultsum, thresh), False)[-1] 

# root specific results above a specified thresh
def getRootSpecGTEThresh( root_spec_res, root, thresh):
	return [x for x in root_spec_res[root] if x[0][0] >= thresh]

# get the last value in the Pareto front of the graph section above the thresh, for a specific root
def getRootSpecBestGTEThresh( root_spec_res, root, thresh):
	return getOverallParetoMax(getRootSpecGTEThresh(root_spec_res, root, thresh), False)[-1] 

# get set of all points on the Pareto front optimal for a particular thresh, if there are duplicate points 
# (if there are not duplicate points it acts the same as getRootSpecBestGTEThresh)
def getSetWithParetoMaxRes( root_spec_res, root, thresh):
	results_above_thresh = getRootSpecGTEThresh( root_spec_res, root, thresh)
	if len(results_above_thresh) == 0:
		return []
	last = getOverallParetoMax(results_above_thresh)[-1]
	return [x for x in results_above_thresh if x[0][0] == last[0] and x[0][1] == last[1]]

# get a dataframe for the overallBestGTEThresh results (TP rate, num TP, configuration, num FP, num KU, the index in the results dataframe, 
# and the overall occurrence and project count), for a list of threshs
def generateTableBestGTE( resultsum, threshVals, results, df_with_path, recall_denom = -1):
	retVal = pd.DataFrame([getOverallBestGTEThresh( resultsum, thresh) for thresh in threshVals], columns = ['TP_thresh', 'num_TP', 'res'])
	retVal.insert(loc=0, column = 'tested_thresh', value = threshVals)
	retVal['configs'] = retVal['res'].apply( lambda x: x[0][2])
	retVal['num_FP'] = retVal['res'].apply( lambda x: x[0][3])
	retVal['num_KU'] = retVal['res'].apply( lambda x: x[0][4])
	retVal['index'] = retVal['res'].apply( lambda x: x[1])
	retVal['overall_counts'] = retVal.apply( getOverallCountAndProjCount, args=( results, df_with_path), axis=1)
	if recall_denom > 0:
		retVal["recall"] = retVal.num_TP / recall_denom
	retVal.drop(['res'], inplace = True, axis=1)
	return(retVal)

# ---------------------------------------------------------------------------------------------------------------------------------- computing broken/correct

# --------------------------------------------------------------------------------------------------------------- the core processing functions

# sum up the tail of the frequency histograms for both portals and eventnames respectively
# this is just a cumulative sum (where the duplicate frequencies are included in the count for this frequency)
# so, for example, if there are 3 events with frequency 1 and 1 with frequency 2, the LTEFreq for those with frequency 1
# would all be 3, and the LTEFreq for the one with frequency 2 would be 5 (2 + 3*1)
def addLTEFreqsToFrame( prdat): 
	prdat.sort_values(['freq'], inplace=True)
	addCumFreqsToFrame(prdat, 'eventname', 'ltefreq_p')
	addCumFreqsToFrame(prdat, 'portal', 'ltefreq_e')

# compute the cumulative frequencies described above
def addCumFreqsToFrame( prdat, col_name, out_col_name):
	reldup = (~prdat[[col_name, 'freq']].duplicated()).astype(int) # compute the rows which are duplicate frequencies
	# now, compute the sum of all the duplicate frequencies (for each event)
	# the multiplication by reldup is a hack so that the cumulative frequency includes duplicate frequencies but only once
	# the result is that the first of a duplicate row (i.e. duplicate frequency for an event) has the total sum of all its duplicates
	# and the rest of the duplicate rows have intermsum = 0. then, when we compute the cumulative sum they all have the same value
	# also note that this only works because we sort the frame by frequency before calling this function (as in addLTEFreqsToFrame)
	prdat['intermsum'] = prdat.groupby([col_name, 'freq'])['freq'].transform('sum')*reldup 
	prdat[out_col_name] = prdat.groupby([col_name])['intermsum'].transform('cumsum')

# function to determine if a particular (portal, eventname) pair is broken
# this is part of a row which includes the relevant frequencies, so basically this 
# is just a wrapper for the binomial CDF computations and relevant comparisons
# returns a string specifying the category (broken/correct/unknown)
def categorizePortalEnamePair( row, prare_e, prare_p, pconfe, pconfp):
	freq_eandp = row['freq']
	freq_sumeandp = row['ltefreq_p']
	freq_eandsump = row['ltefreq_e']
	freq_e = row['freq_e']
	freq_p = row['freq_p']
	if ((binom.cdf( freq_sumeandp, freq_p, prare_p) < pconfp) and (binom.cdf( freq_eandsump, freq_e, prare_e) < pconfe)):
		return 'Broken'
	elif ((binom.cdf( (freq_p - freq_eandp), freq_p, 1 - prare_p) < pconfp) and (binom.cdf( (freq_e - freq_eandp), freq_e, 1 - prare_e) < pconfe) and freq_eandp > 10):
		return 'Correct'
	else:
		return 'Unknown'

# adds the category to each row of the dataframe
def addCatToFrame( prdat, prare_e, prare_p, pconfe, pconfp):
	prdat['category'] = prdat.apply(categorizePortalEnamePair, args=(prare_e, prare_p, pconfe, pconfp), axis=1)

# --------------------------------------------------------------------------------------------------------------------done core processing functions

# function to avoid having to type "prdat[prdat["category"] == category]"
# just returns the slice of the dataframe with the specified category
def getCategoryFromCategorizedFrame( prdat, category):
	return prdat[prdat['category'] == category]

# return a dataframe which is the root-specific results (for the root passed in), 
# with the correct/broken results included
def getRootSpecificDFWithBroken( df, proot, prare_e, prare_p, pconfe, pconfp):
	prdat = df[df['proot'] == proot][['portal', 'eventname', 'freq', 'freq_e', 'freq_p']]
	addLTEFreqsToFrame(prdat)
	addCatToFrame(prdat, prare_e, prare_p, pconfe, pconfp)
	return prdat

# takes in a dataframe, which we have to get the list of proots from
# count these up (i.e. list in order of most to less frequent), and then access
# the relevant indices and return an array of these positions
def getRootsAtFreqIndices( df, inds):
	proots = Counter(df['proot'].values).most_common(max(inds) + 1) # get the list of roots, sorted by appearance frequency
	return [proots[i][0] for i in inds] # get the corresponding list of roots (the [0] is since in Counter it's a 
										# tuple with the freq, but we only want the names)

# returns a dictionary of dataframes (where the key is the root name) 
# for the roots specified by the list of indices
def getDFsFromRootIndices( df, inds):
	return getDFsFromRootNames( df, getRootsAtFreqIndices(df, inds))

# create a dictionary where the root names are keys and the values are 
# the dataframe for that root
def getDFsFromRootNames( df, rootNames):
	return {name: df[df['proot'] == name] for name in rootNames} 

# run addLTEsToFrame over every element in the dictionary of dataframes
def addLTEsToFramesInDict( dcf):
	return {name: addLTEFreqsToFrame( df) for name, df in dcf.items()}

# ------------------------------------------------------------------------------------------------------------------------- experiment
# code for running the experiments
# runner wrapper and main() that's a sample usecase

# "struct" to represent a parameter configuration
Pscs = namedtuple("Pscs", "prare_e prare_p pconfe pconfp")

# start with just a list of packages, but would be trivial to change this to be a list of 
# indices, or make it just run over all the packages
# sample use: runTests(dat, [Ps(0.05, 0.05, 0.05)], ['fs', 'http', 'net'])
def runTests( df, param_configs, pkgs_to_test, append = False):
	# now we need to actually run the experiment
	# run it over all the configs provided, for each package listed
	for ps in param_configs:
		filename = "pe" + str(ps.prare_e) + "_pp" + str(ps.prare_p) + "_pce" + str(ps.pconfe) + "_pcp" + str(ps.pconfp) + "_.csv"
		for pkg in pkgs_to_test:
			cur_frame = df[df['proot'] == pkg]
			addLTEFreqsToFrame(cur_frame)
			print("On package: " + pkg + "\n")
			print("Done adding LTE\n")
			addCatToFrame( cur_frame, ps.prare_e, ps.prare_p, ps.pconfe, ps.pconfp)
			# printDFToFile( cur_frame, filename)
			printCatResultsToFile(cur_frame, "correct_" + filename, "Correct", append)
			printCatResultsToFile(cur_frame, "broken_" + filename, "Broken", append)
			# printCatResultsToFile(cur_frame, "unknown_" + filename, "Unknown", append)
			append = True
		print("Done running: " + filename + "\n") 

# "struct" for the results of the stats computed for each run (i.e. for a particular parameter configuration)
ExpStats = namedtuple("ExpStats", "diagnosed root_results overall_TP_count overall_FP_count overall_U_U_count overall_U_count overall_TP_rate overall_FP_rate overall_U_rate")

# running settings
# currently set to those used for the results in the paper
# the settings mean :
CONSERVATIVE_RUN = True # root count is TP + FP (where FP includes KU)
CONSERVATIVE_RATES = True # rates are TP / (FP + TP)
UNIQUE_RES = True # count all occurrences, not just the existence 
# compute a set of stats about true and false positives, overall and per package
# given a dataframe of the ground truth for both broken and correct pairs, and use this to measure the correctness
# of our computed results, also passed in as dataframes
def computeStats( computed_broken, computed_correct, known_broken, known_correct, known_knownUnknown, check_errors = True, pkgs_to_ignore = []):
	computed_broken.columns = ['freq','portal', 'eventname']
	computed_correct.columns = ['freq','portal', 'eventname']
	diagnosed = []
	if check_errors:
		diagnosed = pd.merge(computed_broken, known_broken, how='left',indicator='True_positive')
		diagnosed['False_positive'] = pd.merge(computed_broken, known_correct, how='left',indicator='False_positive')['False_positive']
	else:
		diagnosed = pd.merge(computed_correct, known_correct, how='left',indicator='True_positive')
		diagnosed['False_positive'] = pd.merge(computed_correct, known_broken, how='left',indicator='False_positive')['False_positive']
	diagnosed['Known_Unknown'] = pd.merge(diagnosed, known_knownUnknown, how='left',indicator='Known_Unknown')['Known_Unknown']
	diagnosed['True_positive'] = np.where(diagnosed.True_positive == 'both', 1, 0)
	diagnosed['False_positive'] = np.where(diagnosed.False_positive == 'both', 1, 0)
	diagnosed['Known_Unknown'] = np.where(diagnosed.Known_Unknown == 'both', 1, 0)
	diagnosed['Known_Unknown'] = np.where((diagnosed.False_positive == 0) & (diagnosed.True_positive == 0) & (diagnosed.Known_Unknown == 1), 1, 0)
	diagnosed['Unknown_Unknown'] = np.where((diagnosed.False_positive == 0) & (diagnosed.True_positive == 0) & (diagnosed.Known_Unknown == 0), 1, 0)
	if not UNIQUE_RES:
		diagnosed['True_positive'] *= diagnosed['freq']
		diagnosed['False_positive'] *= diagnosed['freq']
		diagnosed['Known_Unknown'] *= diagnosed['freq']
		diagnosed['Unknown_Unknown'] *= diagnosed['freq']
	# here we count unknown as false
	diagnosed['False_positive'] += diagnosed['Known_Unknown']
	# now, the "diagnosed" dataframe has the list of true and false positives
	# what are some interesting stats?
	# add the portal root so we can get data by root, easily
	diagnosed['proot'] = diagnosed['portal'].apply(getPortalRoot)
	for pkg in pkgs_to_ignore:
		diagnosed = diagnosed[diagnosed.proot != pkg] # delete rows with packages we're excluding
	diagnosed['root_count'] = diagnosed.groupby(['proot'])['freq'].transform('count')
	if not UNIQUE_RES: 
		diagnosed['root_count'] = diagnosed.groupby(['proot'])['freq'].transform('sum') 
	diagnosed['root_TP_count'] = diagnosed.groupby(['proot'])['True_positive'].transform('sum')
	diagnosed['root_FP_count'] = diagnosed.groupby(['proot'])['False_positive'].transform('sum')
	diagnosed['root_U_count'] = diagnosed.groupby(['proot'])['Unknown_Unknown'].transform('sum')
	if CONSERVATIVE_RUN:
		diagnosed['root_count'] = diagnosed['root_TP_count'] + diagnosed['root_FP_count']
	diagnosed['root_TP_rate'] = diagnosed['root_TP_count'] / diagnosed['root_count'] # rate = # / total calculated
	diagnosed['root_FP_rate'] = diagnosed['root_FP_count'] / diagnosed['root_count'] 
	diagnosed['root_U_rate'] = diagnosed['root_U_count'] / diagnosed.groupby(['proot'])['freq'].transform('count')
	# get root specific results
	root_results = diagnosed[['proot', 'root_count', 'root_TP_count', 'root_FP_count', 'root_U_count', 'root_TP_rate','root_FP_rate', 'root_U_rate']].drop_duplicates()
	# get global results
	overall_TP_count = root_results['root_TP_count'].sum()
	overall_FP_count = root_results['root_FP_count'].sum()
	overall_U_U_count = diagnosed['Known_Unknown'].sum()
	overall_U_count = root_results['root_U_count'].sum()
	overall_TP_rate = overall_TP_count / diagnosed['freq'].count() # this is just the total number of results
	overall_FP_rate = overall_FP_count / diagnosed['freq'].count()
	overall_U_rate = overall_U_count / diagnosed['freq'].count()
	if not UNIQUE_RES:
		overall_TP_rate = overall_TP_count / diagnosed['freq'].sum() # this is just the total number of results
		overall_FP_rate = overall_FP_count / diagnosed['freq'].sum()
		overall_U_rate = overall_U_count / diagnosed['freq'].sum()
	if CONSERVATIVE_RATES:
		overall_TP_rate = overall_TP_count / (overall_TP_count + overall_FP_count)
		overall_FP_rate = overall_FP_count / (overall_TP_count + overall_FP_count)
	diagnosed['root_TP_count'] = diagnosed.groupby(['proot'])['True_positive'].transform('sum')
	diagnosed['root_FP_count'] = diagnosed.groupby(['proot'])['False_positive'].transform('sum')
	diagnosed['root_U_count'] = diagnosed.groupby(['proot'])['Unknown_Unknown'].transform('sum')
	overall_TP_count = root_results['root_TP_count'].sum()
	overall_FP_count = root_results['root_FP_count'].sum()
	overall_U_U_count = diagnosed['Known_Unknown'].sum()
	overall_U_count = root_results['root_U_count'].sum()
	# return set of results
	return ExpStats(diagnosed, root_results, overall_TP_count, overall_FP_count, overall_U_U_count, overall_U_count, overall_TP_rate, overall_FP_rate, overall_U_rate)

# this function takes in a list of (ps_stats.overall_FP_rate, ps_stats, and ps)
# in addition to the packages to test over and column name to examine and sort by
def getRootSpecificSortedResults( list_of_results, pkgs_to_test, col_name1, col_name2):
	# first, extract the (root_results, ps) pairs we want	
	root_res_ps_list = [(k[1].root_results, k[2]) for k in list_of_results]
	root_res_ps_list = list(zip(root_res_ps_list, range(len(root_res_ps_list))))
	root_dict = {}
	for pkg in pkgs_to_test:
		sorted_res = [(getColValSpecIfNone(k[0][0], pkg, col_name1, 0), 
			getColValSpecIfNone(k[0][0], pkg, col_name2,0), k[1]) for k in root_res_ps_list]
		sorted_res = list(zip(sorted_res, range(len(root_res_ps_list))))
		sorted_res.sort()
		root_dict[pkg] = sorted_res
	return root_dict

# get the column element if it exists, otherwise return a specified error value (ex. np.NaN, np.inf, 0, or whatever fits)
def getColValSpecIfNone( root_df, root, col_name, error_val):
	try:
		return root_df[root_df['proot'] == root][col_name].values[0]
	except IndexError as e:
		return error_val

# run the diagnoses for a list of configurations
# return the results in a list sorted by the overall_TP_rate
# note: the optional check_errors parameter specifies to check for correct/incorrect errors identified; if it's false we look at the 
# results over the correct API uses diagnosed
def getExperimentStats( param_configs, known_correct, known_broken, known_knownUnknown, check_errors = True, pkgs_to_ignore = [], debug_mode = True, data_dir = "."):
	# compute stats for each run
	# then, keep a running track of the returned ExpStats
	# at the end, can probably order the list by different metrics 
	results = []
	for ps in param_configs:
		if debug_mode:
			print("Analyzing: " + str(ps) + "\n")
		filename = "pe" + str(ps.prare_e) + "_pp" + str(ps.prare_p) + "_pce" + str(ps.pconfe) + "_pcp" + str(ps.pconfp) + "_.csv"
		computed_correct = pd.DataFrame() 
		computed_broken = pd.DataFrame() 
		try:
			computed_correct = pd.read_csv(data_dir + "/correct_" + filename, sep=',', header=None)
		except pd.errors.EmptyDataError:
			if debug_mode:
				print(data_dir + "/correct_" + filename + " is empty, using empty dataframe\n")
				continue
			if not check_errors: # can't merge on empty dataframe, so we'll get no results anyway
				continue
		try:
			computed_broken = pd.read_csv(data_dir + "/broken_" + filename, sep=',', header=None)
		except pd.errors.EmptyDataError:
			if debug_mode:
				print(data_dir + "/broken_" + filename + " is empty, using empty dataframe\n")
				continue
			if check_errors:
				continue
		ps_stats = computeStats( computed_broken, computed_correct, known_broken, known_correct, known_knownUnknown, check_errors, pkgs_to_ignore)
		results += [(ps_stats.overall_TP_rate, ps_stats, ps)]
	return sorted(results, key = lambda x: x[0])  # sort by the first tuple element, which is the TP rate

# gets a set of the new configs, to avoid repeating work
def addNewConfigs( configs, old_configs):
	return list(set(list(itertools.product(configs + old_configs, repeat=3))) - set(list(itertools.product(old_configs, repeat=3))))

def getPathAndProjCountFromRow( row, dfwp):
	relrows = dfwp[(dfwp['portal'] == row['portal']) & (dfwp['eventname'] == row['eventname'])]['projcount']
	return( (relrows.count(), relrows.sum()))

# take one package, get the best result for that package (above a specific threshold)
# and then, get the results for that configuration on all the other packages
def crossValidateFromRoot( root, thresh, root_spec_res, overallres, pkgs):
	# first, get the best results for the thresh
	# and, get the corresponding index
	root_bests = getSetWithParetoMaxRes( root_spec_res, root, thresh)
	if len(root_bests) == 0:
		return "NONE"
	inds_to_test = [x[1] for x in root_bests]
	# now, for each package, get the max sets from this list
	pkgs = list(set(pkgs) - set([root]))
	pkg_spec_best_ind = []
	tp_counts = []
	tp_rates = []
	for pkg in pkgs:
		tp_counts += [[getColValSpecIfNone(overallres[ind][1].root_results, pkg, 'root_TP_count', np.nan) for ind in inds_to_test]]
		tp_rates += [[getColValSpecIfNone(overallres[ind][1].root_results, pkg, 'root_TP_rate', np.nan) for ind in inds_to_test]]
		pkg_spec_best_ind += [(i, inds_to_test[i]) for i in range(len(inds_to_test)) if tp_counts[-1][i] == max(tp_counts[-1]) and tp_rates[-1][i] == max(tp_rates[-1]) 
																					and not (np.isnan(max(tp_counts[-1])) or np.isnan(max(tp_rates[-1])))]
	best_inds = Counter(pkg_spec_best_ind).most_common()[0]
	ret_frame = pd.DataFrame(pkgs, columns = ['proot'])
	ret_frame['TP_count'] = [ c[best_inds[0][0]] for c in tp_counts]
	ret_frame['TP_rate'] = [ c[best_inds[0][0]] for c in tp_rates]
	return( (ret_frame, root, overallres[best_inds[0][1]][2], best_inds[1], 
				getColValSpecIfNone(overallres[best_inds[0][1]][1].root_results, root, 'root_TP_rate', np.nan), overallres[best_inds[0][1]][1].overall_TP_rate))

# run the cross-validation on all the packages, for the specified threshold
def crossValidateForRootsAtThresh( thresh, root_spec_res, overallres, pkgs):
	configs = []
	ranges = []
	numnans = []
	tp_rate = []
	global_tprate = []
	for pkg in pkgs:
		cv = crossValidateFromRoot( pkg, thresh, root_spec_res, overallres, pkgs)
		if cv == 'None':
			tp_rate += ['NONE']
			global_tprate += ['NONE']
			configs += ['NONE']
			numnans += ['NONE']
			ranges += ['NONE']
			continue
		tp_rate += [cv[4]]
		global_tprate += [cv[5]]
		configs += [cv[2]]
		numnans += [(len(pkgs) - 1) - cv[0]['TP_rate'].count()]
		fullvals = list(filter(lambda x: x > 0 and not np.isnan(x), cv[0]['TP_rate'].values))
		ranges += [(min(fullvals), max(fullvals))]
	ret_frame = pd.DataFrame(pkgs, columns = ['proot'])
	ret_frame['tp_rate'] = tp_rate
	ret_frame['configs'] = configs
	ret_frame['ranges'] = ranges
	ret_frame['numnans'] = numnans
	ret_frame['overall_tprate'] = global_tprate
	return( ret_frame)

# a different setup of cross-validation, where instead of partitioning the data per package we do a random split
# over the entire set of results labelled
# this is more standard cross validation, and it's the approach we used in the paper
def crossValidateAtThresh(thresh, known_correct, known_broken, known_knownUnknown, param_configs, numruns, split_by_freq = -1, data_dir = "."):
	retVal = []
	for i in range(numruns):
		print("On run: " + str(i) + "\n")
		# get a random sampling of half the training set
		# if we're splitting by a given frequency, make sure the known_broken is split around that too
		# note: we're hardcoding this to be training for incorrect, but could easily change this to be 
		# general and train for correct instead
		test_broken = None
		compl_broken = None
		if split_by_freq != -1:
			relevant_broken = known_broken[(known_broken.freq_e > split_by_freq) & (known_broken.freq_p > split_by_freq)]
			# irrelevant_broken = relevant_broken.merge(known_broken, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
			test_broken = relevant_broken.sample(int(len(relevant_broken)/2))
			compl_broken = test_rel_broken.merge(relevant_broken, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
			# test_irel_broken = irrelevant_broken.sample(int(len(irrelevant_broken)/2))
			# compl_irel_broken = test_irel_broken.merge(irrelevant_broken, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
			# test_broken = pd.concat([test_rel_broken, test_irel_broken])
			# compl_broken = pd.concat([compl_rel_broken, compl_irel_broken])
		else:
			test_broken = known_broken.sample(int(len(known_broken)/2))
			compl_broken = test_broken.merge(known_broken, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
		test_correct = known_correct.sample(int(len(known_correct)/2))
		compl_correct = test_correct.merge(known_correct, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
		test_ku = known_knownUnknown.sample(int(len(known_knownUnknown)/2))
		compl_ku = test_ku.merge(known_knownUnknown, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
		# run the experiment over the data we've picked
		results = getExperimentStats( param_configs, test_correct, test_broken, test_ku, True, [], False, data_dir)
		check = [(np.inf if np.isnan(k[0]) else k[0], k[1].overall_TP_count, k[2], k[1].overall_FP_count, k[1].overall_U_U_count) for k in results] # get a more readable list without the giant "diagnosed" frames
		check = list(zip(check, list(range(len(check)))))
		check.sort()
		best_res = getOverallBestGTEThresh(check, thresh)
		best_config = best_res[2][0][2]
		compl_res = getExperimentStats( [best_config], compl_correct, compl_broken, compl_ku, True, [], False, data_dir)
		# return: 
		retVal += [(best_res, compl_res)]
	return(retVal)

# classic k-fold cross validation
def kfoldCrossValidateAtThresh(thresh, known_correct, known_broken, known_knownUnknown, param_configs, k, data_dir = "."):
	correct_splits = splitDFIntoK( known_correct, k)
	broken_splits = splitDFIntoK( known_broken, k)
	ku_splits = splitDFIntoK( known_knownUnknown, k)
	retVal = []
	for fold in range(k):
		print("On fold: " + str(fold) + "\n")
		validation_broken = broken_splits[fold]
		validation_correct = correct_splits[fold]
		validation_ku = ku_splits[fold]
		train_broken = validation_broken.merge(known_broken, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
		train_correct = validation_correct.merge(known_correct, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
		train_ku = validation_ku.merge(known_knownUnknown, indicator='match', how='right')[lambda x: x.match=='right_only'].drop('match',1)
		# run the experiment over the data we've picked
		results = getExperimentStats( param_configs, train_correct, train_broken, train_ku, True, [], False, data_dir)
		check = [(np.inf if np.isnan(k[0]) else k[0], k[1].overall_TP_count, k[2], k[1].overall_FP_count, k[1].overall_U_U_count) for k in results] # get a more readable list without the giant "diagnosed" frames
		check = list(zip(check, list(range(len(check)))))
		check.sort()
		best_res = getOverallBestGTEThresh(check, thresh)
		best_config = best_res[2][0][2]
		compl_res = getExperimentStats( [best_config], validation_correct, validation_broken, validation_ku, True, [], False, data_dir)
		# return: 
		retVal += [(best_res, compl_res)]
	return(retVal)		

def splitDFIntoK( df, k):
	indices = np.arange(len(df))
	np.random.shuffle(indices)
	inc = int(len(df)/k)
	df_splits = []
	for i in range(k - 1):
		rel_indices = indices[ i*inc : (i+1)*inc]
		df_splits += [ df.iloc[rel_indices]]
	df_splits += [ df.iloc[ indices[ (k-1)*inc :]]]
	return( df_splits)


# take the output from cross validation and turn it into a nice table
# the results we want: generated TP, generated num TP, config, compl TP, compl num TP
def getCrossValidationDF( cv_results, recall_denoms = (-1, -1)):
	ret_frame = pd.DataFrame([c[0][0] for c in cv_results], columns=['Gen TP %'])
	if recall_denoms[0] != -1:
		ret_frame["Gen recall"] = [c[0][1]/recall_denoms[0] for c in cv_results]
	ret_frame['Gen num TP'] = [c[0][1] for c in cv_results]
	ret_frame['Configs'] = [c[0][2][0][2] for c in cv_results]
	ret_frame['Compl TP %'] = [c[1][0][1].overall_TP_rate for c in cv_results]
	if recall_denoms[1] != -1:
		ret_frame["Compl recall"] = [c[1][0][1].overall_TP_count/recall_denoms[1] for c in cv_results]
	ret_frame['Compl num TP'] = [c[1][0][1].overall_TP_count for c in cv_results]
	return( ret_frame)

# compute recall rate for cross-validation results -- add it to the dataframe
# second parameter is the number of total bugs (note: unique ones, not total)
# this only makes sense to use if we're doing analysis over unique bugs, not total occurrences
def displayRecallForCrossValDF( cv_df, recall_denom):
	cv_df["Gen recall"] = cv_df['Gen num TP'] / (recall_denom / 2)
	cv_df["Compl recall"] = cv_df['Compl num TP'] / (recall_denom / 2)

# for a row in the results dataframe (recall: this specifies a particular portal, eventname pair) get the total
# number of occurrences of this pattern, and the number of projects it appears in
def getOverallCountAndProjCount( row, results, df_with_path):
	rel_df = results[row['index']][1].diagnosed
	rel_df = rel_df[rel_df.True_positive == 1]
	total_TP = rel_df['freq'].sum()
	total_proj = len(pd.merge(rel_df[['portal', 'eventname']], df_with_path[['portal', 'eventname', 'path']], how='left', indicator='match')['path'].unique())
	return((total_TP, total_proj))

# return some overall stats (including overall count and project count) for a list of indices in the results dataframe
def getResultForSetOfInds( res_inds, results, df_with_path):
	cur_frame = results[res_inds[0]][1].diagnosed[['freq','portal','eventname','True_positive','False_positive']]
	all_paths = pd.merge(cur_frame[['portal', 'eventname']], df_with_path[['portal', 'eventname', 'path']], how='left')['path'].values.tolist()
	for ind in list(set(res_inds) - set([res_inds[0]])):
		inloop_frame = results[ind][1].diagnosed[['freq','portal','eventname','True_positive','False_positive']]
		inloop_frame['True_positive'] = inloop_frame.groupby(['portal','eventname'])['True_positive'].transform('sum')
		inloop_frame['False_positive'] = inloop_frame.groupby(['portal','eventname'])['False_positive'].transform('sum')
		inloop_frame['freq'] = inloop_frame.groupby(['portal','eventname'])['freq'].transform('sum')
		inloop_paths = pd.merge(inloop_frame[['portal', 'eventname']], df_with_path[['portal', 'eventname', 'path']], how='left')['path'].values.tolist()
		merged = pd.merge(cur_frame, inloop_frame, how='left', indicator='match')
		right = pd.merge(cur_frame, inloop_frame, how='right', indicator='match')
		right = right[right['match'] == 'right_only']
		merged = merged.append(right)
		merged.drop(['match'], inplace=True, axis=1)
		merged.drop_duplicates(inplace = True)
		cur_frame = merged
		all_paths += [i for i in inloop_paths if not i in all_paths] # only add new projects
	cur_frame['True_positive'] *= cur_frame['freq']
	cur_frame['False_positive'] *= cur_frame['freq']
	num_tp_pairs = cur_frame[cur_frame['True_positive'] > 0]['freq'].sum()
	num_pairs = cur_frame[cur_frame['False_positive'] > 0]['freq'].sum() + num_tp_pairs
	tp_rate = num_tp_pairs / num_pairs
	total_TP = cur_frame['True_positive'].sum()
	# total_proj = len(pd.merge(cur_frame[['portal', 'eventname']], df_with_path[['portal', 'eventname', 'path']], how='left', indicator='match')['path'])
	total_proj = len(all_paths)
	return( (cur_frame, num_tp_pairs, tp_rate, total_TP, total_proj))

# get the overall stats for all the results above a particular TP rate threshold
def getOverallSummaryResAboveThresh( resultsum, thresh, results, df_with_path):
	return getResultForSetOfInds( [c[1] for c in getOverallGTEThresh( resultsum, thresh)], results, df_with_path)

# return the time taken to run the data generation for one configuration
def timeOneRun(dat, pkgs_to_test, test1p):
	t1 = time.time()
	runTests(dat, test1p, pkgs_to_test)
	return( time.time() - t1)

# function to return the select statement for a query to find <ap, e> pairs for a particular ap pname and e ename
def generatePairSpecificQuery( pname, ename):
	return("from ListenNode ln, Portal p, DataFlow::Node pen,\n" + 
		"where ln.getBase*() = pen and pen = p.getAnExitNode(_)\n" + 
		"and n.getEventName() = \"" +  ename + "\" and p.toString() = \"" + pname + "\"\n" + 
		"select ln, ln.asExpr().getLocation(), p, pen" )

# function that takes in a dataframe of (portal, events) and returns a list of counts per portal root
# used to get the totals from the ground truth for each package analyzed
def getRootTotals( df):
	df['proot'] = df['portal'].apply(getPortalRoot)
	to_ret = df.groupby(['proot']).count()
	df.drop(['proot'], axis=1, inplace=True)
	return( to_ret)
	

# get the results (from the results list computed in main) with a specific configuration
def getResultsWithConfig( results, prare_e, prare_p, pconfe, pconfp):
	return( [results[i] for i in range(len(results)) if results[i][2] == Pscs(prare_e, prare_p, pconfe, pconfp)])

# determine if a portal (in a row) contains a particular string
def portalContainsString( row, s):
	return( row.portal.count( s) > 0)

# function to simplify a portal
# right now we hardcode the list of things to simplify but this might change
def condensePortal( portal):
	# list of things to condense
	# if we see more than one of these in a succession in the access path, remove it and replace with just one
	dups_to_rem = ["(member Stream ", "(member Readable ", "(member Writable ", "(return (member on "]
	for dup in dups_to_rem:
		split_portal = portal.split(dup)
		# when we split on dup, sequential paths will show up as the empty string
		# now, remove all the empty strings that aren't in the first position
		indices = [i for i in range(len(split_portal)) if split_portal[i] == '']
		if indices.count(0) > 0:
			indices.remove(0) # don't remove if it's in the first position
		split_portal = [split_portal[i] for i in range(len(split_portal)) if indices.count(i) == 0]
		portal = dup.join(split_portal) # add the dup back in but only once
		# now, need to make sure to remove the right number of closing parens
		num_parens_to_remove = dup.count("(") * len(indices)
		portal = portal if num_parens_to_remove == 0 else portal[ : -num_parens_to_remove]
	return( portal)


# sample usecase 
def main():
	# first, read in the results of the data mining (note: this is for the version with alias removal)
	dat = processFile('MinedData/listen_merged_data.out')
	dat_with_files = pd.read_csv('MinedData/listen_merged_data_withFile.out', sep=',', header=None)
	dat_with_files.columns = ['proot', 'portal', 'eventname', 'projcount', 'path']
	dat_with_files.portal = dat_with_files.portal.apply(condensePortal) # if we're doing condensing
	dat_with_files.drop_duplicates(inplace=True)

	print("Done reading in the data\n\n")

	# then, set up an experiment and run it
	# this is the set of 4069 configurations we tested in the paper
	ncs = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1]
	vs = list(set(list(itertools.product(ncs, repeat=4)))) 
	param_configs_new = [Pscs(c[0] if c[0] != 1 else 0.25, c[1] if c[1] != 1 else 0.25, c[2], c[3]) for c in vs]

	pkgs_to_test = ['fs', 'http', 'net', 'child_process', 'zlib', 'https', 'socket.io', 
					'socket.io-client', 'stream', 'readable-stream', 'ws', 'process', 'tls', 
					'events', 'cluster', 'readline', 'http2', 'repl']

	# this line runs all the experiments!! 
	# it takes a long time to run (~ 10 hours on ThinkPad P43s with 32 GB RAM)
	runTests(dat, param_configs, pkgs_to_test, True)

	# read in the set of labelled data
	new_known_correct = pd.read_csv('GroundTruthGeneration/correct.csv', sep=',', header=None)
	new_known_correct.columns = ['portal', 'eventname']
	new_known_correct.portal = new_known_correct.portal.apply(condensePortal)
	new_known_correct.drop_duplicates(inplace=True)
	new_known_broken = pd.read_csv('GroundTruthGeneration/broken.csv', sep=',', header=None)
	new_known_broken.columns = ['portal', 'eventname']
	new_known_broken.portal = new_known_broken.portal.apply(condensePortal)
	new_known_broken.drop_duplicates(inplace=True)
	new_known_knownUnknown = pd.read_csv('GroundTruthGeneration/knownUnknown.csv', sep=',', header=None)
	new_known_knownUnknown.columns = ['portal', 'eventname']
	new_known_knownUnknown.portal = new_known_knownUnknown.portal.apply(condensePortal)
	new_known_knownUnknown.drop_duplicates(inplace=True)

	# run the experiments
	new_results = getExperimentStats( param_configs_new, new_known_correct, new_known_broken, new_known_knownUnknown, True, [], True, "list_results")
	new_check = [(np.inf if np.isnan(k[0]) else k[0], k[1].overall_TP_count, k[2], k[1].overall_FP_count, k[1].overall_U_U_count) for k in new_results] # get a more readable list without the giant "diagnosed" frames
	new_check = list(zip(new_check, list(range(len(new_check)))))
	new_check.sort()
	new_graphcheck = [c for c in new_check if (c[0][2].pconfe != 1 or c[0][2].pconfp != 1)] # optional: remove all the results with both threshs set to 1, since this makes no sense as results and skews the graph axis

	threshs = np.arange(1, 0.79, -0.01).tolist()
	reltable = generateTableBestGTE(check, threshs, results, dat_with_files)

	root_spec_FP_res = getRootSpecificSortedResults(results, pkgs_to_test, 'root_TP_rate', 'root_TP_count')

	correct_results = getExperimentStats( param_configs, known_correct, known_broken, False, ['inherits', 'jquery'], True, ".")
	correct_check = [(k[0], k[1].diagnosed['proot'].count(), k[2]) for k in correct_results] # get a more readable list without the giant "diagnosed" frames
	correct_root_spec_FP_res = getRootSpecificSortedResults(correct_results, pkgs_to_test, 'root_FP_rate', 'root_count')

	print("Done the tests!")

# main()
