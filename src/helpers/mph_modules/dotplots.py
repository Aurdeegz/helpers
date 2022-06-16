"""

"""
############################################################################################################
#
# Imports 

# Pyplot has the majority of the matplotlib plotting functions
# and classes, including figures, axes, etc.
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.patches import Patch

import numpy as np

import copy

try:
    from .. import general_helpers as gh
    from .. import argcheck_helpers as ah
    from .. import stats_helpers as sh
    from ..mpl_plotting_helpers import handle_colours, add_errorbar, colours, tab_colours, update_ticks
except:
    from pathlib import Path
    hpath = Path(__file__).parent.absolute()
    help_path = ""
    for folder in str(hpath).split("/")[1:-1]:
        help_path = f"{help_path}/{folder}"
    import sys
    sys.path.insert(0,help_path)
    import general_helpers as gh
    import argcheck_helpers as ah
    import stats_helpers as sh
    from mpl_plotting_helpers import handle_colours, add_errorbar, colours, tab_colours, update_ticks

#
#
############################################################################################################
#
# Functions

def get_xcoords(data_matrix, width = 7):
    """
    =================================================================================================
    get_xcoords(data_matrix, width)
    
    Given a matrix of data (row oriented), find an x position for the center of the group on
    the plot.
    
    =================================================================================================
    Arguments:
    
    data_matrix -> A list of 2-lists, where each sublist contain a group label and replicate
                   measurements for the group
    width       -> An integer defining how many x-values should be used  to plot the group.
    =================================================================================================
    Returns: The x coordinate lists for each group.
    =================================================================================================
    """
    assert width <= 10, "Width must be between 0 and 10"
    width = ah.check_type(width, int, "The given width is not an integer. Please make it an integer.")
    # Get the number of groups in the data matrix
    n = len(data_matrix)
    # Initialize a list for the x coordinates
    x_coords = []
    # Loop over the index of each list
    for i in range(n):
        # Get the size of the ith group
        m = len(data_matrix[i][1])
        # and initialize x coordinates for this group
        xi_coords = []
        # Loop over the number of times width splits the list
        # Integer division rounds down, so for m in [0,width],
        # m // width = 0. The +1 tells us to loop once for that group
        for j in range(m//width +1):
            xi_coords += [(i+1) - 0.2 + 0.05*(k) for k in range(width)]
        if m%2 == 0:
            # Then this is an even number, so get even around the center
            x_coords.append(xi_coords[(width//2 + 1) - m//2:width//2+m//2+1])
        else:
            # Otherwise, this is odd so we want all tha points
            x_coords.append(xi_coords[(width//2 + 1) - m//2:(width//2 + 1) + m//2+1])
        print(x_coords)
    return x_coords, data_matrix

def get_data_info(data_matrix, width = 7):
    """
    =================================================================================================
    get_data_info(data_matrix, width)
    
    Given the matrix of data (row oriented), find all information pertinant to plotting and return
    it as a dictionary. These include: centers, x-coordinates, means, standard deviations, standard
                                       errors of the mean
    
    =================================================================================================
    Arguments:
    
    data_matrix -> A list of 2-lists, where each sublist contain a group label and replicate
                   measurements for the group
    width       -> An integer defining how many x-values should be used  to plot the group.
    =================================================================================================
    Returns: A dictionary containing information pertinant to plotting.
    =================================================================================================
    """
    assert width <= 10, "Width must be between 0 and 10"
    width = ah.check_type(width, int, "The given width is not an integer. Please make it an integer.")
    # Get the number of groups in the data matrix
    bins = len(data_matrix)
    # Get the x coordinates of the data. NOTE: get_xcoords used to sort
    # the data and return it, but I stopped doing that
    xs, data = get_xcoords(data_matrix, width = width)
    # Next we want to define the centers of each group, which is just the mean of the x coordinates
    centers = [sh.mean(x_coords) for x_coords in xs]
    # Next, we need to get the mean values of the data
    means = [sh.mean(item[1]) for item in data]
    # and the standard deviations
    sds = [sh.standard_deviation(item[1]) for item in data_matrix]
    # and the standard error of the means
    sems = [sh.sem(item[1]) for item in data]
    # and finally we return these values as a dictionary
    return {"centers" : centers,
            "xs"      : xs,
            "means"   : means,
            "sds"     : sds,
            "sems"    : sems}, data

def make_ymax(mpl_axes,
              labelled_groups):
    """
    =================================================================================================
    make_ymax(mpl_axes, labelled_groups)
    
    Given a matplotlib axes object and the labelled data, return a new ymax value for the plot.
    =================================================================================================
    Arguments:
    
    mpl_axes        -> A matplotlib axes object
    labelled_groups -> A list of 2-lists that have labelled data.
    =================================================================================================
    Returns: a float defining the new ymax
    =================================================================================================
    """
    # grab the current y tick positions from the axes
    current = list(mpl_axes.get_yticks())
    # and find the y value range
    diff = abs( max(current) - min(current))
    # Now check how many groups there are. If there are less than 5,
    if 1 <= len(labelled_groups) < 5:
        # then extend the ymin by 30%
        return max(current) + 0.3*diff
    # if there are between 5 and 10
    elif 5 <= len(labelled_groups) < 10:
        # then extend the ymin by 45%
        return max(current) + 0.45*diff
    # Otherwise, 
    else:
        # extend the ymin by 60%
        return max(current) + 0.6*diff

def make_ymin(mpl_axes,
              labelled_groups):
    """
    =================================================================================================
    make_ymin(mpl_axes, labelled_groups)
    
    Given a matplotlib axes object and the labelled data, return a new ymin value for the plot.
    =================================================================================================
    Arguments:
    
    mpl_axes        -> A matplotlib axes object
    labelled_groups -> A list of 2-lists that have labelled data.
    =================================================================================================
    Returns: a float defining the new ymin
    =================================================================================================
    """
    # grab the current y tick positions from the axes
    current = list(mpl_axes.get_yticks())
    # and find the y value range
    diff = abs( max(current) - min(current))
    # Now check how many groups there are. If there are less than 5,
    if 1 <= len(labelled_groups) < 5:
        # then extend the ymin by 10%
        return min(current) - 0.1*diff
    # if there are between 5 and 10
    elif 5 <= len(labelled_groups) < 10:
        # then extend the ymin by 15%
        return min(current) - 0.15*diff
    # Otherwise, 
    else:
        # extend the ymin by 20%
        return min(current) - 0.2*diff

def add_relative_axis(mpl_axes,
                      rel_data,
                      data_matrix,
                      info_dict,
                      fontdict = {"fontfamily" : "sans-serif",
                                  "font" : "Arial",
                                  "ha" : "left",
                                  "fontweight" : "bold",
                                  "fontsize" : "12"},
                      ylabel = "Fold Change",
                      axis_fontdict = dict(font = "Arial",
                                         fontsize = 14,
                                         fontweight = "bold",
                                         rotation = 270, va = "baseline"),
                      remove_top_spine = True):
    """
    =================================================================================================
    add_
    """
    n = len(data_matrix)
    rel_index = [i for i in range(n) if rel_data in data_matrix[i]]
    data_scaled = [(item[0], [value / info_dict["means"][rel_index[0]] 
                              for value in item[1]])
                   for item in data_matrix]
    new_y = mpl_axes.twinx()
    if remove_top_spine:
        new_y.spines["top"].set_visible(False)
    ymax = make_ymax(mpl_axes,data_scaled)
    current_lims = mpl_axes.get_ylim()
    new_y.set_ylim(current_lims[0] / info_dict["means"][rel_index[0]],
                   current_lims[1] / info_dict["means"][rel_index[0]])
    for i in range(n):
        new_y.scatter(info_dict["xs"][i], data_scaled[i][1], alpha = 0)
    update_ticks(new_y, which = "y", scino = False, fontdict = fontdict)
    current_lims = mpl_axes.get_ylim()
    new_y.set_ylim(current_lims[0] / info_dict["means"][rel_index[0]],
                   current_lims[1] / info_dict["means"][rel_index[0]])
    new_y.set_ylabel(ylabel,
                     **axis_fontdict)
    return None

def update_ylims(ymin, ymax, mpl_axes, labelled_groups):
    if ymin == ymax:
        ymin = make_ymin(mpl_axes, labelled_groups)
        ymax = make_ymax(mpl_axes, labelled_groups)
        mpl_axes.set_ylim(ymin, ymax)
    elif ymin == None and ymax != None:
        ymin = make_ymin(mpl_axes, labelled_groups)
        mpl_axes.set_ylim(ymin, ymax)
    elif ymin != None and ymax == None:
        ymax = make_ymax(mpl_axes, labelled_groups)
        mpl_axes.set_ylim(ymin, ymax)
    else:
        mpl_axes.set_ylim(ymin, ymax)
    return None

def update_ylims(axes, ymin, ymax):
    current = list(axes.get_ylim())
    if ymin == ymax and ymin == None:
        axes.set_ylim(*current)
        return
    elif ymin == None and ymax != None:
        axes.set_ylim(current[0], ymax)
        return
    elif ymin != None and ymax == None:
        axes.set_ylim(ymin, current[1])
        return
    else:
        axes.set_ylim(ymin, ymax)
        return

def update_xlims(mpl_axes_1, mpl_axes_2):
    xvals = list(mpl_axes_2.get_xlim())
    mpl_axes_1.set_xlim(*xvals)
    return

def plot_id_lines(mpl_axes_1, # Comparison axis
                  mpl_axes_2, # Dotplot_axis
                  labelled_data,
                  centers,  # Center value for each group
                  heights,  # list of highest comparison for the groups
                  colours = [],  # list of colours, one per group
                  plot_kwargs = {"alpha" : 0.5,
                                 "linestyle" : ":"}): 
    ax2_max = max(list(mpl_axes_2.get_ylim()))
    ax1_min = min(list(mpl_axes_1.get_ylim()))
    n = len(centers)
    if colours == []:
        colours = ["grey" for _ in range(n)]
    for i in range(n):
        mpl_axes_2.plot([centers[i], centers[i]],
                        [max(labelled_data[i][1]), ax2_max], colours[i], **plot_kwargs)
        mpl_axes_1.plot([centers[i], centers[i]],
                        [ax1_min, heights[i]], colours[i], **plot_kwargs)
    return None

# Functions to handle comparisons. Comparisons should be a dictionary of groups & pvalues
# from statistical tests in stats_helpers

# These functions are to format the comparisons such that we can
#    a) Filter them on significance if desired/required
#    b) Get the indices of the groups fromt he data
#    c) Get all of the x positions (centers) using the indices

def format_comps(comp_dict):
    # Return a list of [group 1, group 2] and [pval]
    groups = list(zip(comp_dict["Group 1"], comp_dict["Group 2"]))
    ps = comp_dict["pvalue"]
    combined = [[sorted(groups[i]), ps[i]] for i in range(len(ps))]
    return sorted(combined, key = lambda x: x[0][0])

def match_comp_to_data(comp_list, labelled_data):
    # Return the index for each group in the list
    inds = []
    labels = [item[0] for item in labelled_data]
    for item in comp_list:
        inds.append([sorted([labels.index(item[0][0]), labels.index(item[0][1])]), item[1]])
    inds = sorted(inds, key = lambda x: x[0])
    return inds

def make_xvals(comp_dict,
               labelled_data,
               centers,
               filter_by_alpha = False,
               alpha = 0.05):
    comp_list = format_comps(comp_dict)
    if filter_by_alpha:
        comp_list = [item for item in comp_list if item[1] < alpha]
        if len(comp_list) >= 19:
            return "toomany", "comps", "toplot"
        filtered = True
    elif len(comp_list) >= 19:
        print(f"Too many comparisons provided. Filtering by significance : {alpha}")
        comp_list = [item for item in comp_list if item[1] < alpha]
        if len(comp_list) >= 19:
            return "toomany", "comps", "toplot"
        filtered = True
    else:
        filtered = False
    data_inds = match_comp_to_data(comp_list, labelled_data)
    return [[centers[ind[0][0]], centers[ind[0][1]]] for ind in data_inds], [item[1] for item in data_inds], filtered

def make_sigstrings(value_list, sig_dict = {0.05 : "$*$",
                                            0.01 : "$**$",
                                            0.001 : "$**$$*$"}):
    sig_thresholds = list(sig_dict.keys())
    strings = []
    for val in value_list:
        sig = "n.s."
        for thresh in sig_thresholds:
            if val < thresh:
                sig = sig_dict[thresh]
        strings.append(sig)
    return strings

def generate_legend_string(p_or_q,
                           sig_dict,
                           omit = False):
    """
    """
    # Values in the dict are the representation
    # Keys in the dict are the cutoff
    items = list(sig_dict.items())
    #
    if omit:
        string = f"${p_or_q}\geq{items[0][0]}$ omitted\n"
    else:
        string = f"{'n.s.':<9}: ${p_or_q}\geq{items[0][0]}$"
    for item in items:
        centering = len(item[1]) + 5
        string = f"{string}\n{item[1]:<10}: ${p_or_q}<{item[0]}$"
    return string
    


def plot_comparisons(mpl_axes_1,                   # Comparison axes
                     mpl_axes_2,                   # Dotplot axes
                     labelled_data,                # 
                     comp_dict = {},
                     filter_by_alpha = False,
                     alpha = 0.05,
                     centers = [],
                     colours = [],
                     sig_dict = {0.05 : "$*$",
                                 0.01 : "$**$",
                                 0.001 : "$**$$*$"},
                     textdict = {"fontfamily" : "sans-serif",
                                 "font" : "Arial",
                                 "ha" : "left",
                                 "va" : "center",
                                 "fontweight" : "bold"},
                     return_height = True,
                     p_or_q = "p"):
    # Max comparisons hard set to 19, so make the y values for bars and text
    ys = [[i/19+0.02, i/19+0.02] for i in range(20)]
    text_y = [(ys[i][0] + ys[i+1][0])/2 for i in range(19)]
    # Handle anovas
    if comp_dict["id"][0] == "ANOVA":
        comp_dict["Group 1"] = [labelled_data[0][0]]
        comp_dict["Group 2"] = [labelled_data[-1][0]]
    # Need the xvalues for the bars and comparisons
    xs, pvals, filtered = make_xvals(comp_dict, labelled_data, 
                           centers, filter_by_alpha = filter_by_alpha,
                           alpha = alpha)
    if xs == "toomany":
        return None, None
    text_x = [sh.mean(item) for item in xs]
    # Make the significance strings
    pvals = make_sigstrings(pvals)
    # Keep track of the heights
    heights = {}
    maxtext = 0
    # Now we should be able to plot comparisons
    for i in range(len(xs)):
        mpl_axes_1.plot(xs[i], ys[i], color = "black", alpha = 0.5)
        mpl_axes_1.text(text_x[i], text_y[i], pvals[i], **textdict)
        # Update heights
        heights[xs[i][0]] = ys[i][0]
        heights[xs[i][1]] = ys[i][1]
        maxtext = text_y[i]
    # Turn the heights into a list
    if comp_dict["id"][0] == "ANOVA":
        heights = [ys[0][0] for _ in range(len(labelled_data))]
    else:
        heights = sorted([[key, value] for key, value in heights.items()], key = lambda x: x[0])
        heights = [item[1] for item in heights]
    plot_id_lines(mpl_axes_1, mpl_axes_2, labelled_data, centers, heights, colours = colours)
    if filtered:
        string = generate_legend_string(p_or_q, sig_dict, omit = True)
        #string = f"Test : {comp_dict['id'][0]}\n\n${p_or_q}\geq0.05$ omitted\n$*$     : $p < 0.05$\n$**$   : $p<0.01$\n$**$$*$ : $p<0.001$"
    else:
        string = generate_legend_string(p_or_q, sig_dict)
        #string = f"Test : {comp_dict['id'][0]}\n\nn.s.   : $p\geq 0.05$\n$*$       : $p < 0.05$\n$**$    : $p<0.01$\n$**$$*$   : $p<0.001$"
    handles, labels = mpl_axes_1.get_legend_handles_labels()
    handles.append(Patch(color="none", label = string))
    if return_height:
        return maxtext, handles
    else:
        return None, handles

def dotplot(labelled_groups,
            foldchange_axis = False,
            foldchange_group = None,
            comparisons = {},        # Args for plot_comparisons
            filename = "dotplot.pdf",
            save_file = True,
            colours = [],
            title = "Dotplot",
            xlabel = "",
            ylabel = "Abundance",
            ymin = None,
            ymax = None,
            errorbar = "sem",
            filter_by_alpha = False,
            alpha = 0.05, 
            sig_dict = {0.05 : "$*$",
                        0.01 : "$*$$*$",
                        0.001 : "$*$$*$$*$"},
            p_or_q = "p"):
    assert errorbar.lower() in ["sem", "sd"], "The accepted errorbar settings are Standard Error of the Mean (sem) or Standard Deviation (sd)"
    global tab_colours
    if type(colours) == str:
        colours = handle_colours(colours, len(labelled_groups), choice = "random")
        print(f"Colours chosen: {colours}")
    #
    info_dict, groups = get_data_info(labelled_groups)
    #
    fig, ax = plt.subplots(2,1,sharex = True, figsize = (2*len(labelled_groups), 10))
    #
    nrow = 2
    ncol = 1
    #
    gs = gridspec.GridSpec(nrow,ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    #
    ax1 = plt.subplot(gs[0])
    ax1.set_ylim(0,1)
    ax1.axis("off")
    #
    ax2 = plt.subplot(gs[1])
    ax2.spines["right"].set_visible(False)
    #
    print(info_dict)
    for i in range(len(groups)):
        if colours == []:
            ax2.scatter(info_dict["xs"][i], groups[i][1], edgecolors = "black")
        elif colours != []:
            ax2.scatter(info_dict["xs"][i], groups[i][1], edgecolors = "black", color = colours[i])
        if errorbar.lower() == "sem":
            add_errorbar(ax2, info_dict["centers"][i], info_dict["means"][i], info_dict["sems"][i])
        else:
            add_errorbar(ax2, info_dict["centers"][i], info_dict["means"][i], info_dict["sds"][i])
    # Axes related updates
    update_ylims(ax2, ymin, ymax)
    update_xlims(ax1,ax2)
    update_ticks(ax2, which = "y", scino = True)
    ax2.set_xticks(info_dict["centers"])
    update_ticks(ax2, which = "x", labels = [item[0] for item in groups], fontdict = {"fontfamily" : "sans-serif",
                                                                                      "font" : "Arial",
                                                                                      "ha" : "center",
                                                                                      "fontweight" : "bold",
                                                                                      "fontsize" : "12"})
    if foldchange_axis and foldchange_group != None:
        add_relative_axis(ax2, foldchange_group, groups, info_dict)
    ax2.set_xlabel(xlabel, **{"fontfamily" : "sans-serif",
                               "font" : "Arial",
                               "ha" : "center",
                               "fontweight" : "bold",
                               "fontsize" : "14"})
    ax2.set_ylabel(ylabel, **{"fontfamily" : "sans-serif",
                               "font" : "Arial",
                               "ha" : "center",
                               "fontweight" : "bold",
                               "fontsize" : "14"})
    # If comparisons are provided, plot them unless there are too many.
    if comparisons != {}:
        if comparisons["id"][0] == "ANOVA":
            comp_groups = groups
            comp_centers = info_dict["centers"]
            comp_tab_colours = tab_colours
        else:
            comp_labels = list(set(comparisons["Group 1"] + comparisons["Group 2"]))
            keep_comp_inds = [groups.index(item) for item in groups if item[0] in comp_labels]
            comp_groups = [g for g in groups if groups.index(g) in keep_comp_inds]
            comp_centers = [c for c in info_dict["centers"] if info_dict["centers"].index(c) in keep_comp_inds]
            comp_tab_colours = [t for t in tab_colours if tab_colours.index(t) in keep_comp_inds]
        if colours == []:
            title_height, handles = plot_comparisons(ax1, ax2, comp_groups, comparisons,
                                                     filter_by_alpha = filter_by_alpha, alpha = alpha,
                                                     centers = comp_centers,
                                                     colours = comp_tab_colours, sig_dict = sig_dict,
                                                     p_or_q = p_or_q)
        else:
            comp_colours = [t for t in colours if colours.index(t) in keep_comp_inds]
            title_height, handles = plot_comparisons(ax1, ax2, comp_groups, comparisons,
                                                     filter_by_alpha = filter_by_alpha, alpha = alpha,
                                                     centers = comp_centers,
                                                     colours = comp_colours, sig_dict = sig_dict,
                                                     p_or_q = p_or_q)
        if title_height != handles:
            ax1.text(sh.mean(list(ax1.get_xlim())), title_height + 0.1,
                     title, **{"fontfamily" : "sans-serif",
                               "font" : "Arial",
                               "ha" : "center",
                               "fontweight" : "bold",
                               "fontsize" : "16"})
            ax1.legend(handles = handles, loc = "center right", 
                       bbox_to_anchor = (0,0.5),
                       frameon = False)
        else:
            ax2.set_title("title", **{"fontfamily" : "sans-serif",
                                  "font" : "Arial",
                                  "ha" : "center",
                                  "fontweight" : "bold",
                                  "fontsize" : "16"})
    else:
        ax2.set_title(title, **{"fontfamily" : "sans-serif",
                                  "font" : "Arial",
                                  "ha" : "center",
                                  "fontweight" : "bold",
                                  "fontsize" : "16"})
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    if save_file:
        plt.savefig(filename)
    return ax

#
#
############################################################################################################