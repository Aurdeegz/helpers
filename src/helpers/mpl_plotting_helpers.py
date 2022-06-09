"""
=================================================================================================
Kenneth P. Callahan

Original Creation Date: 17 July 2021
Update 1: 6 June 2022

=================================================================================================
Python >= 3.8.5

mpl_plotting_helpers.py

This is a module meant to help with various types of plots. While Python has a ton of support
for plotting in MatPlotLib, some of the things aren't exactly intuitive. Many of these functions
are meant to help with plotting, wrapping MatPlotLib functions and performing formatting.

Currently type checking is not being done.

=================================================================================================
Dependencies:

   PACKAGES        VERSION
    matplotlib  ->  3.3.2
    numpy       ->  1.19.2

If the printed versions ever change, then change the versions here. These are the ones
in my current Python installation.
=================================================================================================
"""
print(f"Loading the module: helpers.mpl_plotting_helpers\n")
############################################################################################################
#
#     Importables

from pathlib import Path
help_path = Path(__file__).parent.absolute()
import sys
sys.path.insert(0,help_path)

# Import base matplotlib to show the version.
import matplotlib

# Pyplot has the majority of the matplotlib plotting functions
# and classes, including figures, axes, etc.
import matplotlib.pyplot as plt
#    cm is used for colourmaps
import matplotlib.cm as cm
#    ticker is used for placing labels on heatmaps
import matplotlib.ticker as ticker

from matplotlib import gridspec
from matplotlib.patches import Patch

# Numpy is used for creating arrays that matplotlib is able to handle
import numpy as np

# Copy is used for making copies of objects, rather than manipulating
# the original form of an object.
import copy

# Random is used for making pseudorandom selections from a list of items.
import random

# General helpers has a number of functions I use frequently in
# my scripts. They are all placed in that module purely for
# convenience and generalizability.
import general_helpers as gh

# The Pandas Helper file has scripts that help manage Pandas
# DataFrames, and perform various actions on lists of dataframes
import pandas_helpers as ph

import stats_helpers as sh

print(f"matplotlib    {matplotlib.__version__}")
print(f"numpy         {np.__version__}\n")

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = "sans-serif"

#
#
############################################################################################################
#
#     Global Variables

# This dictionary has some predefined sets of colours that can be used for
# barplot plotting. These colours should all be base matplotlib colours,
# so I can't imagine they'll be depricated soon.
colours = {"blues"  : ["steelblue", "cyan", "blue", "darkblue", "dodgerblue",
                       "lightblue", "deepskyblue"],
           "pinks"  : [ "mediumvioletred", "darkmagenta", "deeppink","violet", "magenta",
                       "pink", "lavenderblush"],
           "reds"   : ["darkred", "firebrick", "indianred", "red", "tomato", "salmon",
                      "lightcoral", "darksalmon", "mistyrose"],
           "purples": ["indigo", "mediumpurple", "purple",
                       "darkviolet", "mediumorchid", "plum", "thistle"],
           "greens" : ["darkolivegreen", "olivedrab", "green", "forestgreen",
                       "limegreen", "springgreen", "lawngreen"],
           "oranges": ["darkorange", "orange", "goldenrod", "gold", "yellow",
                       "khaki", "lightyellow"],
           "browns" : ["brown","saddlebrown", "sienna", "chocolate", "peru",
                      "sandybrown", "burlywood"],
           "monos"  : ["black", "dimgrey", "grey", "darkgrey",
                       "silver", "lightgrey", "gainsboro"],
           "cc"     : ["darkturquoise", "cyan", "thistle", "fuchsia", "violet"],
           "default": ["blue", "red", "green", "purple", "pink"],
           "all"    : ["darkblue", "steelblue", "blue", "dodgerblue", "deepskyblue",
                      "aqua", "lightblue", "cornflowerblue","darkmagenta", "mediumvioletred", 
                       "deeppink", "violet", "magenta", "pink", "lavenderblush",
                       "darkred", "firebrick", "indianred", "red", "tomato", "salmon",
                      "lightcoral", "darksalmon", "mistyrose",
                       "indigo", "rebeccapurple", "mediumpurple", "purple",
                       "darkviolet", "mediumorchid", "plum", "thistle",
                       "darkolivegreen", "olivedrab", "green", "forestgreen",
                       "limegreen", "springgreen", "lawngreen", "palegreen",
                       "darkorange", "orange", "goldenrod", "gold", "yellow",
                       "khaki", "lightyellow",
                       "brown","saddlebrown", "sienna", "chocolate", "peru",
                      "sandybrown", "burlywood", "wheat",
                       "black", "dimgrey", "grey", "darkgrey",
                       "silver", "lightgrey", "gainsboro", "white"]}

tab_colours = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

class MathTextSciFormatter():
    """
    For updating ticks to nicely formatted scientific notation. Based on
    the response to the following stackoverflow:
    https://stackoverflow.com/questions/25750170/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    """
    def __init__(self, before_dec = 1, after_dec = 2):
        self.b = before_dec
        self.a = after_dec
    def __call__(self, value, weight = "bold"):
        scino_form = f"{value:{self.b}.{self.a}e}"
        scino_form = scino_form.split(".")
        if "+" in scino_form[1]:
            scino_form[1] = scino_form[1].lower().split("e")
            if scino_form[1][1][0] == "+":
                scino_form[1][1] = scino_form[1][1][1:].lstrip("0")
            if scino_form[1][1] == "":
                scino_form[1][1] = "0"
        if weight == "bold":
            return fr"$\mathdefault{{{scino_form[0]}.{scino_form[1][0][:-1]}}} \times \mathdefault{{10}}^{{\mathdefault{{{scino_form[1][1]}}}}}$"
        elif weight == "italic":
            return fr"$\mathit{{{scino_form[0]}.{scino_form[1][0][:-1]} \times 10^{{{scino_form[1][1]}}}}}$"
        else:
            return fr"${scino_form[0]}.{scino_form[1][0][:-1]} \times 10^{{{scino_form[1][1]}}}$"

def _fix_numbers(a_list, roundfloat = 2):
    newlist = []
    for item in a_list:
        if int(item) == item:
            newlist.append(int(item))
        else:
            newlist.append(round(item,2))
    return newlist

def update_ticks(mpl_axes, which = "x", scino=False, scino_before = 1,
                 scino_after = 2, labels = [], rotation = 0,
                  fontdict = {"fontfamily" : "sans-serif",
                             "font" : "Arial",
                              "ha" : "right",
                              "fontweight" : "bold",
                              "fontsize" : "12"}):
    """
    """
    if which == "x" and labels == []:
        xticks = list(mpl_axes.get_xticks())
        mpl_axes.set_xticks(xticks)
        if "fontweight" not in list(fontdict.keys()):
            fontdict["fontweight"] = "bold"
        if scino:
            formatter = MathTextSciFormatter(before_dec = scino_before,
                                             after_dec = scino_after)
            mpl_axes.set_xticklabels([formatter(x, weight = fontdict["fontweight"]) for x in xticks], rotation = rotation,
                                **fontdict)
        else:
            xticks = _fix_numbers(xticks)
            mpl_axes.set_xticklabels([str(x) for x in xticks], rotation = rotation,**fontdict)
    elif which == "x" and labels != []:
        mpl_axes.set_xticklabels([str(x) for x in labels], rotation = rotation, **fontdict)
    elif which == "y" and labels == []:
        yticks = list(mpl_axes.get_yticks())
        mpl_axes.set_yticks(yticks)
        if "fontweight" not in list(fontdict.keys()):
            fontdict["fontweight"] = "bold"
        if scino:
            formatter = MathTextSciFormatter(before_dec = scino_before,
                                             after_dec = scino_after)
            mpl_axes.set_yticklabels([formatter(y, weight = fontdict["fontweight"]) for y in yticks],
                                **fontdict)
        else:
            yticks = _fix_numbers(yticks)
            mpl_axes.set_yticklabels([str(y) for y in yticks], **fontdict)
    else:
        mpl_axes.set_yticklabels([str(x) for x in labels], rotation = rotation, **fontdict)
    return None


               
def handle_colours(colour_type,
                   n_groups,
                   choice = "centered"):
    """
    =================================================================================================
    handle_colours(colout_type, n_groups, choice)
    
    This function is meant to get a list colours that will define the bars on the barplot. It uses
    the colours dictionary defined globally and the keys as the colour_type.
    
    =================================================================================================
    Arguments:
    
    colour_type  ->  A string that is in the keys of the colours dictionary. If a string not in
                     the colours dictionary is provided, the function will use either
                     'default' or 'all', depending on the size of n_groups.
    n_groups     ->  An integer representing the number of groups within a category.
    choice       ->  A string representing how to choose the colours from a list.
                     "centered" "random"
    
    =================================================================================================
    Returns: A list of colour strings of size n_groups.
    
    =================================================================================================
    """
    # Use the global colours dictionary for colour determination.
    global colours
    # First, check to see if the colour_type given is in the keys of the dictionary.
    if f"{colour_type}" in list(colours.keys()):
        # Next, check to see whether there are enough colours to support choice.
        #
        # If there are enough colours to support choices
        if n_groups <= len(colours[f'{colour_type}']):
            # Then check the choice string.
            #
            # If the choice string is random
            if choice == "random":
                # Then choose a random sample from the colours list, of size n_groups
                return random.sample(colours[f"{colour_type}"], n_groups)
            # If the choice string is centered
            elif choice == "centered":
                # Then get the center of the list. For even lists, this
                # will be the right-center.
                ind_cent = len(colours[f"{colour_type}"]) // 2
                # If the number of groups and the number of colours are equivalent
                if n_groups == len(colours[f"{colour_type}"]):
                    # Then return the list of colours
                    return colours[f"{colour_type}"]
                # If there is an even number of groups
                elif n_groups %2 == 0:
                    # Then get a slice object that is evenly spaced about
                    # the center of the list.
                    x = slice(ind_cent - n_groups//2, ind_cent + n_groups //2)
                    # Then attempt to return the colours list, sliced by x
                    try:
                        return colours[f'{colour_type}'][x]
                    # And if that fails, then return a random sample from the list
                    except:
                        return random.sample(colours[f'{colour_type}'], n_groups)
                # If there is an odd number of groups
                else:
                    # Then get a slice object that is left centered.
                    x = slice(ind_cent - 1 - n_groups //2, ind_cent - n_groups//2 + 2)
                    # and try to return the colours sliced at that point
                    try:
                        return colours[f'{colour_type}'][x]
                    # Otherwise return a random sample of the colours from that list.
                    except:
                        return random.sample(colours[f'{colour_type}'], n_groups)
            # If something other than centered or random is chosen
            else:
                # Then return a random sample from the colour
                return random.sample(colours[f"{colour_type}"])
        # If there are not enough colours in the chosen list,
        else:
            # Then return a random sample from all
            return random.sample(colours['all'], n_groups)
    # If the colour type is not in the keys and the number of groups is
    # less than or equal to the default list
    elif colour_type not in list(colours.keys()) and n_groups <= len(colours["default"]):
        # Then return a slice of the default list
        return colours["default"][:n_groups]
    # Otherwise, all things in the world are wrong
    else:
        # So just return a random sample from the all list of size n_groups.
        return random.sample(colours['all'], n_groups)

def get_range(a_list):
    """
    =================================================================================================
    get_range(a_list)
    
    This is meant to find the maximal span of a list of values.
    
    =================================================================================================
    Arguments:
    
    a_list  ->  A list of floats/ints.  [1,2,-3]
    
    =================================================================================================
    Returns: a tuple of the values that are either at the end/beginning of the list. (-3,3)

    =================================================================================================
    """
    # Make sure the input list is correctly formatted
    assert type(a_list) == list, "The input a_list should be of type list..."
    # First, unpack the list of lists. This makes one list with all values from
    # the lists within the input list.
    #print(a_list)
    unpacked = gh.unpack_list(a_list)
    #print(unpacked)
    # Next, float the items in the unpacked list. This will fail if any
    # strings that are not floatable are in the list.
    unpacked = [float(item) for item in unpacked if float(item) == float(item)]
    # Then we can get the max and min of the list.
    maxi = max(unpacked)
    mini = min(unpacked)
    # If the max value is greater than or equal to the minimum value
    if abs(maxi) >= abs(mini):
        # Then the bound is the int of max plus 1
        bound = int(abs(maxi)) + 1
        # We can then return the bounds, plus and minus bound
        return (-bound, bound)
    # If the min value is greater than the max value
    elif abs(maxi) < abs(mini):
        # Then the bound is the int of the absolute value
        # of mini plus 1
        bound = int(abs(mini)) + 1
        # We can then return the bounds, plus and minus bound
        return (-bound, bound)
    # If something goes wrong,
    else:
        # Then raise an error
        raise ValueError("This is an unexpected outcome...")

def add_errorbar(mpl_axes, x_pos, y_pos, 
                 std, color = "grey", x_offset = 0.05, 
                 transparency = 0.75):
    """
    Adds error bars to MatPlotLib axes object. Modifies the mpl_axes input, returns None.

    mpl_axes -> matplotlib axes object
    x_pos    -> position of the center of the error bars on the x axis
    y_pos    -> position of the center of the error bars on the y axis
    std      -> standard deviation of the data (or vertical offset for error bars)
    color    -> The color of the error bars (Default: "grey")
    x_offset -> Size of the middle of the error bars (Default: 0.05)
    """
    # Plot the vertical middle bar
    mpl_axes.plot([x_pos, x_pos], [y_pos + std, y_pos - std], color = color, alpha = transparency)
    # Plot the horizontal middle bar
    mpl_axes.plot([x_pos - x_offset, x_pos + x_offset], [y_pos, y_pos], color = color, alpha = transparency)
    # Plot the horizontal top bar
    mpl_axes.plot([x_pos - 0.75*x_offset, x_pos + 0.75*x_offset], [y_pos + std, y_pos + std], color = color, alpha = transparency)
    # Plot the horizontal bottom bar
    mpl_axes.plot([x_pos - 0.75*x_offset, x_pos + 0.75*x_offset], [y_pos - std, y_pos - std], color = color, alpha = transparency)
    return None

#
#
############################################################################################################
#
#     Functions: Bar Plots

from mph_modules.barcharts import bars

#
#
############################################################################################################
#
#      Functions: Heatmaps

from mph_modules.heatmaps import heatmap

#
#
############################################################################################################
#
#

# Make dotplots

from mph_modules.dotplots import dotplot

#
#
############################################################################################################
#
#         Scatterplot with linear regression

def scatter_gausskde(d1,d2, ax, scatterargs = {"s" : 20,
                                               "alpha" : 0.5,
                                               "marker" : "s"},
                    llsr = True,
                    update_ticks = True,
                    xlim = [0,25],
                    ylim = [0,25]):
    if llsr and xlim != None and ylim != None:
        data1, data2 = sh.remove_nanpairs(d1,d2)
        a, b, r2= sh.least_squares_fit(data1, data2)
        xs = [xlim[0], ylim[1]]
        # Finish by getting y values and plotting a line
        ys = [a+b*xs[0], a+b*xs[1]]
        ax.plot(xs, ys, color = "black", linestyle = ":",
                label = f"$y={a:.2f}+{b:.2f}x$\n$r^2 = {r2:.3f}$")
    else:
        data1, data2 = sh.remove_nanpairs(d1,d2)
        a, b, r2= sh.least_squares_fit(data1, data2)
        xs = [min(data1), max(data1)]
        # Finish by getting y values and plotting a line
        ys = [a+b*xs[0], a+b*xs[1]]
        ax.plot(xs, ys, color = "black", linestyle = ":",
                label = f"$y={a:.2f}+{b:.2f}x$\n$r^2 = {r2:.3f}$")
    xy=np.vstack(remove_nanpairs(d1,d2))
    z = gaussian_kde(xy)(xy)
    ax.scatter(*remove_nanpairs(d1,d2), c=z, label = f"$n={len(data1)}$",
               **scatterargs)
    ax.legend(loc="upper left")
    if xlim != None:
        ax.set_xlim(*xlim)
    if ylim != None:
        ax.set_ylim(*ylim)
    if update_ticks:
        ax.set_xticks(ax.get_xticks())
        xlabs = []
        for tick in ax.get_xticks():
            if int(tick) == float(tick):
                xlabs.append(str(int(tick)))
            else:
                xlabs.append(str(tick))
        ax.set_xticklabels(xlabs,
                           fontfamily ="sans-serif", font = "Arial",
                           fontweight = "bold")
        ax.set_yticks(ax.get_yticks())
        ylabs = []
        for tick in ax.get_yticks():
            if int(tick) == float(tick):
                ylabs.append(str(int(tick)))
            else:
                ylabs.append(str(tick))
        ax.set_yticklabels(ylabs,
                           fontfamily ="sans-serif", font = "Arial",
                           fontweight = "bold")
    if llsr:
        return ax, [a,b,r2]
    else:
        return ax, [float("nan"), float("nan"), float("nan")]


#
#
############################################################################################################
#
#         Exploiting matplotlib.axes.Axes.table()

from matplotlib.transforms import Bbox

def make_mpl_table(*lists,
                   fig = None,
                   ax = None, 
                   colours = [], 
                   columns = True,
                   colLabels = [],
                   rowLabels = [],
                   colLabels_colours = [],
                   rowLabels_colours = [],
                   tableargs = {"loc" : (0,0),
                                "edges" : "open"},
                   fontfamily = "sans-serif",
                   font = "Arial",
                   fontsize = "14",
                   fontweight = "bold"):
    """
    """
    if ax == None and fig == None:
        x = float(input("Please input the X component of figure size (float):\t" ))
        y = float(input("Please input the Y component of figure size (float):\t" ))
        fig, ax = plt.subplots(figsize = (x,y))
    elif ax == None or fig == None:
        assert False, "Both a figure and axes object should be passed in, or neither."
    if columns:
        lists = gh.transpose(*lists)
    if colLabels != []:
        assert len(colLabels) == len(lists[0]), "Not enough column labels provided"
    else:
        colLabels = ["" for _ in range(len(lists[0]))]
    if rowLabels != []:
        assert len(rowLabels) == len(lists), "Not enough row labels provided"
    else:
        rowLabels = ["" for _ in range(len(lists))]
    if colLabels_colours != []:
        assert len(colLabels_colours) == len(colLabels), "The number of colours provided for column labels\ndoes not match the number of labels"
    else:
        colLabel_colours = ["black" for _ in range(len(colLabels))]
    if rowLabels_colours != []:
        assert len(rowLabels_colours) == len(rowLabels), "The number of colours provided for row labels\ndoes not match the number of labels"
    else:
        rowLabel_colours = ["black" for _ in range(len(rowLabels))]
    if colours != [] and columns:
        colours = gh.transpose(*colours)
        assert all([len(row) == len(lists[0]) for row in colours]), "Too few colours provided for each column"
        assert len(colours) == len(lists), "Too few rows of colours provided"
    elif colours != [] and not columns:
        assert all([len(row) == len(lists[0]) for row in colours]), "Too few colours provided for each column"
        assert len(colours) == len(lists), "Too few rows of colours provided"
    else:
        colours = [["black" for _ in range(len(lists[0])+1)] for i in range(len(lists)+1)]
    tab = ax.table(lists, rowLabels = rowLabels, colLabels = colLabels, **tableargs)
    r = fig.canvas.get_renderer()
    cellwidths = [[0 for _ in range(len(lists[0])+1)] for i in range(len(lists)+1)]
    cellheights = [[0 for _ in range(len(lists[0])+1)] for i in range(len(lists)+1)]
    for cell in tab._cells:
        text = tab._cells[cell].get_text()
        text.set_fontfamily(fontfamily)
        text.set_font(font)
        text.set_fontweight(fontweight)
        text.set_fontsize(fontsize)
        text.set_ha("center")
        coords = text.get_window_extent(renderer=r)
        coords = Bbox(ax.transData.inverted().transform(coords))
        width = coords.width
        height = coords.height
        cellwidths[cell[0]][cell[1]+1] = width
        cellheights[cell[0]][cell[1]+1] = height
    cellwidths = [max(col) for col in gh.transpose(*cellwidths)]
    cellheights = [max(col) for col in cellheights]
    for cell in tab._cells:
        if cell[0] == 0:
            text = tab._cells[cell].get_text()
            text.set_color(colLabel_colours[cell[1]])
            tab._cells[cell].set_width(cellwidths[cell[1]+1])
            tab._cells[cell].set_height(cellheights[cell[0]])
        elif cell[1] == -1:
            text = tab._cells[cell].get_text()
            text.set_color(rowLabel_colours[cell[0]-1])
            tab._cells[cell].set_width(cellwidths[cell[1]+1])
            tab._cells[cell].set_height(cellheights[cell[0]])
        else:
            text = tab._cells[cell].get_text()
            text.set_color(colours[cell[0]-1][cell[1]])
            tab._cells[cell].set_width(cellwidths[cell[1]+1])
            tab._cells[cell].set_height(cellheights[cell[0]])
    return fig, ax, tab

#
#
############################################################################################################