"""


"""
###################################################################################################################
#
# Imports

# Pyplot has the majority of the matplotlib plotting functions
# and classes, including figures, axes, etc.
import matplotlib.pyplot as plt

#from .. import general_helpers as gh

try:
    from .. import general_helpers as gh
    from ..mpl_plotting_helpers import handle_colours
except:
    from pathlib import Path
    hpath = Path(__file__).parent.absolute()
    help_path = ""
    for folder in str(hpath).split("/")[1:-1]:
        help_path = f"{help_path}/{folder}"
    import sys
    sys.path.insert(0,help_path)
    import general_helpers as gh
    from mpl_plotting_helpers import handle_colours

#
#
###################################################################################################################
#
# Functions

def bars_ind(n_items,
             n_groups,
             sep_dist = 0.4,
             transpos = True):
    """
    =================================================================================================
    bars_ind(n_items, n_groups, sep_dist, transpos)
    
    This funciton is used to determine the center positions of all bars used for a bar plot, as
    well as the width of each bar.
    
    =================================================================================================
    Arguments:
    
    n_items   ->  An integer representing the number of categories to be plotted
    n_groups  ->  An integer representing the number of groups per category to be plotted
    sep_dist  ->  A float representing the distance between categories in the plot
    transpos  ->  A boolean determining whether to transpose the output list.
    
    =================================================================================================
    Returns: A list of lists determining the position of bars for each group, and the width
             of all bars in the plot.
    
    =================================================================================================
    """
    # Determine the width of a bar, by subtracting the separation distance from 1
    # and dividing by the number of groups.
    width = (1-sep_dist) / n_groups
    # If there is only one group,
    if n_groups == 1:
        # Then return a list with one sublist, centered
        # on the integers.
        return [[i for i in range(len(n_groups))]]
    # Initialize the output list. This will hold the lists returned.
    output = []
    # Next, loop over the number of categories
    for i in range(n_items):
        # Get the left most position of the interval
        interval = i + sep_dist/2 - 0.5
        # Add equidistant points from the interval to the output
        # Up to the number of groups per category and add that
        # to the outputs
        output.append([interval + j * width for j in range(1,n_groups+1)])
    # If the output must be transposed
    if transpos:
        # Then return the transposed output list and width
        return gh.transpose(*output), width
    # Otherwise,
    else:
        # Return the list itself and the width
        return output, width

def bars(xvals,
         yval_matrix,
         col_labels = None,
         separation = 0.3,
         colour_type = "all",
         colour_choice = "centered",
         img_type = "pdf",
         img_name = "pokedex_completion",
         show = True,
         subplot_args = {"figsize" : (24,12)},
         set_kwargs = {"xlabel" : "Ash Ketchum",
                       "ylabel" : "Pokemon Caught",
                       "title"  : "I wanna be, the very best, like NO ONE EVER WAS"}):
    """
    =================================================================================================
    bars(*args, **kwargs)
    
    This function is meant to wrap axes.bars() and provide some of the formatting options for
    barplots
    
    =================================================================================================
    Arguments:
    
    xvals          ->  A list of strings that determine the categories of the bar chart. These values
                       and the values in each sublist of yval_matrix should be index paired.
    yval_matrix    ->  A list of lists of values, where each sublist represents the values for a
                       a specific group.
    col_labels     ->  A list of labels for the bars. These values should be index paired with the
                       yval_matrix list.
    separation     ->  A float (0<separation<1) that determines how much space is between two
                       categories
    colour_type    ->  A string that represents the colour group to use for the bars
    colour_choice  ->  A string that represents how to choose the colours from the colour list.
    img_type       ->  A string representing the file extension for the image.
    img_name       ->  A string representing the name of the file.
    show           ->  A boolean that determines whether or not to show the plot.
    subplot_args   ->  A dictionary of keyword arguments to be passed into plt.subplots()
    set_kwargs     ->  A dictionary of keyword arguments to be passed into ax.set()
    
    =================================================================================================
    Returns: None, but a figure is saved.
    
    =================================================================================================
    """
    # Set the global font size parameter to 20.
    plt.rcParams["font.size"] = 20
    # Get the indices list (of lists) and the width of the bars
    indices, width = bars_ind(len(xvals),
                              len(yval_matrix),
                              sep_dist = separation)
    # Get the colour list that wil be used for barplots
    color = handle_colours(colour_type,
                           len(yval_matrix),
                           choice = colour_choice)
    # With the indices and colours list, we are set to start plotting.
    # Create a figure and axes object using plt.subplots().
    fig, ax = plt.subplots(**subplot_args)
    # Then, loop over the number of indices lists created by
    # bars_ind()
    for i in range(len(indices)):
        # If column labels are provided, then we assume they are
        # index paired with the lists in yval_matrix and thus
        # the indices list.
        if col_labels != None:
            # So plot a bar to the axes for this combination,
            # with a colour.
            ax.bar(indices[i], yval_matrix[i],
                   width = width, label = col_labels[i],
                   edgecolor = "black", color = color[i])
        # If no labels are provided,
        else:
            # Then plot a bar all the same, but without a label.
            ax.bar(indices[i], yval_matrix[i],
                   width = width, label = col_labels[i], color = color[i])
    # Once all of the bars are plotted, plot a legend.
    ax.legend()
    # If the number of indices lists is even
    if len(indices) % 2 == 0:
        # Then calculate the position of the labels based on an even assumption.
        ax.set_xticks(indices[len(indices)//2 + 1])
        ax.set_xticklabels(xvals,rotation = 45, ha = 'right', rotation_mode = "anchor")
    # If the number of indices lists is odd
    elif len(indices) % 2 == 1:
        # Then calculate the position of the labels based on an odd assumption.
        ax.set_xticks([item + width/2 for item in indices[len(indices)//2]])
        ax.set_xticklabels(xvals,rotation = 45, ha = 'right', rotation_mode = "anchor")
    # If set_kwargs is not en empty dictioanry
    if set_kwargs != {}:
        # Then run ax.set{} using that dictionary. This will fail if the
        # arguments are not arguments in the set() method.
        ax.set(**set_kwargs)
    # Save the figure using the image name and image type
    if show:
        plt.savefig(f"{img_name}.{img_type}", bbox_inches = "tight")
    # If show is True
    return ax

#
#
###################################################################################################################