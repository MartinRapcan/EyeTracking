import os
import argparse
import csv
import numpy
from matplotlib import pyplot, image

def display(imagefile=None):
    if imagefile != None:
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        img = image.imread(imagefile)

    dpi = 100.0
    width = len(img[0])
    height = len(img)

    figsize = (width / dpi, height / dpi)
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)

    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.axis([0, width, 0, height])
    ax.imshow(img)

    return fig, ax

def gaussian(x, sx):
    xo = x / 2
    yo = x / 2

    M = numpy.zeros([x, x], dtype=float)

    for i in range(x):
        for j in range(x):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sx * sx))))

    return M

def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    fig, ax = display(imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh // 2
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig

input_path = "./coordinates/test.csv"

imageName = "./images/scan_test.jpg"
img = image.imread(imageName)
w, h = len(img[0]), len(img)
alpha = 0.5
output_name = "./images/heatmap_test_after"
ngaussian = 200
sd = 16

with open(input_path) as f:
	reader = csv.reader(f)
	raw = list(reader)
	
gaza_data = [] 
filtered = list(filter(lambda q: (int(q[0]) < w and int(q[1]) < h), raw))
gaze_data = list(map(lambda q: (int(q[0]), int(q[1])), filtered))
draw_heatmap(gaze_data, (w, h), alpha=alpha, savefilename=output_name, imagefile=imageName, gaussianwh=ngaussian, gaussiansd=sd)

def heatmap(imageName):
    input_path = f"./coordinates/{imageName}.csv"
    output_name = f"./images/{imageName}_heatmap"
    ngaussian = 200
    sd = 16

    with open(input_path) as f:
        reader = csv.reader(f)
        raw = list(reader)

    points = []
    