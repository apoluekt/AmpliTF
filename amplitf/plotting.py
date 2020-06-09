import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

data_color = "black"
fit_color  = "C0"
sig_color  = "C1"
bck_color  = "C2"
diff_color = "C3"

def set_lhcb_style(grid = True, size = 10, usetex = "auto", font = "serif") : 
  """
    Set matplotlib plotting style close to "official" LHCb style
    (serif fonts, tick sizes and location, etc.)
  """
  if usetex == "auto" : 
    plt.rc('text', usetex=os.path.isfile("/usr/bin/latex"))
  else : 
    plt.rc('text', usetex=usetex)
  plt.rc('font', family=font, size=size)
  plt.rcParams['axes.linewidth']=1.3
  plt.rcParams['axes.grid']=grid
  plt.rcParams['grid.alpha']=0.3
  plt.rcParams["axes.axisbelow"] = False
  plt.rcParams['xtick.major.width']=1
  plt.rcParams['ytick.major.width']=1
  plt.rcParams['xtick.minor.width']=1
  plt.rcParams['ytick.minor.width']=1
  plt.rcParams['xtick.major.size']=6
  plt.rcParams['ytick.major.size']=6
  plt.rcParams['xtick.minor.size']=3
  plt.rcParams['ytick.minor.size']=3
  plt.rcParams['xtick.direction']="in"
  plt.rcParams['ytick.direction']="in"
  plt.rcParams['xtick.minor.visible']=True
  plt.rcParams['ytick.minor.visible']=True
  plt.rcParams['xtick.bottom']=True
  plt.rcParams['xtick.top']=True
  plt.rcParams['ytick.left']=True
  plt.rcParams['ytick.right']=True

def label_title(title, units = None) : 
  label = title
  if units : title += " (" + units + ")"
  return title

def plot_distr1d_comparison(data, fit, bins, range, ax, label, log = False, units = None, weights = None, pull = False, 
                            legend = None, color = None) : 
  """
    Plot 1D histogram and its fit result. 
      hist : histogram to be plotted
      func : fitting function in the same format as fitting.fit_hist1d
      pars : list of fitted parameter values (output of fitting.fit_hist2d)
      ax   : matplotlib axis object
      label : x axis label title
      units : Units for x axis
  """
  dlab, flab = "Data", "Fit"
  fitscale = np.sum(data)/np.sum(fit)
  datahist, _ = np.histogram(data, bins = bins, range = range)
  fithist, edges = np.histogram(fit, bins = bins, range = range)
  left,right = edges[:-1],edges[1:]
  xarr = np.array([left,right]).T.flatten()
  fitarr = np.array([fithist,fithist]).T.flatten()*fitscale
  dataarr = np.array([datahist,datahist]).T.flatten()
  ax.plot(xarr, fitarr, label = flab, color = fit_color)

  if isinstance(weights, list) : 
    cxarr = None
    for i,w in enumerate(weights) : 
      chist, cedges = np.histogram(fit, bins = bins, range = range, weights = w)
      if cxarr is None : 
        cleft,cright = cedges[:-1],cedges[1:]
        cxarr = (cleft+cright)/2.
      fitarr = chist*fitscale
      if color : this_color = color[i]
      else : this_color = f"C{i+1}"
      if legend : lab = legend[i]
      else : lab = None
      ax.plot(cxarr, fitarr, color = this_color, label = lab)
      ax.fill_between( cxarr, fitarr, 0., color = this_color, alpha = 0.1)

  xarr = (left+right)/2.
  ax.errorbar(xarr, datahist, np.sqrt(datahist), label = dlab, color = data_color, marker = ".", linestyle = '')

  ax.legend(loc = "best")
  ax.set_ylim(bottom = 0.)
  ax.set_xlabel(label_title(label, units), ha='right', x=1.0)
  ax.set_ylabel(r"Entries", ha='right', y=1.0)
  ax.set_title(label + r" distribution")
  if pull : 
    pullhist = (datahist-fithist)/np.sqrt(datahist)
    pullarr = np.array([pullhist,pullhist]).T.flatten()
    ax2 = ax.twinx()
    ax2.set_ylim(bottom = -10.)
    ax2.set_ylim(top =  10.)
    ax2.plot(xarr, pullarr, color = diff_color, alpha = 0.3)
    ax2.grid(False)
    ax2.set_ylabel(r"Pull", ha='right', y=1.0)
    return [ ax2 ]
  return []

def plot_distr2d(xarr, yarr, bins, ranges, fig, ax, labels, cmap = "YlOrBr", 
                 log = False, ztitle = None, title = None, units = (None, None), 
                 weights = None, colorbar = True) : 
  """
    Plot 2D distribution including colorbox.
      hist   : histogram to be plotted
      fig    : matplotlib figure object
      ax     : matplotlib axis object
      labels : Axis label titles (2-element list)
      cmap   : matplotlib colormap
      log    : if True, use log z scale
      ztitle : x axis title (default is "Entries")
      title : plot title
      units : 2-element list for x axis and y axis units
  """
  #print(xarr.shape, yarr.shape, bins)
  #print("hist2d start")
  counts, xedges, yedges = np.histogram2d(xarr, yarr, bins = bins, range = ranges, weights = weights)
  #print("hist2d end")
  norm = None
  if log : 
    vmax = np.max(counts)
    vmin = np.min(counts)
    if vmin <= 0. : vmin = 1.
    if vmax <= vmin : vmax = vmin
    norm = matplotlib.colors.LogNorm(vmin = vmin, vmax = vmax)

  arr = counts.T

  X, Y = np.meshgrid(xedges, yedges)
  p = ax.pcolormesh(X, Y, arr, cmap = cmap, norm = norm, linewidth=0, rasterized=True)
  ax.set_xlabel(label_title(labels[0], units[0]), ha='right', x=1.0)
  ax.set_ylabel(label_title(labels[1], units[1]), ha='right', y=1.0)
  if title : ax.set_title(title)
  zt = ztitle
  if not ztitle : zt = r"Entries"
  if colorbar : 
    cb = fig.colorbar(p, pad = 0.01, ax = ax)
    cb.ax.set_ylabel(zt, ha='right', y=1.0)
    if log : 
      cb.ax.set_yscale("log")
