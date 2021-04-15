import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
import scipy
import os
colors = sns.color_palette("hls", n_colors=11)

"""
label example: bs_1024-input_65536-output_8192
"""

def filter_data(rule, data):
    keys = []
    new_data = {}
    for k,v in iter(data.items()):
        if k == 'err_jobs' or k == 'states':
            continue
        new_data[k] = []
        keys.append(k)
    for i in range(len(data['labels'])):
        f = 0
        l = data['labels'][i]
        for j in l.split('-'):
            if j.split('_')[0] in rule and \
                not int(j.split('_')[1]) in rule[ j.split('_')[0]]:
                f = 1
                break
        if f == 0:
            for k in keys:
                new_data[k].append(data[k][i])
    return new_data

'''
Return a list of hyperparameters that are swept from a label.
'''
def get_keyword(l):
    keywords = []
    for j in l.split('-'):
        p = j.split('_')
        if len(p) > 1 and p[1].isnumeric():
            keywords.append(p[0])
    return keywords

def get_n_from_label(l, dim_name):
    if dim_name == 'op':
        return l.split('_')[-1].strip('0123456789-')
    for i in l.split('-'):
        if dim_name in i:
            return i.split('_')[-1]
    return None

'''
Get the range of hyperparameters from a list of labels.
'''


def get_range(labels, verbose=True):
    keywords = []
    for i in labels[0].split('-'):
        if len(i.split('_')) > 1:
            keywords.append(i.split('_')[0])

    values = []
    for i in range(len(keywords)):
        values.append(set())

    for f in labels:
        f_split = f.split('.')[0].split('-')
        idx = 0
        for i in range(len(f_split)):
            if len(f_split[i].split('_')) > 1 and f_split[i].split('_')[1].isnumeric():
                values[idx].add(int(f_split[i].split('_')[1]))
                idx += 1
    results = {}
    for i in range(len(keywords)):
        if verbose == True:
            print(keywords[i], sorted(list(values[i])))
        results[keywords[i]] = sorted(list(values[i]))
    if verbose == True:
        print('-----------------------')
    return results


def speedup_params(a_labels, a_perf, b_labels, b_perf, b_params, legend_box=(), loc='', marker='', lim=[], color='', ncol=1, title=''):
    markers = ['.', '^', '*', '+', 'x', 'd', '>']
    speedups = []
    p = []
    l = []
    for i in range(len(a_labels)):
        label = a_labels[i]
        if not label in b_labels:
            continue
        ind = b_labels.index(label)
        if a_perf[i] == 0.0 or b_perf[ind] == 0.0:
            continue
        speedups.append(a_perf[i] / b_perf[ind])
        p.append(b_params[ind])
        l.append(label)
    print('length of speedups', len(speedups))
    print(f"max speedup: {max(speedups)} label: {l[np.argmax(speedups)]}")
    print(f"min speedup: {min(speedups)} label: {l[np.argmin(speedups)]}")
    fig, ax = plt.subplots(figsize=(3, 3))
    if color == '':
        ax.plot(p, speedups, '.')
    else:
        f = {}
        k = get_keyword(l[0])
        k_ind = k.index(color)
        r = get_range(l, verbose=True)
        num_color = len(r[color])
        colors = sns.color_palette("hls", n_colors=num_color)
        for i in range(len(l)):
            n = int(l[i].split('-')[k_ind].split('_')[-1])
            if n in f:
                if marker == '':
                    ax.plot(p[i], speedups[i], marker=markers[f[n]],
                            color=colors[f[n]])
                else:
                    ax.plot(p[i], speedups[i], marker=marker,
                            color=colors[f[n]])
            else:
                f[n] = r[color].index(n)
                if marker == '':
                    ax.plot(p[i], speedups[i], marker=markers[f[n]],
                            color=colors[f[n]], label=str(n))
                else:
                    ax.plot(p[i], speedups[i], marker=marker,
                            color=colors[f[n]], label=str(n))
        handles, labels = ax.get_legend_handles_labels()
        labels = [int(i) for i in labels]
        labels, handles = zip(
            *sorted(zip(labels, handles), key=lambda t: t[0]))
        if color == 'batchsize':
            color = 'bs'
        if color == 'embeddingsize':
            color = 'embed'
        if color == 'block':
            color = 'blk'
        if color == 'filtersz':
            color = 'filter'
        labels = [color + '-' + str(i) for i in labels]
        new_labels = []
        for l in labels:
            if '1024' in l:
                new_labels.append(l.replace('1024', '1k'))
            elif '2048' in l:
                new_labels.append(l.replace('2048', '2k'))
            elif '4096' in l:
                new_labels.append(l.replace('4096', '4k'))
            elif '8192' in l:
                new_labels.append(l.replace('8192', '8k'))
            elif '16384' in l:
                new_labels.append(l.replace('16384', '16k'))
            else:
                new_labels.append(l)

        if loc != '':
            ax.legend(handles, new_labels, loc=loc,
                      frameon=True, ncol=ncol, fontsize=12)
        else:
            ax.legend(handles, new_labels, frameon=True,
                      ncol=ncol, fontsize=12)
        if legend_box != ():
            ax.legend(handles, new_labels, frameon=True, ncol=ncol,
                      fontsize=12, bbox_to_anchor=legend_box)
    plt.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    left, right = ax.get_xlim()
    ax.set_xlim([left, right])
    ax.plot([left, right], [1, 1], 'k-', color='orange')
    if lim != []:
        ax.set_ylim(lim)
    ax.set_xlabel('Params', fontsize=20)
    ax.set_ylabel('Speedups', fontsize=20)
    ax.set_title(title, fontsize=18)
    return fig


def plot_roofline(f, ax, d, flops_peak, membdw_peak, \
                  scale='absolute', color_map={}, color_dim='', color=0, thre=1, marker='.', label='', title=''):
    markersize = 3
    colormap = {}
    flops = np.array(d['flops'])
    # filter data that is zero
    indices = np.where(flops > 0.0)
    flops = flops[indices]

    # convert into GFLOPS/s
    flops = np.multiply(flops, 1e-9)
    
    labels = np.array(d['labels'])[indices]
    intensity = np.array(d['arithemetic_intensity'])[indices]
    if color_dim == '':
        if color == 0:
          ax.plot(intensity, flops, marker, label=label)
        else:
          ax.plot(intensity, flops, marker, label=label, color=color, alpha=0.9)
    else:
        hist = {}
        for i in range(len(labels)):
            l = labels[i]
            n = get_n_from_label(l, color_dim)       
            if intensity[i]<=0 or flops[i]<=0:
                continue
            if not n in hist:
                hist[n] = 0    
            hist[n] += 1
        for k, v in iter(hist.items()):
            hist[k] = v*1.0/len(labels)
        
        m = {}
        mycolors = sns.color_palette("hls", n_colors=len(hist)+2)
        for i in range(len(labels)):
            # if time[i] < thre:
            #   continue
            if intensity[i]<=0 or flops[i]<=0:
              continue
            l = labels[i]
            n = get_n_from_label(l, color_dim)
                    
            if color_map != {}:
                if n in color_map:
                    if n in m:
                      ax.plot(intensity[i], flops[i], marker, color=color_map[n], marker=marker, ms=markersize)
                    else:
                      ax.plot(intensity[i], flops[i], marker, color=color_map[n], marker=marker, ms=markersize, label = n)
                      m[n] = 1
                continue    
            # second time
            if n in m:
                ax.plot(intensity[i], flops[i], marker,
                        color=mycolors[m[n]], marker=marker, ms=markersize)
            elif not n in m:
                
                m[n] = len(m) % len(colors)
                colormap[n] = mycolors[m[n]]
                ax.plot(intensity[i], flops[i], marker,
                        color=mycolors[m[n]], label = n, 
                        #markeredgecolor='black', markeredgewidth=0.5, 
                        marker=marker, ms=markersize)

        ax.legend(frameon=True)

    x1 = flops_peak / membdw_peak
    y1 = flops_peak
      
    # the flops peak
    if max(intensity) > x1:
        if color == 0:
            ax.hlines(y=y1, xmin=x1, xmax=max(intensity), linewidth=2, color=colors[0])
        else:
            ax.hlines(y=y1, xmin=x1, xmax=max(intensity), linewidth=2, color=color)
    
    #x2 = min(d['flops_perc'])*(flops_peak/100)/membdw_peak
    #y2 = min(d['flops_perc'])*(flops_peak/100)
    x2 = min(flops)/membdw_peak
    y2 = min(flops)
    # x2 = 10e3/membdw_peak
    # y2 = 10e3
    if scale == 'relative':
        y1 = 100
        y2 = x2 * membdw_peak / flops_peak * 100
    if color == 0:
        ax.plot([x1, x2], [y1, y2], linewidth=2, color=colors[0])
    else:
        ax.plot([x1, x2], [y1, y2], linewidth=2, color=color)
        
    ax.set_yscale('log')
    ax.set_xscale('log')
    if scale == 'absolute':
        ax.set_ylabel('GFLOP/S', fontsize=15)
    else:
        ax.set_ylabel('FLOPS %', fontsize=15)
        ax.set_ylim()
    ax.set_xlabel('FLOP/Byte', fontsize=15)
    ax.set_title(title, fontsize=15)

    if colormap == {}:
        colormap = color_map
    return f, ax, colormap