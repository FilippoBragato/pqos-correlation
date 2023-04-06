import numpy as np
import seaborn as sns
import pandas as pd
import tikzplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = 64/9, 4
import matplotlib.pyplot as plt
plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


def multi_barplot(distribution_data: [np.ndarray],
                  distribution_labels: [str],
                  distribution_legends: [str],
                  value_key: str,
                  label_key: str,
                  legend_key: str,
                  output_file: str,
                  palette=None,
                  lim=None,
                  plot_sizes=None,
                  plot_format='eps'
                  ):

    if plot_sizes is not None:
        fig = plt.figure(figsize=plot_sizes)
    else:
        fig = plt.figure()

    ax = fig.add_subplot(111)

    data_values = []
    data_labels = []
    data_legends = []

    value_num = len(distribution_data[0])
    legend_num = len(list(set(distribution_legends)))

    for values, label, legend in zip(distribution_data, distribution_labels, distribution_legends):

        data_values.append(np.mean(values))

        data_labels += [label]

        if legend is not None:
            if legend_key is None:
                data_legends += [str(legend)]    
            else:
                data_legends += [legend_key + ' = '+ str(legend)]

    if legend_num > 1:

        data = pd.DataFrame({value_key: data_values, label_key: data_labels, legend_key: data_legends})
        sns.barplot(data=data, ax=ax, x=label_key, y=value_key, hue=legend_key, palette=palette)

    else:

        data = pd.DataFrame({value_key: data_values, label_key: data_labels})

        sns.barplot(data=data, ax=ax, x=label_key, y=value_key, palette=palette)

    ax.grid(b=True, color='darkgrey', linestyle='-')
    ax.grid(b=True, which='minor', color='darkgrey', linestyle='-')

    if legend_num > 1:

        ax.legend(handles=ax.legend_.legendHandles,
                ncol=legend_num,
                bbox_to_anchor=(0.5, 1.15),
                loc='upper center')

    if lim is not None:
        ax.set_ylim([lim[0], lim[1]])

    plt.savefig(output_file.replace(" ", "_") + '.png', format='png', bbox_inches='tight')

    if plot_format == 'tex':
        tikzplotlib.save(output_file.replace(" ", "_") + '.tex')

    elif plot_format != 'png':
        plt.savefig(output_file.replace(" ", "_") + '.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.close()