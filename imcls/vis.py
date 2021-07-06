import argparse
import torch
import os.path as osp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def normalize(feature):
    norm = np.sqrt((feature**2).sum(1, keepdims=True))
    return feature / (norm + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, default='', help='path to source file')
    parser.add_argument('--dst', type=str, default='', help='destination directory')
    parser.add_argument('--method', type=str, default='tsne', help='tnse, pca or none')
    args = parser.parse_args()

    if not args.dst:
        args.dst = osp.dirname(args.src)

    print('Loading file from "{}"'.format(args.src))
    file = torch.load(args.src)

    embed = file['embed']
    domain = file['domain']
    dnames = file['dnames']

    #dim = embed.shape[1] // 2
    #embed = embed[:, dim:]

    #domain = file['label']
    #dnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    nd_src = len(dnames)
    embed = normalize(embed)
    print('Loaded features with shape {}'.format(embed.shape))

    embed2d_path = osp.join(args.dst, 'embed2d_' + args.method + '.pt')

    if osp.exists(embed2d_path):
        embed2d = torch.load(embed2d_path)
        print('Loaded embed2d from "{}"'.format(embed2d_path))

    else:
        if args.method == 'tsne':
            print('Dimension reduction with t-SNE (dim=2) ...')
            tsne = TSNE(
                n_components=2, metric='euclidean', verbose=1,
                perplexity=50, n_iter=1000, learning_rate=200.
            )
            embed2d = tsne.fit_transform(embed)

            torch.save(embed2d, embed2d_path)
            print('Saved embed2d to "{}"'.format(embed2d_path))

        elif args.method == 'pca':
            print('Dimension reduction with PCA (dim=2) ...')
            pca = PCA(n_components=2)
            embed2d = pca.fit_transform(embed)

            torch.save(embed2d, embed2d_path)
            print('Saved embed2d to "{}"'.format(embed2d_path))

        elif args.method == 'none':
            # the original embedding is 2-D
            embed2d = embed

    avai_domains = list(set(domain.tolist()))
    avai_domains.sort()

    print('Plotting ...')

    SIZE = 3
    COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    LEGEND_MS = 3

    fig, ax = plt.subplots()

    for d in avai_domains:
        d = int(d)
        e = embed2d[domain == d]

        """
        label = '$D_{}$'.format(str(d + 1))
        if d < nd_src:
            label += ' ($\mathcal{S}$)'
        else:
            label += ' ($\mathcal{N}$)'
        """
        label = dnames[d]

        ax.scatter(
            e[:, 0],
            e[:, 1],
            s=SIZE,
            c=COLORS[d],
            edgecolors='none',
            label=label,
            alpha=1,
            rasterized=False
        )

    #ax.legend(loc='upper left', fontsize=10, markerscale=LEGEND_MS)
    ax.legend(fontsize=10, markerscale=LEGEND_MS)
    ax.set_xticks([])
    ax.set_yticks([])
    #LIM = 22
    #ax.set_xlim(-LIM, LIM)
    #ax.set_ylim(-LIM, LIM)

    figname = 'embed.pdf'
    fig.savefig(osp.join(args.dst, figname), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
