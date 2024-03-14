import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Normalization function for centering to 0 the attribution map visualization
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# Clip highest or lowest values in the top percentile
def clip(mapping,percentile=99.95):
    or_shape = mapping.shape
    mapping_max = np.percentile(np.abs(mapping.flatten()),percentile)
    mapping = mapping.clip(-mapping_max,mapping_max)
    mapping = mapping.reshape(or_shape)
    return mapping

# Normalize tensor to values between 0 and 1
def normalize_tensor(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max()+1e-8)
    return tensor

# Plot tensor as image
def show_tensor(tensor, title='',ax=None,cmap=None , normalize=True):
    if(normalize):
        tensor = normalize_tensor(tensor)
    if(cmap is None):
        cmap =  'viridis'
    if(ax is None):
        plt.imshow(tensor,cmap=cmap)
        plt.axis('off')
        plt.title(title, fontsize=9)
    else:
        ax.imshow(tensor,cmap=cmap)
        ax.axis('off')
        ax.set_title(title, fontsize=9)

# Normalize positive and negative attributions in the range between -1,1 and
# return the attribution map
def normalize_attributions(pos_attribution,neg_attribution):
    # Clip the lowest or largest values for better visualization
    pos_attribution = clip(pos_attribution)
    neg_attribution = clip(neg_attribution)
    # Set min value to 0
    pos_attribution = pos_attribution-pos_attribution.min()
    neg_attribution = neg_attribution-neg_attribution.min()

    # Get highest or lowest value
    max_pos_neg_attribution = max(np.abs(pos_attribution).max(),
                               np.abs(neg_attribution).max())
    pos_attribution = pos_attribution/max_pos_neg_attribution
    neg_attribution = neg_attribution/max_pos_neg_attribution

    return pos_attribution-neg_attribution

def show_attribution(image,attributions, ax, title=''):
    ax.imshow(normalize_tensor(image))
    ax.imshow(attributions,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.7)
    ax.axis('off')
    ax.set_title(title,fontsize=9)

# Visualize DMBP results (attribution maps and linear mappings)
def visualize_exp(results, target_class='', src_class = '',
                           imfile=None):
    num_models = len(results.keys())
    figure, axes = plt.subplots(1,4)

    ## Show Original Image
    image = results['x']
    src = results['src']
    ax = axes[0]
    show_tensor(image,ax=ax,title='Class: ' + target_class)
    # show_tensor(image,ax=ax,title='Adv. Example')

    ## Show Attribution Map
    # Combine positive and negative attributions
    x_pos_attribution = results['x_pos_att'].mean(axis=2)
    x_neg_attribution = -results['x_neg_att'].mean(axis=2)
    x_attributions = normalize_attributions(x_pos_attribution,x_neg_attribution)

    src_pos_attribution = results['src_pos_att'].mean(axis=2)
    src_neg_attribution = -results['src_neg_att'].mean(axis=2)
    src_attributions = normalize_attributions(src_pos_attribution,src_neg_attribution)

    # Show
    ax = axes[1]
    show_attribution(image, x_attributions, ax=ax, title='Attribution Map')

    ax = axes[2]
    show_tensor(src, ax=ax, title='Class: ' + src_class)

    ax = axes[3]
    show_attribution(src, src_attributions, ax=ax, title='Attribution Map')

    # Show figure or save into a file
    if(imfile is None):
        plt.show()
    else:
        plt.subplots_adjust(hspace = 0.01, wspace=0.01,left=0, bottom=0, right=1, top=1)
        #plt.tight_layout()
        plt.savefig(imfile,dpi=600,bbox_inches='tight')

