import loader as loader
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import dmbp
import utils
import os

from dmbp.dmbp import results_to_numpy
# from metric_funcs import *

if __name__=='__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Parse opts
    parser = argparse.ArgumentParser(description='Opts Parser')

    opts = loader.parse_opts(parser)

    # Load Model
    model_dict = loader.init_model(opts.model)
    model = model_dict['model']
    model.to(device)
    model.eval()

    classes_dict = model_dict['classes']
    prob_layer = model_dict['prob_layer']
    im_transform = model_dict['im_transform']

    # Attribution map ops
    im_path = opts.image_path
    target_label = opts.target_label
    src_label = opts.source_label
    src_im_path = opts.source_path

    # Load image and compute probability for target class
    input_image = Image.open(im_path)
    source_image = Image.open(src_im_path)

    x = im_transform(input_image)
    src = im_transform(source_image)

    x = x.unsqueeze(0).to(device)
    src = src.unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)
    with torch.no_grad():
        y_src = model(src)

    # Get target label
    if(target_label==-1):
        _,target_label = torch.max(y , dim=1)
        target_label = target_label.item()
    target_class = classes_dict[target_label]

    if(src_label==-1):
        _,src_label = torch.max(y_src , dim=1)
        src_label = src_label.item()
    src_target_class = classes_dict[src_label]
    # Compute attribution map
    print('Computing Attribution Map with DMBP...')

    # Initialize DMBP with base model
    dmbp1 = dmbp.DMBP(model)
    dmbp2 = dmbp.DMBP(model)

    # Compute attribution results
    attribution_results = dmbp1.attributions(x,target_label,num_its = 200)
    src_attribution_results = dmbp2.attributions(src, src_label, num_its=200)
    results = {}
    results['x'] = x
    results['src'] = src
    results['x_pos_att'] = attribution_results['pos_attribution']
    results['x_neg_att'] = attribution_results['neg_attribution']
    results['src_pos_att'] = src_attribution_results['pos_attribution']
    results['src_neg_att'] = src_attribution_results['neg_attribution']
    results = results_to_numpy(results)

    #Visualize results
    image_path = os.path.basename(opts.image_path)
    image_path = os.path.splitext(image_path)[0]
    results_filename = 'results/des/' + image_path + '_' + opts.model + '_class_' + target_class + '_dmbp.png'
    utils.visualize_exp(results,target_class=target_class,src_class=src_target_class,
                                 imfile=results_filename)
