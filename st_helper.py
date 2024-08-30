import sys
from glob import glob
import shutil

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from imageio import imread, imwrite

import utils
from vgg_pt import Vgg16_pt
from pyr_lap import dec_lap_pyr, syn_lap_pyr
from stylize_objectives import objective_class

def style_transfer(stylized_im, content_im, style_path, output_path, scl, long_side, mask, 
                   content_weight=0., use_guidance=False, regions=0, coords=0, lr=2e-3):
    """
    Perform style transfer on the given content image using the specified style.

    Args:
        stylized_im (torch.Tensor): Initial stylized image.
        content_im (torch.Tensor): Content image.
        style_path (str): Path to the style image or folder.
        output_path (str): Path to save the output image.
        scl (float): Scale factor for the output image.
        long_side (int): Length of the long side of the image.
        mask (torch.Tensor): Mask for region-based style transfer.
        content_weight (float, optional): Weight for content loss. Defaults to 0.
        use_guidance (bool, optional): Whether to use guided style transfer. Defaults to False.
        regions (tuple, optional): Regions for region-based style transfer. Defaults to 0.
        coords (torch.Tensor, optional): Coordinates for guided style transfer. Defaults to 0.
        lr (float, optional): Learning rate for optimization. Defaults to 2e-3.

    Returns:
        tuple: A tuple containing:
            - stylized_im (torch.Tensor): The final stylized image.
            - ell (float): The final loss value.
    """
    REPORT_INTERVAL = 100
    RESAMPLE_FREQ = 1
    MAX_ITER = 250

    use_pyr = True
    temp_name = f'./{output_path.split("/")[-1].split(".")[0]}_temp.png'

    # Keep track of current output image for GUI
    canvas = aug_canvas(stylized_im, scl, 0)
    imwrite(temp_name, canvas)
    shutil.move(temp_name, output_path)

    # Define feature extractor
    cnn = utils.to_device(Vgg16_pt())
    phi = lambda x: cnn.forward(x)
    phi2 = lambda x, y, z: cnn.forward_cat(x, z, samps=y, forward_func=cnn.forward)

    # Optimize over Laplacian pyramid instead of pixels directly
    if use_pyr:
        s_pyr = dec_lap_pyr(stylized_im, 5)
        s_pyr = [Variable(li.data, requires_grad=True) for li in s_pyr]
    else:
        s_pyr = [Variable(stylized_im.data, requires_grad=True)]

    optimizer = optim.RMSprop(s_pyr, lr=lr)

    # Pre-extract content features
    z_c = phi(content_im)

    # Pre-extract style features from a folder
    paths = glob(f'{style_path}*')[::3]

    # Create objective object
    objective_wrapper = objective_class(objective='remd_dp_g')

    z_s_all = []
    for ri in range(len(regions[1])):
        z_s, _ = load_style_folder(phi2, paths, regions, ri, n_samps=-1, subsamps=1000, scale=long_side, inner=5)
        z_s_all.append(z_s)

    # Extract guidance features if required
    gs = torch.tensor([0.])
    if use_guidance:
        gs = load_style_guidance(phi, style_path, coords[:, 2:], scale=long_side)

    # Initialize regions
    stylized_im = syn_lap_pyr(s_pyr) if use_pyr else s_pyr[0]
    for ri, r_temp in enumerate(regions[0]):
        r_temp = F.interpolate(r_temp.unsqueeze(0).unsqueeze(0), 
                               (stylized_im.size(3), stylized_im.size(2)), 
                               mode='bilinear')[0, 0].numpy()
        r = r_temp > 0.5 if r_temp.max() >= 0.1 else r_temp + 1 > 0.5
        objective_wrapper.init_inds(z_c, z_s_all, r, ri)

    if use_guidance:
        objective_wrapper.init_g_inds(coords, stylized_im)

    for i in range(MAX_ITER):
        optimizer.zero_grad()
        stylized_im = syn_lap_pyr(s_pyr) if use_pyr else s_pyr[0]

        # Resample spatial locations
        if i == 0 or i % (RESAMPLE_FREQ * 10) == 0:
            for ri, r_temp in enumerate(regions[0]):
                r_temp = F.interpolate(r_temp.unsqueeze(0).unsqueeze(0), 
                                       (stylized_im.size(3), stylized_im.size(2)), 
                                       mode='bilinear')[0, 0].numpy()
                r = r_temp > 0.5 if r_temp.max() >= 0.1 else r_temp + 1 > 0.5
                objective_wrapper.init_inds(z_c, z_s_all, r, ri)

        if i == 0 or i % RESAMPLE_FREQ == 0:
            objective_wrapper.shuffle_feature_inds()

        # Extract features from current output
        z_x = phi(stylized_im)

        # Compute objective and take gradient step
        ell = objective_wrapper.eval(z_x, z_c, z_s_all, gs, 0., content_weight=content_weight, moment_weight=1.0)
        ell.backward()
        optimizer.step()

        # Periodically save output image for GUI
        if (i + 1) % 10 == 0:
            canvas = aug_canvas(stylized_im, scl, i)
            imwrite(temp_name, canvas)
            shutil.move(temp_name, output_path)

        # Periodically report loss
        if (i + 1) % REPORT_INTERVAL == 0:
            print(f'\t{i + 1} {ell}')

    return stylized_im, ell

# Additional helper functions (aug_canvas, load_style_folder, load_style_guidance) should be implemented here
