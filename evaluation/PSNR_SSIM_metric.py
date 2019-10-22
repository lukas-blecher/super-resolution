'''
Original File taken from https://github.com/xinntao/BasicSR
'''
import argparse
import os
import math
import numpy as np
import cv2
import glob


def get_metrics(ground_truth, generated, suffix='', image_type='png', test_Y=False, MAX=255):

    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = ground_truth
    folder_Gen = generated

    crop_border = 4
    suffix = suffix  # suffix for Gen images
    test_Y = test_Y  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_GT = cv2.imread(img_path) / MAX

        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.' + image_type))
        if im_Gen is None:
            continue

        im_Gen = im_Gen / MAX

        # evaluate on Y channel in YCbCr color space
        if test_Y and im_GT.shape[2] == 3:
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError(
                'Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * MAX, cropped_Gen * MAX, MAX=MAX)

        SSIM = calculate_ssim(cropped_GT * MAX, cropped_Gen * MAX, dynamic_range=MAX)
        #print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(i + 1, base_name, PSNR, SSIM))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))
    return np.mean(PSNR_all), np.std(PSNR_all), np.mean(SSIM_all), np.std(SSIM_all)


def calculate_psnr(img1, img2, MAX=255):
    # img1 and img2 have range [0, MAX]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(MAX / math.sqrt(mse))


def ssim(img1, img2, dynamic_range=255):

    C1 = (0.01 * dynamic_range)**2
    C2 = (0.03 * dynamic_range)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, dynamic_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, dynamic_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", type=str, required=True, help="Folder containing ground truth images")
    parser.add_argument("--generated", type=str, required=True, help="Folder containing generated images")
    parser.add_argument("--suffix", type=str, default="", help="suffix of generated images")
    parser.add_argument("--image_type", type=str, default="png", help="Image type (e.g. jpg, png, gif, etc.)")
    parser.add_argument("--test_Y", type=bool, default=False, help="Test only the Y channel")
    args = parser.parse_args()
    print(args)
    get_metrics(args.ground_truth, args.generated, args.suffix, args.image_type, args.test_Y)
