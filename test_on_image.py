from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
import glob
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import ImageDataset


def test_image(opt):  

    os.makedirs(opt.output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and load model checkpoint
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    generator.load_state_dict(torch.load(opt.checkpoint_model))
    generator.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    def sr_image(im_path):
        # Prepare input
        image = Image.open(im_path)
        if opt.downsample:
            hr_height, hr_width = image.size
            image = transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC)(image)

        image_tensor = Variable(transform(image)).to(device).unsqueeze(0)

        # Upsample image
        with torch.no_grad():
            sr_image = denormalize(generator(image_tensor)).cpu()

        # Save image
        fn = os.path.basename(im_path)
        save_image(sr_image, os.path.join(opt.output_path, fn))

    # check if the input is a image file or a folder
    if not os.path.isdir(opt.image_path):
        sr_image(opt.image_path)
    else:
        files = sorted(glob.glob(opt.image_path + "/*.*"))
        if opt.downsample:
            dataloader = DataLoader(
                ImageDataset(opt.image_path, hr_shape=(opt.hr_height, opt.hr_width)),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_cpu)
            for i, imgs in enumerate(dataloader):
                with torch.no_grad():
                    esr_image = denormalize(generator(imgs['lr'].to(device)).cpu())
                for j in range(len(esr_image)):
                    save_image(esr_image[j], os.path.join(opt.output_path, os.path.basename(files[i+j])))
        else:
            for f in tqdm(files):
                sr_image(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--output_path", type=str, default='images/outputs', help="Path where output will be saved")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
    parser.add_argument("--downsample", type=bool, default=False, help="Whether to downsample or not")
    opt = parser.parse_args()
    print(opt)
    test_image(opt)
