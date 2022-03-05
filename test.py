import sys
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 0  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.phase = ''

def main(argv):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    #visualizer = Visualizer(opt)
    # create website
    # web_dir = os.path.join(opt.results_dir, opt.name,
    #                        '%s_%s' % (opt.phase, opt.which_epoch))
    # webpage = html.HTML(
    #     web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase,
    #                                                           opt.which_epoch))
    # test
    avgPSNR = 0.0
    avgPSNR_b = 0.0
    #avgSSIM = 0.0
    #avgSSIM_b = 0.0
    counter = 0

    #--dataroot "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/" --name run_test --learn_residual --resize_or_crop resize_and_crop --which_model_netG resnet_9blocks  --batchSize 1 --gpu_id 0 --model RDehazingnet --R_Dehazing_premodel ./checkpoints/30_netR_Dehazing.pth
    for i, data in enumerate(dataset):
        img_name = data['A_paths'][0].split("/")[-1]
        print("File name: ", img_name)
        # if i >= opt.how_many:
        #     break

        counter = i
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        pil_result = Image.fromarray(visuals['r_dehazing_img'])
        pil_result.save("./results/" + img_name)
        #visualizer.display_current_results(visuals, counter)
        # PSNR_b = PSNR(visuals['syn_haze_img'], visuals['clear_img'])
        # PSNR_d = PSNR(visuals['s_dehazing_img'], visuals['clear_img'])
        # avgPSNR += PSNR_d
        # avgPSNR_b += PSNR_b
        #pilReala = Image.fromarray(visuals['real_A'])
        #pilFake = Image.fromarray(visuals['fake_B'])
        #pilReal = Image.fromarray(visuals['real_B'])
        #SSIM_b = SSIM(pilReala,pilReal)
        #SSIM_b = SSIM(pilReala).cw_ssim_value(pilReal)
        #SSIM_d = SSIM(pilFake).cw_ssim_value(pilReal)
        #avgSSIM += SSIM_d
        #avgSSIM_b += SSIM_b
        #img_path = model.get_image_paths()

if __name__=="__main__":
    main(sys.argv)