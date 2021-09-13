import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.plot_loss import Plot_loss

# TODO: try the following experiments to reduce the artifacts in the generated images:
#  instead of lsgan for gan_mode use opt.gan_mode =='hinge'
#  check the effects of opt.num_patches = 16 , it was 256 originally

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # opt.name = 'cmr_test_hinge_210720'
    # opt.name = 'cmr_test_original_cut_210721'
    # opt.name = 'cmr_test_original_cut_wgangp_210721'
    # opt.name = 'cmr_test_multi_scale_D_lsgan_210721'
    # opt.gan_mode =='wgangp'
    # opt.gan_mode =='lsgan'
    # opt.name = 'cmr_default_cut_210729'
    opt.name = 'sim2real_default_cut_210907'
    opt.dataset_mode = 'cmr'
    opt.model = 'cut'
    # opt.model = 'cycle_gan'
    # opt.nce_idt = True
    opt.amp = False
    # opt.direction = 'BtoA'
    opt.direction = 'AtoB'
    # opt.lambda_identity = 0.5
    # opt.image_dir_A = '/data/sina/dataset/seb/SA_files_processed/'
    opt.image_dir_A = '/data/sina/dataset/sim2real/sim_phases_train/'
    opt.image_dir_B = '/data/sina/dataset/seb/mms2_processed/Philips/'
    opt.max_dataset_size = 5850 
    opt.output_nc =  1 
    opt.input_nc = 1 
    opt.batch_size = 4
    # opt.num_patches = 16 

    opt.display_freq = 50
    opt.update_html_freq = 50
    opt.save_epoch_freq = 5
    opt.evaluation_freq = 500
    # opt.normG = 'batch'
    # opt.normD = 'batch'
    # opt.netD = 'n_layers'   # with the basic discriminator i see GAN artifacts in the translated images and that might be a sign of the fact that the discriminator gets trained faster
    # opt.netD = 'multi_scale'
    # opt.n_layers_D = 4
    opt.n_epochs  = 50
    opt.n_epochs_decay = 5
    opt.log_file = opt.checkpoints_dir + '/' + opt.name +  '/loss_log.txt'
    opt.loss_freq = 1000

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
 

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                


            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            if total_iters % opt.loss_freq == 0:  
                Plot_loss.plot_loss(opt)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
            

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
