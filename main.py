import os, scipy
import argparse
import numpy as np
import tensorflow as tf
from configs import prjt_configs
from GAN import GAN
from generator import UNet_builder
from discriminator import toy_discriminator
from utils.model_stru_vis import model_structure_vis
from utils.augmentation import tf_random_rotate_image
from dataloader.dataloader import ds



def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (_lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss





def main():
    parser = argparse.ArgumentParser()
    # Add '--image_folder' argument using add_argument() including a help. The type is string (by default):
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('--gen_input_shape', default=prjt_configs["model"]["gen_input_shape"])
    parser.add_argument('--disc_input_shape', default=prjt_configs["model"]["disc_input_shape"])
    parser.add_argument('--gen_filter_nums', type=int, default=prjt_configs["train"]["gen_filter_nums"])
    parser.add_argument('--disc_filter_nums', type=int, default=prjt_configs["train"]["disc_filter_nums"])
    parser.add_argument('--kernel_size', type=int, default=prjt_configs["train"]["kernel_size"]) 
    parser.add_argument('--g_lr', type=float, default=prjt_configs["train"]["g_lr"])
    parser.add_argument('--d_lr', type=float, default=prjt_configs["train"]["d_lr"])
    parser.add_argument('--lambda', type=int, default=prjt_configs["train"]["_lambda"])
    parser.add_argument('--batchsize', type=int, default=prjt_configs["train"]["_batchSize"])
    parser.add_argument('--epochs', type=int, default=prjt_configs["train"]["_epochs"])
    
    # Parse the argument and store it in a dictionary:
    args = vars(parser.parse_args())
    print(args)

    # Self-defined loss function for GAN
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    _lambda = args["lambda"]

    # The tuple may have some issue, find a way to deal with it
    generator = UNet_builder.build_U_Net3D(input_size=args["gen_input_shape"], 
                                           filter_num=args["gen_filter_nums"], 
                                           kernel_size=args["kernel_size"])

    discriminator = toy_discriminator.build_toy_discriminator(input_shape=args["disc_input_shape"], 
                                                              init_filter_nums=args["disc_filter_nums"], 
                                                              init_kernel_size=args["kernel_size"], 
                                                              kernel_init=tf.random_normal_initializer(0., 0.02), 
                                                              repetitions=1)
                                                              
    # Linked the generator and discriminator to create a pix2pix model
    pix2pix = GAN(discriminator=discriminator, generator=generator)

    # Set up optimisers 
    generator_optimizer = tf.keras.optimizers.Adam(args["g_lr"], beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(args["d_lr"], beta_1=0.5)

    # Compile the model 
    pix2pix.compile(g_optimizer= generator_optimizer, 
                    d_optimizer = discriminator_optimizer,
                    loss_fn= [generator_loss, discriminator_loss])
    
    #print(model_structure_vis(model=generator))
    #print(model_structure_vis(model=discriminator))

    # Set callbacks
    #callbacks = 

    # Fit the model 
    hist = pix2pix.fit(
        ds.skip(10).map(tf_random_rotate_image, num_parallel_calls=32)
        .shuffle(50).batch(args["batchsize"]), 
        epochs=args["epochs"],
        callbacks=None,
        verbose=1
        )


if __name__ == "__main__":
    main()