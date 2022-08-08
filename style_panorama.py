import numpy as np
import torch
import os
import argparse
import time
import datetime
from pathlib import Path
from PIL import Image

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from data_handling import MyDataset
from generate_dataset import get_new_perspectives_and_masks
from network import ImageTransformNet
from panorama_network import PanoramaTransformNet
from vgg import Vgg16


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7

def train(args):          
    # GPU enabling
    dtype=torch.FloatTensor
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(0)
        print("Current device: %d" %torch.cuda.current_device())

    # visualization of training controlled by flag
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Resize(300),                  # scale shortest side to image_size
            transforms.CenterCrop(300),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        test_panorama1 = {}

        # Test image panorama 1
        panorama1_name = 'panorama1'
        panorama1_path = 'content_imgs/panorama1'
        for f in os.listdir(panorama1_path):
            filename=os.path.split(f)[1].split('.')[0]

            im = utils.load_image(os.path.join(panorama1_path, f))
            im = img_transform_512(im)
            #im =  Variable(im.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

            test_panorama1[filename] = im.cpu().detach().numpy()


        # Test image panorama 2
        test_panorama2 = {}
        panorama2_name='panorama2'
        panorama2_path = 'content_imgs/panorama2'
        for f in os.listdir(panorama2_path):
            filename = os.path.split(f)[1].split('.')[0]

            im = utils.load_image(os.path.join(panorama2_path, f))
            im = img_transform_512(im)
            #im = Variable(im.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

            test_panorama2[filename] = im.cpu().detach().numpy()




    # define networks
    # Image transformer is used for the first frame only, pretrained model
    image_transformer = ImageTransformNet().type(dtype)
    image_transformer.load_state_dict(torch.load('./models/image/mosaic.model'))

    # continue training previous model
    if args.model_path is None:
        print(f'- Initializing model.')
        panorama_transformer = ImageTransformNet().type(dtype)
        panorama_transformer.load_state_dict(torch.load('./models/image/mosaic.model'))
    else:
        print(f'- Loading model from path={args.model_path}')
        panorama_transformer = ImageTransformNet().type(dtype)
        panorama_transformer.load_state_dict(torch.load(args.model_path))


    optimizer = Adam(panorama_transformer.parameters(), LEARNING_RATE)
    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # Load training dataset
    input_path = args.dataset
    print(f'- Preparing the training dataset...')
    training_dataset = MyDataset(input_path)
    loader_training = DataLoader(
        dataset=training_dataset,
        batch_size=BATCH_SIZE)
    print(f'- Training dataset loaded. Dataset size={training_dataset.__len__()}')

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    style = utils.load_image(args.style_image)
    style = style_transform(style)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)
    style_name = os.path.split(args.style_image)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    EPOCHS = args.epochs
    print(f'- Start training...\n- Parameters:\n - Epochs={EPOCHS}\n - Batch size={BATCH_SIZE}\n'
          f' - Learning Rate={LEARNING_RATE}\n - Style weight={STYLE_WEIGHT}\n'
          f' - Content Weight={CONTENT_WEIGHT}\n - TV Weigth={TV_WEIGHT}\n')

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0
        aggregate_overlap_loss = 0.0

        # train network
        panorama_transformer.train()
        image_transformer.eval()

        # inputs: (Image1, image1 id (ids are numbers 1-6), neighboring image, neighboring image id)
        for batch_num, x in enumerate(loader_training):
            img_batch_read = len(x[0])
            img_count += img_batch_read

            # Unpack the input data
            x1=Variable(x[:][0]).type(dtype)
            x1_ids=x[:][1].cpu().detach().numpy()
            x2=Variable(x[:][2]).type(dtype)
            x2_ids=x[:][3].cpu().detach().numpy()

            # Stylize the first frames of the batch using image_transfromer model
            stylized_x1 = image_transformer(x1)
            stylized_x1_np = stylized_x1.cpu().detach().numpy()
            stylized_x1_np = np.moveaxis(stylized_x1_np, 1, 3)

            #stacked_x = []
            persperctives=[]
            masks=[]

            # calculate perspectives and masks, stack all to a shape (BATCH_SIZE, 9, 256, 256)
            for i in range(len(stylized_x1_np)):
                pers, mask = utils.get_perspective([(stylized_x1_np[i], x1_ids[i]), (x2[i], x2_ids[i])])

                # from numpy shape (WIDTH, HEIGHT, COLOR) to (COLOR, WIDTH, HEIGHT)
                pers = torch.from_numpy(pers).permute(2,0,1)
                mask = torch.from_numpy(mask).permute(2,0,1)

                persperctives.append(pers)
                masks.append(mask)

                # Stack all images together
                #stacked_x.append(torch.cat((x2[i], Variable(pers).type(dtype), Variable(mask).type(dtype)), 0))
            persperctives = torch.stack(persperctives)
            masks = torch.stack(masks)

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            #stacked_x = Variable(stacked_x).type(dtype)
            persperctives = Variable(persperctives).type(dtype)
            masks = Variable(masks).type(dtype)
            y_hat = panorama_transformer(x2)

            # get vgg features
            y_c_features = vgg(x2)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # overlap loss
            overlaps=[]
            for i, im in enumerate(x2):
                overlaps.append(im*masks[i])
            overlaps = torch.stack(overlaps)
            overlap_loss = loss_mse(persperctives, overlaps)
            aggregate_overlap_loss+=overlap_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss + overlap_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  agg_overlap: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f}  overlap: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(training_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0), aggregate_overlap_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item(), overlap_loss.item()
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                panorama_transformer.eval()

                save_path=f"visualization/{style_name}/panoramas/{str(int(datetime.datetime.utcnow().timestamp()))}-{panorama2_name}-{e+1}-{batch_num+1}"
                Path(save_path).mkdir(parents=True, exist_ok=True)

                ########################
                # This is for the stacked image model stuff

                # Unpack the input data
                # x1 = Variable(torch.from_numpy(np.array(list(test_panorama2.values())))).type(dtype)
                # x1_ids = list(test_panorama2.keys())

                # print(x1.shape)
                #
                # # Stylize the first frames of the batch using image_transfromer model
                # stylized_x1 = image_transformer(x1)
                # stylized_x1_np = stylized_x1.cpu().detach().numpy()
                # stylized_x1_np = np.moveaxis(stylized_x1_np, 1, 3)
                #
                # stylized_cubemap = {}
                # for i in range(len(stylized_x1_np)):
                #     stylized_cubemap[x1_ids[i]] = stylized_x1_np[i]
                #
                # # Calculate all new perspectives and masks
                # combined_masks = {}
                # combined_pers={}
                #
                # pers, mask = get_new_perspectives_and_masks(cubemap=stylized_cubemap, width=300, height=300)
                #
                # for key in mask.keys():
                #     combined_mask = np.zeros((stylized_x1_np[0].shape))
                #     combined_stylized = np.zeros((stylized_x1_np[0].shape))
                #     for key2 in mask[key].keys():
                #         combined_mask +=mask[key][key2]
                #         combined_stylized += pers[key][key2]
                #     combined_masks[key] = combined_mask
                #     combined_pers[key] = combined_stylized

                ########################

                # calculate perspectives and masks, stack all to a shape (BATCH_SIZE, 9, 256, 256)
                stylized_testpanorama2 = {}
                for key in test_panorama2.keys():
                    im = torch.from_numpy(test_panorama2[key])
                    #im = torch.permute(im, (2, 1, 0))
                    im = Variable(im[None, :]).type(dtype)

                    ########################
                    # This is for the stacked image model stuff

                    # # from numpy shape (WIDTH, HEIGHT, COLOR) to (COLOR, WIDTH, HEIGHT)
                    # im = torch.from_numpy(test_panorama2[key])
                    # pers = torch.from_numpy(combined_pers[key]).permute(2, 0, 1)
                    # mask = torch.from_numpy(combined_masks[key]).permute(2, 0, 1)

                    # Stack all images together
                    #stacked_x=torch.cat((Variable(im).type(dtype), Variable(pers).type(dtype), Variable(mask).type(dtype)), 0)

                    #stacked_x = stacked_x[None,:]

                    ########################

                    outputTestImage = panorama_transformer(im).cpu()
                    im_path = f"{save_path}/{key}.jpg"
                    utils.save_image(im_path, outputTestImage.data[0])

                    stylized_testpanorama2[key] = utils.transform_image_to_original(outputTestImage.data[0])

                # Create and save a equirectangular image from cubemap
                eq_im=utils.cubemap_to_equirectangular(stylized_testpanorama2)
                eq_im = Image.fromarray(eq_im)
                eq_im.save(f"{save_path}/equirectangular_img.jpg")

                print(f"\n- Visualization images saved to the path=visualization/{style_name}/panoramas/panorama1-{e+1}-{batch_num+1}")

                Path('models/panorama').mkdir(parents=True, exist_ok=True)
                filename = "models/panorama/" + str(style_name) + ".model"
                torch.save(obj=panorama_transformer.state_dict(), f=filename)

                print('- Model checkpoint saved.\n')

                panorama_transformer.train()

    # save model
    panorama_transformer.eval()

    if use_cuda:
        image_transformer.cpu()
        panorama_transformer.cpu()

    Path('models/panorama').mkdir(parents=True, exist_ok=True)

    filename = "models/panorama/" + str(style_name) + ".model"
    torch.save(obj=panorama_transformer.state_dict(), f=filename)

    print('- Model checkpoint saved.')
    print('- Training Done.')
    
    if use_cuda:
        image_transformer.cuda()
        panorama_transformer.cuda()

def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.device('cuda')
        print("Current device: %d" %torch.cuda.current_device())

    # content image
    img_transform_512 = transforms.Compose([
            transforms.Resize(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs. Default 1")
    train_parser.add_argument("--model-path", type=str, default=None, help="Model path to continue pretrained model.")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")
    style_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print("Training!")
        train(args)
    elif (args.subcommand == "transfer"):
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()








