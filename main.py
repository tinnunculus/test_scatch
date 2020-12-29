import time
from PIL import Image
from torch.optim import Adam
import torch.nn.functional as F
import argparse
import torchvision
from torchvision.utils import save_image
from model import *
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def loss_function(x, output, mu, logvar):
    MLE = F.binary_cross_entropy(output, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MLE + KLD

class prepare_dataset():
    def __init__(self, train="C:/workspace/dataset/danbooru/archive/danbooru-images/danbooru-train", scatch_train="C:/workspace/dataset/danbooru/archive/danbooru-images/xdog-train", ref_train="C:/workspace/dataset/real_scatch", image_size = 256, batch_size = 1):

        x_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        r_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])

        x_dataset = datasets.ImageFolder(train, x_transform)
        scatch_dataset = datasets.ImageFolder(scatch_train, x_transform)
        ref_dataset = datasets.ImageFolder(ref_train, r_transform)
        self.x_train_loader = DataLoader(x_dataset, batch_size = batch_size)
        self.scatch_loader = DataLoader(scatch_dataset, batch_size = batch_size)
        self.ref_loader = DataLoader(ref_dataset, batch_size = batch_size)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    save_enc_model = "model/enc.pth"
    save_dec_model = "model/dec.pth"

    pre_dataset = prepare_dataset(image_size=args.image_size, train="C:/workspace/dataset/real_scatch", batch_size = args.batch_size)

    if args.pretrain == 1:
        encoder = real_scatch_ref_encoder(image_size = args.image_size).to(device)
        decoder = real_scatch_ref_decoder(image_size = args.image_size).to(device)
        state_enc_dict = torch.load(save_enc_model)
        state_dec_dict = torch.load(save_dec_model)
        encoder.load_state_dict(state_enc_dict)
        decoder.load_state_dict(state_dec_dict)
    else :
        encoder = real_scatch_ref_encoder(image_size = args.image_size).to(device)
        decoder = real_scatch_ref_decoder(image_size = args.image_size).to(device)

    op_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(op_parameters, 0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)


    for e in range(args.epochs):
        encoder.train()
        decoder.train()
        count = 0
        train_loss = 0


        for batch_id, (x,_) in enumerate(pre_dataset.x_train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            z, mu, logvar = encoder(x)
            y = decoder(z)
            loss = loss_function(x, y, mu,logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_id % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
                    e+1, batch_id * n_batch, len(pre_dataset.x_train_loader.dataset),
                    100. * batch_id / len(pre_dataset.x_train_loader),
                    loss.item() / len(x),time.ctime()))

                torchvision.utils.save_image(x.cpu(), "true.jpg")
                torchvision.utils.save_image(y.cpu(), "test.jpg")
        scheduler.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            e, train_loss / len(pre_dataset.x_train_loader.dataset)))

        #save model
        torch.save(encoder.state_dict(), save_enc_model)
        torch.save(decoder.state_dict(), save_dec_model)

def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    #torch.manual_seed(args.seed)


    with torch.no_grad():
        decoder = real_scatch_ref_decoder(image_size = args.image_size).to(device)

        state_dec_dict = torch.load("model/dec.pth")
        decoder.load_state_dict(state_dec_dict)
        decoder.eval()
        z1 = torch.randn([4,128]).to(device)
        output = decoder(z1)
        output = F.upsample(output.cpu(), scale_factor = 8)
    torchvision.utils.save_image(output, "output.jpg")

def gen_train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    save_gen_model = "model/gen.pth"
    save_ref_encoder = "model/enc.pth"
    pre_dataset = prepare_dataset(batch_size = args.batch_size)

    ref_encoder = real_scatch_ref_encoder(image_size = 256).to(device)
    state_enc_dict = torch.load(save_ref_encoder)
    ref_encoder.load_state_dict(state_enc_dict)

    if args.pretrain == 1:
        G = generator().to(device)
        state_gen_dict = torch.load(save_gen_model)
        G.load_state_dict(state_gen_dict)
    else :
        G = generator().to(device)

    g_optimizer = Adam(G.parameters(), 0.0001)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = 3, gamma = 0.5)
    mse_loss = torch.nn.MSELoss()

    for e in range(args.epochs):
        G.train()
        it_s = iter(pre_dataset.scatch_loader)
        it_r = iter(pre_dataset.ref_loader)
        count = 0

        for batch_id, (x, _) in enumerate(pre_dataset.x_train_loader):
            y, _ = next(it_s)
            try:
                r, _ = next(it_r)
            except:
                it_r = iter(pre_dataset.ref_loader)
                r, _ = next(it_r)
            n_batch = len(x)
            count += n_batch
            x = x.to(device)
            y = y.to(device)
            r = r.to(device)
            z, mu, logvar = ref_encoder(r)

            g_optimizer.zero_grad()
            result = G(x, z)
            loss = mse_loss(y, result)
            loss.backward()
            g_optimizer.step()
            if batch_id % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
                    e+1, batch_id * n_batch, len(pre_dataset.x_train_loader.dataset),
                    100. * batch_id / len(pre_dataset.x_train_loader),
                    loss.item() / len(x), time.ctime()))

                torchvision.utils.save_image(x.cpu(), "true.jpg")
                torchvision.utils.save_image(y.cpu(), "test.jpg")
        g_scheduler.step()











def load_image(filename, size = None, scale = None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def main():
    main_arg_parser = argparse.ArgumentParser(description='VAE Pretrain')
    main_arg_parser.add_argument('--batch-size', type=int, default = 4, metavar='N',
                        help='input batch size for training (default: 32)')
    main_arg_parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 10)')
    main_arg_parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    main_arg_parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    main_arg_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    main_arg_parser.add_argument("--image-size", type=int, default=512,
                                  help="size of input training images, default is 32 X 32")
    main_arg_parser.add_argument("--pretrain", type=int, default=0,
                                  help="There is training model?")
    main_arg_parser.add_argument("--train-test", type=int, default=0,
                                  help="Train or Test")
    args = main_arg_parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.train_test==0:
        gen_train(args)
    elif args.train_test == 1:
        test(args)

if __name__ == '__main__':
    main()