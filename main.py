from __future__ import print_function
import os
import time
import random
import argparse
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.transforms as transforms
import model as M
import util as Util
import Dataset
from torch.utils.serialization import load_lua


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/home/ubuntu/zl/icml2016-master/raw_datasets')
parser.add_argument('--workers', default=5)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--diter', type=int, default=3)
parser.add_argument('--niter', type=int, default=800)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--imSize', type=int, default=64)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--experiment', default='./502')
parser.add_argument('--log', default='log.txt')
parser.add_argument('--lrD', type=float, default=0.0000625)
parser.add_argument('--lrG', type=float, default=0.0002)
parser.add_argument('--decay_every', type=int, default=100)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--GPU', type=int, default=1, help='<0: use all GPUs; otherwise: use the specified GPU')
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--resume', type=int, default=1, help='>0: will resume from the latest model')
parser.add_argument('--display', type=int, default=1)

opt = parser.parse_args()
print(opt)

log_file = opt.experiment + '/' + opt.log

if not os.path.isdir(opt.experiment):
    os.system('mkdir {0}'.format(opt.experiment))


noise = torch.FloatTensor(opt.batchSize, 100, 1, 1)
noise_int = torch.FloatTensor(opt.batchSize*3/2, 100, 1, 1)
# for test
fixed_noise = torch.FloatTensor(opt.batchSize, 100, 1, 1).normal_(0, 1)
fixed_noise =fixed_noise.cuda()
fixed_noise_v = Variable(fixed_noise)
fixed_txt = load_lua('./txt1024.t7')  #this flower has white petals and a yellow stamen
fixed_txt = fixed_txt.view(1, 1024, 1, 1)
fixed_txt = fixed_txt.expand(opt.batchSize, 1024, 1, 1)
fixed_txt = fixed_txt.cuda()
fixed_txt_v = Variable(fixed_txt)

input_img = torch.FloatTensor(opt.batchSize, opt.nc, opt.imSize, opt.imSize)
input_txt = torch.FloatTensor(opt.batchSize, 1024, 1, 1)
input_txt_int = torch.FloatTensor(opt.batchSize*3/2, 1024, 1, 1)
input_txt_int_cpu = torch.FloatTensor(opt.batchSize*3/2, 1024, 1, 1)
fake_img = torch.FloatTensor(opt.batchSize, 3, opt.imSize, opt.imSize)
fake_img_int = torch.FloatTensor(opt.batchSize*3/2, 3, opt.imSize, opt.imSize)
wrong_img = torch.FloatTensor(opt.batchSize, 3, opt.imSize, opt.imSize)
target_img = torch.FloatTensor(opt.batchSize, 3, opt.imSize, opt.imSize)
noise, noise_int = noise.cuda(), noise_int.cuda()

input_img = input_img.cuda()
input_txt = input_txt.cuda()
fake_img = fake_img.cuda()
target_img = target_img.cuda()
wrong_img = wrong_img.cuda()

input_txt_int = input_txt_int.cuda()
fake_img_int = fake_img_int.cuda()

trans_target = transforms.Compose([
    transforms.Scale(opt.imSize),
    transforms.CenterCrop(opt.imSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# fix seed
opt.manualSeed = random.randint(1, 10000)
print('Random Seed: %d' % (opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


netG = M.icml_G(opt.nc, opt.ngf)
netD = M.icml_D(opt.nc, opt.ndf)


netG.apply(Util.weights_init)
netD.apply(Util.weights_init)


if opt.resume > 0:
    netG.load_state_dict(torch.load('{0}/latestG.pth'.format(opt.experiment)))
    netD.load_state_dict(torch.load('{0}/latestD.pth'.format(opt.experiment)))


if opt.GPU >= 0:
    netD.cuda()
    netG.cuda()
else:
    netD.cuda()
    netG.cuda()
    netG = torch.nn.DataParallel(netG, device_ids=[0, 1])
    netD = torch.nn.DataParallel(netD, device_ids=[0, 1])



train_dataset = Dataset.Dataset_flower(root=opt.dataroot,
                           train=True,
                           transform=trans_target,)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=int(opt.workers))



criterion = nn.BCELoss()

# setup optimizer
# optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD, weight_decay=0.001)
# optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG, weight_decay=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(0.5, 0.999))

# training
cnt_G, cnt_D = 0, 0
for epoch in range(opt.niter):
    if epoch % opt.decay_every == 9:
        optimizerD.param_groups[0]['lr'] = optimizerD.param_groups[0]['lr'] * opt.lr_decay
        optimizerG.param_groups[0]['lr'] = optimizerG.param_groups[0]['lr'] * opt.lr_decay

    for i, (input_txt_cpu, target_img_cpu, wrong_img_cpu, raw_txt_cpu) in enumerate(train_loader):
        if (i+1) == len(train_loader): # '+1'skips the last iteration of epoch
            break

        start_time = time.time()
        # 1. Update D network [Multiple Times]
        for p in netD.parameters():  # reset requires_grad: they are set to False below in netG update
            p.requires_grad = True

        # prepare data
        input_txt_cpu = input_txt_cpu.cuda()
        target_img_cpu = target_img_cpu.cuda()
        wrong_img_cpu = wrong_img_cpu.cuda()

        input_txt_int.narrow(0, 0, opt.batchSize).copy_(input_txt_cpu)
        input_txt_int.narrow(0, opt.batchSize, opt.batchSize / 2).copy_(input_txt_cpu.narrow(0, 0, opt.batchSize / 2))
        input_txt_int.narrow(0, opt.batchSize, opt.batchSize / 2).add_(input_txt_cpu.narrow(0, opt.batchSize / 2, opt.batchSize / 2))
        input_txt_int.narrow(0, opt.batchSize, opt.batchSize / 2).mul_(0.5)

        input_txt_int_v = Variable(input_txt_int)


        input_txt.copy_(input_txt_cpu)
        target_img.copy_(target_img_cpu)
        wrong_img.copy_(wrong_img_cpu)

        input_txt_v = Variable(input_txt)
        target_img_v = Variable(target_img)
        wrong_img_v = Variable(wrong_img)

        # 1. training D
        netD.zero_grad()
        # 1.1 train with real example

        # errD_real = - torch.mean(netD(target_img_v, input_txt_v))               # - o_real
        label = torch.ones(64)
        label = Variable(label).cuda()
        output = netD(target_img_v, input_txt_v)
        errD_real = criterion(output, label)

        # errD_real.backward()

        # 1.2 train with fake example. # Note: need to freeze G for now
        noise.resize_(opt.batchSize, 100, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)  # totally freeze netG
        input_txt_v_v = Variable(input_txt, volatile=True)
        fake_img_v = Variable(netG(input_txt_v_v, noisev).data)
        # errD_fake = torch.mean(netD(fake_img_v, input_txt_v)) - 1               # o_fake - 1
        label = torch.zeros(64)
        label = Variable(label).cuda()
        output = netD(fake_img_v, input_txt_v)
        errD_fake = 0.5 * criterion(output, label)

        # errD_fake.backward()

        # 1.3 train with wrong example.
        # errD_wrong = (torch.mean(netD(wrong_img_v, input_txt_v)) - 1)  # o_wrong - 1
        label = torch.zeros(64)
        label = Variable(label).cuda()
        output = netD(wrong_img_v, input_txt_v)
        errD_wrong = 0.5 * criterion(output, label)

        # errD_wrong.backward()

        # getting the gross error and BP
        # errD = 0.5*errD_fake + errD_real + 0.5*errD_wrong
        errD = errD_fake + errD_real + errD_wrong
        errD.backward()
        optimizerD.step()
        cnt_D += 1

        # for p in netD.parameters():  # chop parameters [lower, upper]
        #     p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        Util.log(log_file, '[%d/%d - %d/%d] [%s] [time: %.3fs]      [errD: %.4f   o_real: %.4f   o_fake: %.4f]'
              % (epoch + 1, opt.niter, i + 1, len(train_loader), time.strftime("%m-%d %H:%M:%S"), (time.time() - start_time),
                 errD.data[0], - errD_real.data[0], errD_fake.data[0] + 1))

        # if (cnt_D % 50 == 0 and cnt_G < 2) or (cnt_D % opt.diter == 0 and cnt_G >= 2):
        if True:
            # 2 Update G network
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation. I don't want/need to update D for now
            netG.zero_grad()

            noise_int.resize_(opt.batchSize*3/2, 100, 1, 1).normal_(0, 1)


            noise_int_v = Variable(noise_int)
            fake_img_int = netG(input_txt_int_v, noise_int_v)
            # errG = - torch.mean(netD(fake_img_int, input_txt_int_v))
            label = torch.ones(96)
            label = Variable(label).cuda()
            output = netD(fake_img_int, input_txt_int_v)
            errG = criterion(output, label)

            errG.backward()
            optimizerG.step()
            cnt_G += 1


            # visualization & log
            output = vutils.make_grid(fake_img_int.data.cpu(), padding=2)
            # input = vutils.make_grid(input_img.cpu(), padding=2)
            # target = vutils.make_grid(target_img.cpu(), padding=2)

            vutils.save_image(output, './output2.png')
            # vutils.save_image(input, './input.png')
            # vutils.save_image(target, './target.png')

            Util.log(log_file, '[%d/%d - %d/%d] [%s] [time: %.3fs]      [errD: %.4f   o_real: %.4f   o_fake: %.4f   o_G: %.4f]'\
                  % (epoch + 1, opt.niter, i + 1, len(train_loader), time.strftime("%m-%d %H:%M:%S"),\
                   (time.time() - start_time), errD.data[0], - errD_real.data[0], errD_fake.data[0] + 1, - errG.data[0]))

    # save checkpoints
    if (epoch+1) % opt.save_every == 0:
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch+1))
        # torch.save(netG.state_dict(), '{0}/latestG.pth'.format(opt.experiment, epoch+1))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch+1))
        # torch.save(netD.state_dict(), '{0}/latestD.pth'.format(opt.experiment, epoch+1))
        vutils.save_image(fake_img_int.data, '{0}/{1}.png'.format(opt.experiment, epoch+1))

        alphabet = '''abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} '''
        raw_txt_cpu1 = raw_txt_cpu[0]
        for k in range(raw_txt_cpu1.size(0)):
            if raw_txt_cpu1[k] != 0.0:
                c = raw_txt_cpu1[k] - 1
                c = int(c)
                out_txt = alphabet[c]
                print(out_txt)
                # Util.log(out_txt, '%s' % (out_txt))

        # test
        testIm = netG(input_txt_v, fixed_noise_v)
        vutils.save_image(testIm.data, '{0}/test_{1}.png'.format(opt.experiment, epoch+1))
