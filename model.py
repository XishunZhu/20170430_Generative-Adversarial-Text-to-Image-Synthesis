import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

class G_tconv(nn.Module):
    def __init__(self, nc, ngf):
        super(G_tconv, self).__init__()

        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ngf * 4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf * 8)

        self.convt1 = nn.ConvTranspose2d(1024, 128, 4, 1, 0, bias=False)

        self.convt2 = nn.ConvTranspose2d(ngf * 8 + 128, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(ngf * 8)
        self.convt3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(ngf * 4)
        self.convt4 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(ngf * 2)
        self.convt5 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(ngf)
        self.convt6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.conv_e1 = nn.Conv2d(ngf*8, ngf*2, 1, 1, 0, bias=False)
        self.bn_e1 = nn.BatchNorm2d(ngf*2)

        self.conv_e2 = nn.Conv2d(ngf*2, ngf*8, 3, 1, 1, bias=False)
        self.bn_e2 = nn.BatchNorm2d(ngf*8)


        self.conv_e3 = nn.Conv2d(ngf*4, ngf, 1, 1, 0, bias=False)
        self.bn_e3 = nn.BatchNorm2d(ngf)

        self.conv_e4 = nn.Conv2d(ngf, ngf*4, 3, 1, 1, bias=False)
        self.bn_e4 = nn.BatchNorm2d(ngf*4)


        self.linear = nn.Linear(1024, 128)

    def forward(self, batchSize, input1, input2):

        e2 = F.relu(self.conv1(input1))
        e3 = F.relu(self.batchnorm1(self.conv2(e2)))
        e4 = F.relu(self.batchnorm2(self.conv3(e3)))
        e5 = F.relu(self.batchnorm3(self.conv4(e4)))
        e6 = F.relu(self.batchnorm4(self.conv5(e5)))

        c1 = self.linear(input2.view(batchSize, 1024))
        c2 = c1.view(batchSize, 128, 1, 1)
        c3 = c2.expand(batchSize, 128, 4, 4)

        e8 = torch.cat((e6, c3), 1)
        d1_ = F.relu(self.batchnorm5(self.convt2(e8)))
        d1 = F.relu(self.bn_e2(self.conv_e2(F.relu(self.bn_e1(self.conv_e1(d1_))))))
        d2_ = F.relu(self.batchnorm6(self.convt3(d1)))
        d2 = F.relu(self.bn_e4(self.conv_e4(F.relu(self.bn_e3(self.conv_e3(d2_))))))
        d3_ = F.relu(self.batchnorm7(self.convt4(d2)))
        d4_ = F.relu(self.batchnorm8(self.convt5(d3_)))
        d5_ = self.convt6(F.relu(d4_))

        o1 = F.tanh(d5_)

        return o1


class G_unet_upsample(nn.Module):
    def __init__(self, nc, ngf):
        super(G_unet_upsample, self).__init__()

        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf * 8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf * 8)

        self.linear = nn.Linear(1024, 128)

        # self.upsamling = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsamling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv6 = nn.Conv2d(ngf * 8 + 128, ngf * 8, 1, 1, 0, bias=False)  # 8x8
        self.bn6 = nn.BatchNorm2d(ngf * 8)
        self.conv7 = nn.Conv2d(ngf * 16, ngf * 4, 3, 1, 1, bias=False)  # 16x16
        self.bn7 = nn.BatchNorm2d(ngf * 4)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 2, 3, 1, 1, bias=False)  # 32x32
        self.bn8 = nn.BatchNorm2d(ngf * 2)
        self.conv9 = nn.Conv2d(ngf * 4, ngf, 5, 1, 2, bias=False)  # 64x64
        self.bn9 = nn.BatchNorm2d(ngf)
        self.conv10 = nn.Conv2d(ngf, nc, 5, 1, 2, bias=False)  # 128x128

    def forward(self, batchSize, input1, input2):
        #                                                                   # input1: 3x128x128
        e2 = self.conv1(input1)                                             # 64 64x64
        e3 = self.bn2(self.conv2(F.leaky_relu(e2, 0.2, True)))              # 128 32x32
        e4 = self.bn3(self.conv3(F.leaky_relu(e3, 0.2, True)))              # 256 16x16
        e5 = self.bn4(self.conv4(F.leaky_relu(e4, 0.2, True)))              # 512 8x8
        e6 = self.bn5(self.conv5(F.leaky_relu(e5, 0.2, True)))              # 512 4x4

        #                                                                   # input2: 1024x1
        c1 = self.linear(input2.view(batchSize, 1024))  # 128
        c2 = c1.view(batchSize, 128, 1, 1)  # 128 1x1
        c3 = c2.expand(batchSize, 128, 4, 4)  # 128 4x4

        e8 = torch.cat((e6, c3), 1)  # 640 4x4
        d1_ = self.bn6(self.conv6(self.upsamling(F.relu(e8, True))))  # 512 8x8
        d1 = torch.cat((d1_, e5), 1)  # 1024 8x8
        d2_ = self.bn7(self.conv7(self.upsamling(F.relu(d1, True))))  # 256 16x16
        d2 = torch.cat((d2_, e4), 1)  # 512 16x16
        d3_ = self.bn8(self.conv8(self.upsamling(F.relu(d2, True))))  # 128 32x32
        d3 = torch.cat((d3_, e3), 1)  # 256 32x32
        d4_ = self.bn9(self.conv9(self.upsamling(F.relu(d3, True))))  # 64  64x64
        d5_ = self.conv10(self.upsamling(F.relu(d4_, True)))  # 3 128x128
        o1 = F.tanh(d5_)

        return o1


class G_unet_tconv(nn.Module):
    def __init__(self, nc, ngf):
        super(G_unet_tconv, self).__init__()

        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ngf * 4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf * 8)

        self.convt1 = nn.ConvTranspose2d(1024,128,4,1,0,bias=False)

        self.convt2 = nn.ConvTranspose2d(ngf * 8 + 128, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(ngf * 8)
        self.convt3 = nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(ngf * 4)
        self.convt4 = nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(ngf * 2)
        self.convt5 = nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(ngf)
        self.convt6 = nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False)


        self.linear = nn.Linear(1024, 128)

    def forward(self, batchSize, input1, input2):

        e2 = self.conv1(input1)
        e3 = self.batchnorm1(self.conv2(F.leaky_relu(e2)))
        e4 = self.batchnorm2(self.conv3(F.leaky_relu(e3)))
        e5 = self.batchnorm3(self.conv4(F.leaky_relu(e4)))
        e6 = self.batchnorm4(self.conv5(F.leaky_relu(e5)))

        c1 = self.linear(input2.view(batchSize, 1024))
        c2 = c1.view(batchSize, 128, 1, 1)
        c3 = c2.expand(batchSize, 128, 4, 4)

        e8 = torch.cat((e6, c3), 1)
        d1_ = self.batchnorm5(self.convt2(F.relu(e8)))
        d1 = torch.cat((d1_, e5), 1)
        d2_ = self.batchnorm6(self.convt3(F.relu(d1)))
        d2 = torch.cat((d2_, e4), 1)
        d3_ = self.batchnorm7(self.convt4(F.relu(d2)))
        d3 = torch.cat((d3_, e3), 1)
        d4_ = self.batchnorm8(self.convt5(F.relu(d3)))
        d4 = torch.cat((d4_, e2), 1)
        d5_ = self.convt6(F.relu(d4))

        o1 = F.tanh(d5_)

        return o1

class G_unet_tconv_convlast(nn.Module):
    def __init__(self, nc, ngf):
        super(G_unet_tconv_convlast, self).__init__()

        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ngf * 4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf * 8)

        self.convt1 = nn.ConvTranspose2d(1024,128,4,1,0,bias=False)

        self.convt2 = nn.ConvTranspose2d(ngf * 8 + 128, ngf * 8, 4, 2, 1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(ngf * 8)
        self.convt3 = nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(ngf * 4)
        self.convt4 = nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(ngf * 2)
        self.convt5 = nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(ngf)
        self.convt6 = nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False)

        self.conv_last = nn.Conv2d(nc, nc, 3, 1, 1, bias=False)

        self.linear = nn.Linear(1024, 128)

    def forward(self, batchSize, input1, input2):

        e2 = self.conv1(input1)
        e3 = self.batchnorm1(self.conv2(F.leaky_relu(e2)))
        e4 = self.batchnorm2(self.conv3(F.leaky_relu(e3)))
        e5 = self.batchnorm3(self.conv4(F.leaky_relu(e4)))
        e6 = self.batchnorm4(self.conv5(F.leaky_relu(e5)))

        c1 = self.linear(input2.view(batchSize, 1024))
        c2 = c1.view(batchSize, 128, 1, 1)
        c3 = c2.expand(batchSize, 128, 4, 4)

        e8 = torch.cat((e6, c3), 1)
        d1_ = self.batchnorm5(self.convt2(F.relu(e8)))
        d1 = torch.cat((d1_, e5), 1)
        d2_ = self.batchnorm6(self.convt3(F.relu(d1)))
        d2 = torch.cat((d2_, e4), 1)
        d3_ = self.batchnorm7(self.convt4(F.relu(d2)))
        d3 = torch.cat((d3_, e3), 1)
        d4_ = self.batchnorm8(self.convt5(F.relu(d3)))
        d4 = torch.cat((d4_, e2), 1)
        d5_ = self.convt6(F.relu(d4))
        d6 = self.conv_last(d5_)
        o1 = F.tanh(d6)

        return o1


class D_adv(nn.Module):
    def __init__(self, nc, ndf):
        super(D_adv, self).__init__()

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 4, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 2, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(ndf * 2)
        self.conv6 = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(ndf * 2)
        self.conv7 = nn.Conv2d(ndf * 2, ndf * 8, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(ndf * 8)
        self.conv8 = nn.Conv2d(ndf * 8 + 128, ndf * 8, 4, 2, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(ndf * 8)
        self.conv9 = nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False)

        self.linear = nn.Linear(1024, 128)

    def forward(self, batchSize, input1, input2):
        d1 = self.bn2(self.conv2(F.leaky_relu(self.conv1(input1), 0.2, True)))
        d2 = self.bn3(self.conv3(F.leaky_relu(d1, 0.2, True)))
        d3 = self.bn4(self.conv4(d2))

        v1 = self.bn5(self.conv5(d3))
        v2 = self.bn6(self.conv6(F.leaky_relu(v1, 0.2, True)))
        v3 = self.bn7(self.conv7(F.leaky_relu(v2, 0.2, True)))

        d4 = F.leaky_relu(d3 + v3)

        # f3 = F.leaky_relu(self.convt1(input2))
        c1 = self.linear(input2.view(batchSize, 1024))
        c2 = c1.view(batchSize, 128, 1, 1)
        f3 = c2.expand(batchSize, 128, 4, 4)

        o1 = torch.cat((d4, f3), 1)
        o2 = self.bn8(self.conv8(o1))
        o3 = (self.conv9(F.leaky_relu(o2, 0.2, True))).view(batchSize)
        o3 = o3.mean(0)

        return o3


class D_DCGAN(nn.Module):
    def __init__(self, nc, ndf):
        super(D_DCGAN, self).__init__()


        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)     # 64 64x64
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.extra_conv = nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.extra_bn = nn.BatchNorm2d(ndf)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)  # 128 32x32
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)  # 256 16x16
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # 512 8x8
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False)  # 512 4x4
        self.bn5 = nn.BatchNorm2d(ndf * 8)

        self.conv6 = nn.Conv2d(ndf * 8 + 128, 1, 4, 1, 0, bias=False)  # 1 1x1

        self.linear = nn.Linear(1024, 128)


    def forward(self, batchSize, input1, input2):

        o1 = self.lrelu(self.bn3(self.conv3(self.lrelu(self.bn2(self.conv2(self.lrelu(self.conv1(input1))))))))
        o2 = self.lrelu(self.bn5(self.conv5(self.lrelu(self.bn4(self.conv4(o1))))))

        c1 = self.linear(input2.view(batchSize, 1024))
        c2 = c1.view(batchSize, 128, 1, 1)
        c3 = c2.expand(batchSize, 128, 4, 4)

        o3 = torch.cat((o2, c3), 1)

        output = self.conv6(o3)

        output = output.view(batchSize).mean(0)

        return output


class DCGAN_D01(nn.Container):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D01, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input, dummy):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G01(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G01, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2  #ngf*8
            tisize = tisize * 2  #tisize=64

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))  #state batch*512*4*4

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2


        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))  #state batch*3*64*64
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)


class icml_G(nn.Module):
    def __init__(self, nc, ngf):
        super(icml_G, self).__init__()
        self.linear = nn.Linear(1024, 128)
        self.convt_228_512 = nn.ConvTranspose2d(100 + 128, ngf * 8, 4, 1, 0, bias=False)

        self.conv_512_128 = nn.Conv2d(ngf * 8, ngf * 2, 1, 1, 0, bias=False)
        self.conv_128_128 = nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False)
        self.conv_128_512 = nn.Conv2d(ngf * 2, ngf * 8, 3, 1, 1, bias=False)

        self.convt_512_256 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)

        self.conv_256_64 = nn.Conv2d(ngf * 4, ngf, 1, 1, 0, bias=False)

        self.conv_64_64 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.conv_64_256 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)

        self.convt_256_128 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)

        self.convt_128_64 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.convt_64_3 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.bn_512 = nn.BatchNorm2d(ngf * 8)
        self.bn_256 = nn.BatchNorm2d(ngf * 4)
        self.bn_128 = nn.BatchNorm2d(ngf * 2)
        self.bn_64 = nn.BatchNorm2d(ngf)

    def forward(self, txt, noise):  # input1 is txt; input2 is noise
        batchSize = txt.size(0)

        # txt: 1024, nose: 100
        txt_emd = F.leaky_relu(self.linear(txt.view(batchSize, 1024)), 0.2, True)
        input = torch.cat((txt_emd.view(batchSize, 128, 1, 1), noise), 1)  # input 228

        e1 = self.bn_512(self.convt_228_512(input))  # 4x4 512

        e2 = F.relu(self.bn_128(self.conv_512_128(e1)), True)  # 4x4 128
        e3 = F.relu(self.bn_128(self.conv_128_128(e2)), True)  # 4x4 128
        e4 = self.bn_512(self.conv_128_512(e3))  # 4x4 512

        e5 = self.bn_256(self.convt_512_256(F.relu((e1 + e4), True)))  # 8x8 256

        e6 = F.relu(self.bn_64(self.conv_256_64(e5)), True)  # 8x8 64
        e7 = F.relu(self.bn_64(self.conv_64_64(e6)), True)  # 8x8 64
        e8 = self.bn_256(self.conv_64_256(e7))  # 8x8 256

        e9 = F.relu(self.bn_128(self.convt_256_128(F.relu((e5 + e8), True))))  # 16x16 128
        e10 = F.relu(self.bn_64(self.convt_128_64(e9)))  # 32x32 64
        o = F.tanh(self.convt_64_3(e10))  # 64x64 3

        return o


class icml_D(nn.Module):
    def __init__(self, nc, ndf):
        super(icml_D, self).__init__()

        self.linear = nn.Linear(1024, 128)

        self.conv_3_64 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv_64_128 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv_128_256 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv_256_512 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv_512_128 = nn.Conv2d(ndf * 8, ndf * 2, 1, 1, 0, bias=False)
        self.conv_128_128 = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.conv_128_512 = nn.Conv2d(ndf * 2, ndf * 8, 3, 1, 1, bias=False)
        self.conv_640_512 = nn.Conv2d(ndf * 8 + 128, ndf * 8, 1, 1, 0, bias=False)
        self.conv_512_1 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        self.bn_128 = nn.BatchNorm2d(ndf * 2)
        self.bn_256 = nn.BatchNorm2d(ndf * 4)
        self.bn_512 = nn.BatchNorm2d(ndf * 8)


    def forward(self, img, txt):  # input1 is img; input2 is txt
        batchSize = img.size(0)

        e1 = F.leaky_relu(self.conv_3_64(img), 0.2, True)  # 64 32x32
        e2 = F.leaky_relu(self.bn_128(self.conv_64_128(e1)), 0.2, True)  # 128 16x16
        e3 = F.leaky_relu(self.bn_256(self.conv_128_256(e2)), 0.2, True)  # 256 8x8
        e4 = self.bn_512(self.conv_256_512(e3))  # 512 4x4

        e5 = F.leaky_relu(self.bn_128(self.conv_512_128(e4)), 0.2, True)
        e6 = F.leaky_relu(self.bn_128(self.conv_128_128(e5)), 0.2, True)
        e7 = self.bn_512(self.conv_128_512(e6))

        e8 = F.leaky_relu((e4 + e7), 0.2, True)  # 512 4x4

        txt_emb = F.leaky_relu(self.linear(txt.view(batchSize, 1024)), 0.2, True)  # 128
        t = txt_emb.view(batchSize, 128, 1, 1)
        e9 = t.expand(batchSize, 128, 4, 4)  # 128 4x4
        e12 = torch.cat((e8, e9), 1)  # 640 4x4
        e13 = F.leaky_relu(self.bn_512(self.conv_640_512(e12)), 0.2, True)  # 512 4x4
        e14 = self.conv_512_1(e13)   # 1
        o = e14.view(batchSize)
        o = F.sigmoid(o)

        return o


class icml_D_simple(nn.Module):
    def __init__(self, nc, ndf):
        super(icml_D_simple, self).__init__()

        self.linear = nn.Linear(1024, 128)

        self.conv_3_64 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv_64_128 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv_128_256 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv_256_512 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv_512_128 = nn.Conv2d(ndf * 8, ndf * 2, 1, 1, 0, bias=False)
        self.conv_128_128 = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.conv_128_512 = nn.Conv2d(ndf * 2, ndf * 8, 3, 1, 1, bias=False)
        self.conv_640_512 = nn.Conv2d(ndf * 8 + 128, ndf * 8, 1, 1, 0, bias=False)
        self.conv_512_1 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        self.bn_128 = nn.BatchNorm2d(ndf * 2)
        self.bn_256 = nn.BatchNorm2d(ndf * 4)
        self.bn_512 = nn.BatchNorm2d(ndf * 8)


    def forward(self, img, txt):  # input1 is img; input2 is txt
        batchSize = img.size(0)

        e1 = F.leaky_relu(self.conv_3_64(img), 0.2, True)  # 64 32x32
        e2 = F.leaky_relu(self.bn_128(self.conv_64_128(e1)), 0.2, True)  # 128 16x16
        e3 = F.leaky_relu(self.bn_256(self.conv_128_256(e2)))  # 256 8x8
        e4 = self.bn_512(self.conv_256_512(e3))  # 512 4x4

        # e8 = F.leaky_relu((e4 + e7), 0.2, True)  # 512 4x4
        # e8 = torch.cat((e4, e7), 1) # 1024 4x4

        txt_emb = F.leaky_relu(self.linear(txt.view(batchSize, 1024)), 0.2, True)  # 128
        t = txt_emb.view(batchSize, 128, 1, 1)
        e9 = t.expand(batchSize, 128, 4, 4)  # 128 4x4
        e12 = torch.cat((e4, e9), 1)  # 640 4x4
        e13 = F.leaky_relu(self.bn_512(self.conv_640_512(e12)), 0.2, True)  # 512 4x4
        # d3 = F.sigmoid(self.conv9(d2))
        e14 = self.conv_512_1(e13)   # 1
        o = e14.view(batchSize)

        return o


class icml_D_noskip(nn.Module):
    def __init__(self, nc, ndf):
        super(icml_D_noskip, self).__init__()

        self.linear = nn.Linear(1024, 128)

        self.conv_3_64 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv_64_128 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv_128_256 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv_256_512 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv_512_128 = nn.Conv2d(ndf * 8, ndf * 2, 1, 1, 0, bias=False)
        self.conv_128_128 = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.conv_128_512 = nn.Conv2d(ndf * 2, ndf * 8, 3, 1, 1, bias=False)
        self.conv_640_512 = nn.Conv2d(ndf * 8 + 128, ndf * 8, 1, 1, 0, bias=False)
        self.conv_512_256 = nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False)
        self.conv_256_128 = nn.Conv2d(ndf * 4, ndf * 2, 1, 1, 0, bias=False)
        self.conv_128_64 = nn.Conv2d(ndf * 2, ndf * 1, 1, 1, 0, bias=False)
        self.conv_64_1 = nn.Conv2d(ndf, 1, 4, 1, 0, bias=False)

        self.conv_512_1 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        self.conv_576_1 = nn.Conv2d(576, 1, 4, 1, 0, bias=False)


        self.bn_64 = nn.BatchNorm2d(ndf)
        self.bn_128 = nn.BatchNorm2d(ndf * 2)
        self.bn_256 = nn.BatchNorm2d(ndf * 4)
        self.bn_512 = nn.BatchNorm2d(ndf * 8)

    def forward(self, img, txt):  # img: 3 64x64
        batchSize = img.size(0)
        txt_emb = F.leaky_relu(self.linear(txt.view(batchSize, 1024))) # 128


        e1 = F.leaky_relu(self.conv_3_64(img), 0.2, True)  # 64 32x32
        e2 = F.leaky_relu(self.bn_128(self.conv_64_128(e1)), 0.2, True)  # 128 16x16
        e3 = F.leaky_relu(self.bn_256(self.conv_128_256(e2)), 0.2, True)  # 256 8x8
        e4 = F.leaky_relu(self.bn_512(self.conv_256_512(e3)), 0.2, True)  # 512 4x4

        t = txt_emb.view(batchSize, 128, 1, 1)
        e5 = t.expand(batchSize, 128, 4, 4)  # 128 4x4
        e6 = torch.cat((e4, e5), 1)  # 640 4x4



        e7 = F.leaky_relu(self.bn_512(self.conv_640_512(e6)), 0.2, True)  # 512 4x4
        e8 = F.leaky_relu(self.bn_256(self.conv_512_256(e7)), 0.2, True)  # 256 4x4
        e9 = F.leaky_relu(self.bn_128(self.conv_256_128(e8)), 0.2, True)  # 128 4x4
        e10 = F.leaky_relu(self.bn_64(self.conv_128_64(e9)), 0.2, True)  # 64 4x4

        e11 = torch.cat((e4, e10), 1)
        e12 = self.conv_576_1(e11)

        #e12 = self.conv_64_1(e10)   # 1
        o = e12.view(batchSize)

        return o
