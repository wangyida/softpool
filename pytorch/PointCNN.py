from Utils import knn_indices_func_cpu
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch import cuda, FloatTensor, LongTensor
from xConv import XConv,Dense
from collections import OrderedDict


class PointCnnLayer(nn.Module):
    def __init__(self,features,settings):
        """
        :param points: Points cloud
        :param features: features as input, can be None if nothing known
        :param settings: Settings of the network, inside there are:
            :setting xcon parameters : parameters for the xconvolutions : xconv_param_name = ('K', 'D', 'P', 'C') "C = C_out" 8, 1, -1, 32 * x
            : setting fc parameters : parameters for the fully convolutional part of the network  :  fc_param_name = ('C', 'dropout_rate')
        """
        super(PointCnnLayer,self).__init__()
        # N = pc
        #print("There are " + str(N) + "  points in the begining")
        with_X_transformation = True ## investigating that
        sorting_method = None ## sorting or not points along a dimension
        sampling = 'fps' ## investigating that
        self.nb_xconv = len(settings[0])
        self.settings = settings

        #C_mid = C_out // 2 if C_in == 0 else C_out // 4
        #depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)])
        self.xvonc1 = XConv(C_in = 0,
                            C_out = settings[0][0].get('C'),
                            dims = 3,
                            K = settings[0][0].get('K'),
                            P = self.settings[0][0].get('P'),
                            C_mid = settings[0][0].get('C') //2 ,
                            depth_multiplier = 4)  ## First XConvolution

        self.dense1 = Dense(settings[0][0].get('C'),settings[0][1].get('C')//2,drop=0)
        self.xvonc2 = XConv(C_in = settings[0][1].get('C')//2,
                            C_out = settings[0][1].get('C'),
                            dims = 3,
                            K =  settings[0][1].get('K'),
                            P = self.settings[0][1].get('P'),
                            C_mid = settings[0][1].get('C') // 4 ,
                            depth_multiplier = settings[0][0].get('C')//4)  ## Second XConvolution

        self.dense2 = Dense(settings[0][1].get('C'),settings[0][2].get('C')//2,drop=0)
        self.xvonc3 = XConv(C_in = settings[0][2].get('C')//2,
                            C_out = settings[0][2].get('C'),
                            dims = 3,
                            K =  settings[0][2].get('K'),
                            P = self.settings[0][2].get('P'),
                            C_mid = settings[0][2].get('C') // 4 ,
                            depth_multiplier = settings[0][1].get('C')//4)  ## Third XConvolution

        self.dense3 = Dense(settings[0][2].get('C'),settings[0][3].get('C')//2,drop=0)
        self.xvonc4 = XConv(C_in = settings[0][3].get('C')//2,
                            C_out = settings[0][3].get('C'),
                            dims = 3,
                            K =  settings[0][3].get('K'),
                            P = self.settings[0][3].get('P'),
                            C_mid = settings[0][3].get('C') // 4 ,
                            depth_multiplier = settings[0][2].get('C')//4)  ## Third XConvolution
        self.layers_conv =[self.xvonc1,self.xvonc2,self.xvonc3,self.xvonc4]
        ## deconvolution inputs :  pts (output of previous conv/deconv/), fts (output of previous conv/deconv/), N , K, D, P (of concatenated layer output),
        #C (of concatenated layer output),C_prev (C of previous layer ) // 4 , depth_multiplier = 1
        #    xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
        #print(self.layers_conv)
        deconvolutions = OrderedDict()
        dense_deconv =  OrderedDict()
        fc =  OrderedDict()
        for i in range(0,len(settings[1])):
            deconvolutions["deconv" + str(i)] = XConv(C_in = self.layers_conv[settings[1][i].get('pts_layer_idx')].C ,
                                C_out = self.layers_conv[settings[1][i].get('qrs_layer_idx')].C,
                                dims = 3,
                                K =  settings[1][i].get('K'),
                                P = self.layers_conv[settings[1][i].get('qrs_layer_idx')].P,
                                C_mid = self.layers_conv[settings[1][i].get('pts_layer_idx')].C//4 ,
                                depth_multiplier = 1)  ## First dXConvolution

            dense_deconv["dense" + str(i)] = Dense(self.layers_conv[settings[1][i].get('qrs_layer_idx')].C*2,self.layers_conv[settings[1][i].get('qrs_layer_idx')].C)
        self.deconvolutions = nn.Sequential(deconvolutions)
        self.dense_deconv = nn.Sequential(dense_deconv)


        for i in range(0,len(settings[2])):
            fc["fc" + str(i)] = Dense(self.dense_deconv[-1].out if i==0 else fc["fc" + str(i-1)].out,
                                settings[2][i].get('C'),
                                drop=settings[2][i].get('dropout_rate'),
                                acti=True)

        self.fc = nn.Sequential(fc)


        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        layer_pts = [x]
        outs = [None]
        pts_regionals = []
        fts_regionals = []
        #print("First CONVOLUTION")
        if(self.settings[0][0].get('P')!=-1):
            idxx = np.random.choice(x.size()[1], self.settings[0][0].get('P'), replace = False).tolist() ## select representative points
            rep_pts = x[:,idxx,:]
        else:
            rep_pts = x
        pts_idx = knn_indices_func_cpu(rep_pts,x,self.settings[0][0].get('K'),self.settings[0][0].get('D') )
        #pts_idx = pts_idx[:,::self.settings[0][0].get('D'),:]
        pts_regional = torch.stack([x[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        out = self.xvonc1(rep_pts,pts_regional,None) ## FTS
        layer_pts.append(rep_pts)
        outs.append(out)
        pts_regionals.append(pts_regional)
        fts_regionals.append(None)

        #print("SECOND CONVOLUTION")
        if(not (self.settings[0][1].get('P')==-1)):
            # print("been there" + str(self.settings[0][1].get('P')))
            idxx = np.random.choice(rep_pts.size()[1], self.settings[0][1].get('P'), replace = False).tolist() ## select representative points
            rep_pts2 = rep_pts[:,idxx,:]
        else:
            rep_pts2 = rep_pts
        fts = self.dense1(out)
        pts_idx = knn_indices_func_cpu(rep_pts2,rep_pts,self.settings[0][1].get('K') ,self.settings[0][1].get('D'))
        #pts_idx = pts_idx[:,:,::self.settings[0][1].get('D')]
        pts_regional = torch.stack([rep_pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        fts_regional = torch.stack([  fts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        out2 = self.xvonc2(rep_pts2,pts_regional,fts_regional)
        layer_pts.append(rep_pts2)
        outs.append(out2)
        pts_regionals.append(pts_regional)
        fts_regionals.append(fts_regional)

        #print("THIRD CONVOLUTION")
        if(not (self.settings[0][2].get('P')==-1)):
            #print("been there" + str(self.settings[0][1].get('P')))
            idxx = np.random.choice(rep_pts2.size()[1], self.settings[0][2].get('P'), replace = False).tolist() ## select representative points
            rep_pts3 = rep_pts2[:,idxx,:]
        else:
            rep_pts3 = rep_pts2
        fts = self.dense2(out2)
        pts_idx = knn_indices_func_cpu(rep_pts3,rep_pts2,self.settings[0][2].get('K')  ,self.settings[0][2].get('D'))
        #pts_idx = pts_idx[:,:,::self.settings[0][2].get('D')]
        pts_regional = torch.stack([rep_pts2[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        fts_regional = torch.stack([  fts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        out3 = self.xvonc3(rep_pts3,pts_regional,fts_regional)
        layer_pts.append(rep_pts3)
        outs.append(out3)
        pts_regionals.append(pts_regional)
        fts_regionals.append(fts_regional)

        #print("FOURTH CONVOLUTION")
        if(not (self.settings[0][3].get('P')==-1)):
            # print("been there" + str(self.settings[0][1].get('P')))
            idxx = np.random.choice(rep_pts3.size()[1], self.settings[0][3].get('P'), replace = False).tolist() ## select representative points
            rep_pts4 = rep_pts3[:,idxx,:]
        else:
            rep_pts4 = rep_pts3
        fts = self.dense3(out3)
        #print("dimensions rep pts  :  " +str(rep_pts4.shape)  +" :  " + str(rep_pts3.shape))
        #print("inputs  " + str(rep_pts4.shape) + "  :   " + str(rep_pts3.shape))
        pts_idx = knn_indices_func_cpu(rep_pts4,rep_pts3,self.settings[0][3].get('K'),self.settings[0][3].get('D'))
        #pts_idx = pts_idx[:,:,::self.settings[0][3].get('D')]
        pts_regional = torch.stack([rep_pts3[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        fts_regional = torch.stack([  fts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
        out4 = self.xvonc4(rep_pts4,pts_regional,fts_regional)
        layer_pts.append(rep_pts4)
        outs.append(out4)
        pts_regionals.append(pts_regional)
        fts_regionals.append(fts_regional)

        ############################END CONVOLUTION, START DECONVOLUTIONS ####################

        for i in range(0,len(self.deconvolutions)):

            #print("DECONVOLUTION " + str(i))
            this_out = outs[self.settings[1][i].get('pts_layer_idx')+1] if i == 0 else outs[-1]
            rep = layer_pts[self.settings[1][i].get('qrs_layer_idx')+1]
            rep2 = layer_pts[self.settings[1][i].get('pts_layer_idx')+1]
            #print("dimensions rep pts  :  " +str(rep.shape) + "   :   "+ str(rep2.shape) )
            pts_idx = knn_indices_func_cpu(rep,
                                        rep2,
                                        self.settings[1][i].get('K'),
                                        self.settings[1][i].get('D'))
                                        #pts_idx = pts_idx[:,:,::self.settings[0][3].get('D')]
            pts_regional = torch.stack([rep2[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
            this_out = torch.stack([  this_out[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))], dim = 0)
            #print("features in : " + str( this_out.shape))
            out = self.deconvolutions[i](rep, pts_regional,this_out)

            out = torch.cat((out,outs[self.settings[1][i].get('qrs_layer_idx')+1]),-1)
            #print("OUT CONTATENATED   :  " + str(out.shape))
            densed =  self.dense_deconv[i](out)
            #print("DENSED : "+ str(densed.shape))
            outs.append( densed  )

    ############################END DECONVOLUTIONS, START Fully connected ####################
        #print("FULLY_CONNECTED")
        output = outs[-1]
        for i in range(0,len(self.fc)):
            output = self.fc[i](output)

        return  output # self.sigmoid(output)


if __name__ == "__main__":
    x = 8
    xyz = torch.rand(4, 2048,3).cuda()
    xconv_param_name = ('K', 'D', 'P', 'C')
    xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 256),
                 (12, 2, 768, 256),
                 (16, 2, 384, 512),
                 (16, 4, 128, 1024)]]

    xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
    xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                [(16, 4, 3, 3),
                  (16, 2, 3, 2),
                  (12, 2, 2, 1),
                  (8, 2, 1, 0)]]

    fc_param_name = ('C', 'dropout_rate')
    fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.0),
              (32 * x, 0.5),
              (3,0.5)]]

    model = PointCnnLayer(xyz, ["features"],[ xconv_params,xdconv_params,fc_params ]).cuda()
    out = model(xyz)
    print(out.shape)
