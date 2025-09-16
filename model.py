import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import simple_batch_graphify
from selective_modeling.modules.graph_selective_modeling import Mamba
from graph_adj import dynamic_regional_graph
from utils import contrastive_loss
from adaptive_hypergraph import cross_modal_hypergraph as HGNN

class Model(nn.Module):
    def __init__(self,args,D_a=300,D_v=342,D_t=1024,n_speakers=9,n_classes=7):
        super(Model, self).__init__()
        self.args = args
        D_g=args.dim  
        self.top_k=args.top_k 
        
        self.mamba_t = Mamba(
             d_model=args.dim, 
             d_state=args.d_state,  
             d_conv=args.d_conv,  
             expand=args.expand,  
             args=args
            )

        self.mamba_a = Mamba(
             d_model=args.dim,
             d_state=args.d_state,
             d_conv=args.d_conv,  
             expand=args.expand,  
             args=args
            )

        self.mamba_v = Mamba(
             d_model=args.dim,
             d_state=args.d_state,
             d_conv=args.d_conv,  
             expand=args.expand,  
             args=args
            )

        self.regional_graph = dynamic_regional_graph(window_size=args.window_size_1)

        self.norm_t = nn.LayerNorm(args.dim)
        self.norm_a = nn.LayerNorm(args.dim)
        self.norm_v = nn.LayerNorm(args.dim)

        self.linear_a = nn.Linear(D_a, D_g)
        self.linear_v = nn.Linear(D_v, D_g)
        self.linear_t = nn.Linear(D_t, D_g)
        
        self.dropout_ = nn.Dropout(args.dropout)

        self.graph_model=HGNN(args)
          
        self.smax_fc = nn.Linear(3 * args.dim, n_classes)   

        


    def forward(self,U_t,U_a,U_v,lengths,qmask,e,i,train=False):
        
        [r1, r2, r3, r4] = U_t
        seq_len, _, feature_dim = r1.size()
        U_t = r1 + r2 + r3 + r4

        U_t=self.linear_t(U_t)    
        U_t = nn.LeakyReLU()(U_t)
        U_t = U_t.transpose(0, 1)
        adj_t = self.regional_graph(U_t, lengths, qmask)
        U_t =(self.norm_t(self.mamba_t(U_t,adj_t))).transpose(0, 1)
        U_t = self.dropout_(U_t)  

        U_a = self.linear_a(U_a)
        U_a = nn.LeakyReLU()(U_a)
        U_a = U_a.transpose(0, 1)
        adj_a = self.regional_graph(U_a, lengths, qmask)
        U_a =(self.norm_a(self.mamba_a(U_a,adj_a))).transpose(0, 1)
        U_a = self.dropout_(U_a) 
        
        U_v = self.linear_v(U_v)  
        U_v = nn.LeakyReLU()(U_v)
        U_v = U_v.transpose(0, 1)
        adj_v = self.regional_graph(U_v, lengths, qmask) 
        U_v =(self.norm_v(self.mamba_v(U_v,adj_v))).transpose(0, 1)
        U_v = self.dropout_(U_v)
        
        features_a = simple_batch_graphify(U_a,lengths) 
        features_v = simple_batch_graphify(U_v,lengths) 
        features_t = simple_batch_graphify(U_t,lengths) 
       
        feature, loss_cl = self.graph_model(features_a, features_v, features_t, lengths)
        
        emotions_feat = feature
        emotions_feat = nn.LeakyReLU()(emotions_feat)
       
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        
        return log_prob, loss_cl, emotions_feat