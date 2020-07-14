
import torch
import numpy as np 
from zipfile import ZipFile
import os
import json
import torch.nn as nn
import torch.nn.functional as F


class EventEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim = 128, pos_dim = 128, hidden = 256, event_num = 8):
        super(EventEmbedding, self).__init__()
        self.pos_dim = pos_dim
        self.event_num = event_num
        self.hidden_dim = hidden
        self.embed_dim = embedding_dim
        # Word embedding
        self.embeddings = nn.Embedding(vocab_size, self.embed_dim)
        # Bi-LSTM
        self.lstm = nn.LSTM(self.embed_dim + self.pos_dim, self.hidden_dim, bidirectional=True, batch_first=True) 
        # SRU (Selective Reading), now just a plain GRU (should be self attention + GRU?)
        self.SRU = nn.GRU(2*self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True) 


    
    def forward(self, X):
        '''
        X: Input (b x event_len x src_len)
        '''
        bs = X.size(0)

        ### Can pack sequences to accelerate training

        pos = torch.FloatTensor(bs, self.event_num, self.pos_dim).uniform_(-0.02, 0.02)# randomly initialize position embedding, b x event_len x pos_dim
        pos = pos.unsqueeze(2) # b x event_len x 1 x pos_dim
        
        embeds = self.embeddings(X) # b x event_len x src_len x embd_dim
        pos_ = pos.expand(bs, self.event_num, embeds.size(2), self.pos_dim) # b x event_len x src_len x pos_dim

        
        lstm_in = torch.cat((embeds, pos_), dim=3).view(-1, embeds.size(2), self.embed_dim + self.pos_dim)
        h_0, c_0 = self.init_hidden(lstm_in.size(0))

        lstm_out, hidden = self.lstm(lstm_in, (h_0,c_0)) # lstm_out: b * event_len x src_len x 2*hidden, hidden: 2 x b*event x hidden each

        _, events_last = self.SRU(lstm_out, hidden[0])  # 2 x b*event x hidden

        events = events_last.view(bs,self.event_num,-1)


        return lstm_out, events, pos


    def init_hidden(self, batch_size):

        return torch.FloatTensor(2,batch_size, self.hidden_dim).uniform_(-0.02, 0.02), torch.FloatTensor(2,batch_size, self.hidden_dim).uniform_(-0.02, 0.02)


#---------------------------------------------


class TimeEventMemory(nn.Module):

    def __init__(self, global_dim = 512, event_num = 8, hidden_dim = 256):

        super(TimeEventMemory,self).__init__()
        self.global_dim = global_dim
        self.event_num = event_num
        self.event_rep_dim = 2 * hidden_dim

        self.linear = nn.Linear(self.global_dim+self.event_rep_dim, 1)

    def forward(self, pos_emb, event_rep):
        '''
        event_rep: Event representation a_i [b x event_len x 2*hidden]
        '''
        v1 = event_rep #b x event_len x 2*hidden
        bs = v1.size(0)

        v2 = torch.FloatTensor(bs, self.event_num, self.global_dim).uniform_(-0.02, 0.02) # b x event_len x global_dim
        concat = torch.cat((v2,v1),dim = 2).view(bs,self.event_num,-1) # b x event_len x global_dim + 2*hidden_dim


        V = F.sigmoid(self.linear(concat)) # b x event_len x 1
        V = V.expand(bs, self.event_num, self.global_dim)

        new_v2 = V * v2 + (1-V) * v1

        pos_emb = pos_emb.squeeze()

        return pos_emb, v1, new_v2


#---------------------------------------------


class SummaryGenerator(nn.Module):

    def __init__(self, vocab_size, hidden_dim = 256, event_num = 8, embedding_dim = 128, pos_dim = 128):
        super(SummaryGenerator, self).__init__()

        self.hidden_dim = hidden_dim
        self.event_num = event_num
        self.embed_dim = embedding_dim
        self.pos_dim = pos_dim
        
        self.embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(2*self.hidden_dim+self.embed_dim, self.hidden_dim,batch_first=True) 

        self.eventattention = WordEventAttention(self.hidden_dim, self.event_num)
        self.memoryattention = TimeMemoryAttention(self.hidden_dim, self.pos_dim)

        self.Vocab = nn.Linear(7*self.hidden_dim,vocab_size,bias=True)

    

    def forward(self, encoder_output, h_prev, decoder_input, context_prev, pos_emb, event_rep, v2):
        '''
        event_rep: event representation [b x event_len x 2*hidden]
        context_prev: [b x 1 x 2*hidden]
        decoder_input: [b]
        h_prev: [b x 1 x hidden]
        '''


        bs = event_rep.size(0)

        y_prev = self.embeddings(decoder_input) # b x emb
        y_prev = y_prev.unsqueeze(1)
        lstm_in = torch.cat((context_prev,y_prev),dim=2) # b x 1 x emb+2*hidden

        lstm_out, h = self.lstm(lstm_in, h_prev) #h: 1 x b x hidden

        h1, h2 = h
        decoder_hidden = torch.cat((h1,h2),dim=2).view(bs,1,-1) #b x 1 x 2*hidden
        context, e = self.eventattention(decoder_hidden, encoder_output, event_rep) # b x 1 x 2*hidden

        h, m1 = self.memoryattention(pos_emb, h, event_rep, v2, h_prev, context) # h: b x 1 x hidden, m1: b x 1 x 2*hidden

        # Output
        out = torch.cat((m1,h,context,e),dim=2).squeeze_() 
        final_dist = self.Vocab(out) 
        final_dist = F.softmax(final_dist,dim=1) # b x vocab_size

        
        return final_dist, h, context


#---------------------------------------------


class WordEventAttention(nn.Module):

    def __init__(self, hidden_dim = 256, event_num = 8):

        super(WordEventAttention, self).__init__()
        self.event_num = event_num
        self.hidden_dim = hidden_dim

        self.W_h1 = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim, bias=False)
        self.W_h2 = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim, bias=False)
        self.W_b = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim, bias=False)
        self.W_d = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim, bias=False)
        self.W_a = nn.Linear(2*self.hidden_dim, 1)
        self.W_c = nn.Linear(2*self.hidden_dim, 1)
        

    def forward(self, decoder_hidden, encoder_output, event_rep):
        '''
        decoder_hidden: h, the previous hidden state at t (i.e. state for t-1) [b, 1, 2*hidden]
        encoder_output: h_j^i, the jth word representation in event x_i [b x event_len x src_len x 2*hidden]
        event_rep: a_i, event representation of event x_i  [b x event_len x 2*hidden]
        '''
        batch_size = decoder_hidden.size(0)

        encoder_output = encoder_output.view(batch_size * self.event_num, -1, 2 * self.hidden_dim) # b * event_len x src_len x 2*hidden

        b_event, src_len, hid = list(encoder_output.size())
        decoder_hidden_word_expanded = decoder_hidden.expand(batch_size,self.event_num * src_len,hid).contiguous()
        decoder_hidden_word_expanded = decoder_hidden_word_expanded.view(b_event,src_len,hid) # b * event_len x src_len x 2*hidden
        decoder_hidden_event_expanded = decoder_hidden.expand(batch_size, self.event_num, hid).contiguous()
        decoder_hidden_event_expanded = decoder_hidden_event_expanded.view(b_event,1,hid) # b * event_len x 1 x 2*hidden

        ### Word level attention weight alpha
        alpha_ = self.W_a(F.tanh(self.W_b(decoder_hidden_word_expanded) + self.W_h1(encoder_output))) # b * event_len x src_len x 1
        alpha_ = alpha_.squeeze(2)

        ## The attention is over all words?
        alpha = alpha_.view(batch_size, -1) # b x event_len * src_len
        alpha = F.softmax(alpha_, dim = 1) # b x event_len * src_len
        alpha = alpha.view(batch_size, -1, src_len) # b x event_len x src_len
 

        ### Event level attention weight beta
        event_rep = event_rep.view(b_event,1,hid) # b*event, 1, 2*hidden
        beta_ = F.tanh(self.W_c(self.W_d(decoder_hidden_event_expanded) + self.W_h2(event_rep))) # b * event_len x 1 x 1
        beta_ = beta_.squeeze().view(batch_size, -1) # b x event_len
        beta = F.softmax(beta_, dim = 1) # b x event_len
        e = torch.bmm(beta.unsqueeze(1),event_rep.view(batch_size,self.event_num,hid)) # b x 1 x 2*hidden
        beta = beta.unsqueeze_(-1).expand(batch_size, -1, src_len) # b x event_len x src_len

        ### Context vector
        gamma = alpha * beta # b x event_len x src_len
        gamma = gamma.view(batch_size,-1).unsqueeze(1) # b x 1 x event_len * src_len
        encoder_output = encoder_output.contiguous().view(batch_size,-1,hid) # b x event_len * src_len x 2*hidden
        context = torch.bmm(gamma, encoder_output) # b x event_len * src_len x 2*hidden
        context = torch.sum(context, dim=1, keepdim=True) # b x 1 x 2*hidden

        return context, e 

        
    
#---------------------------------------------


class TimeMemoryAttention(nn.Module):

    def __init__(self, hidden = 256, pos_dim = 128):
        super(TimeMemoryAttention, self).__init__()
        self.hidden_dim = hidden
        self.pos_dim = pos_dim

        self.linear = nn.Linear(self.pos_dim,self.hidden_dim,bias=False)
        self.fusion = nn.Linear(5*self.hidden_dim,2*self.hidden_dim,bias=False)
        self.fusion2 = nn.Linear(5*self.hidden_dim,self.hidden_dim,bias=False)
        self.linear2 = nn.Linear(2*self.hidden_dim,self.hidden_dim,bias=False)
    
    def forward(self, pos_emb, h, v1, v2, h_prev, context):
        '''
        h: current hidden state dec_h and dec_c
        pos_emb: position embeddings [b x event_len x pos_dim]
        v1: event rep [b x event_len x 2*hidden]
        v2: global rep [b x event_len x global_dim], global_dim = 2*hidden
        h_prev: dec_h and dec_c, [1 x b x hidden] each
        context: # b x 1 x 2*hidden
        '''
        bs = h[0].size(1)

        h = h[0].view(bs,1,self.hidden_dim) # b x 1 x hidden
        h_prev = h_prev[0].view(bs,1,self.hidden_dim)

        pos_ = self.linear(pos_emb) # b x event_len x hidden

        pi = torch.bmm(pos_,h.view(bs,self.hidden_dim,1)).squeeze() # b x event_len
        pi = F.softmax(pi).unsqueeze(1) # b x 1 x event_len

        m1 = torch.matmul(pi,v1) # b x 1 x 2*hidden
        m2 = torch.matmul(pi,v2) # b x 1 x 2*hidden

        g1 = torch.cat((h_prev,context,m1),dim=2) # b x 1 x 5*hidden
        g1 = self.fusion(g1)
        m1 = g1 * m1 # b x 1 x 2*hidden

        g2 = torch.cat((h_prev,context,m2),dim=2) # b x 1 x 5*hidden
        g2 = self.fusion2(g2) # b x 1 x hidden
        m2 = self.linear2(m2) # b x 1 x hidden
        h = g2 * h + (1-g2) * m2 # b x 1 x hidden

        return h, m1
        
    


#---------------------------------------------


## Final model

class MTS(object):

    def __init__(self, vocab_size):
        super(MTS, self).__init__()

        self.Encoder = EventEmbedding(vocab_size)
        self.Memory = TimeEventMemory()
        self.Decoder = SummaryGenerator(vocab_size)
        
        if torch.cuda.is_available():
            self.Encoder = self.Encoder.cuda()
            self.Memory = self.Memory.cuda()
            self.Decoder = self.Decoder.cuda()


#---------------------------------------------

### Decoder initialization

class InitDecoder(nn.Module):

    def __init__(self, hidden = 256, event_num = 8):
        super(InitDecoder, self).__init__()
        self.hidden_dim = hidden
        self.event_num = event_num
        self.lstm_0 = nn.LSTM(self.event_num * 2 * self.hidden_dim,self.hidden_dim, batch_first=True)

    def forward(self, event_rep):
        bs = event_rep.size(0)
        h_c1 = torch.FloatTensor(1, bs, self.hidden_dim).uniform_(-0.02, 0.02) # 1 x b x hidden
        h_c2 = torch.FloatTensor(1, bs, self.hidden_dim).uniform_(-0.02, 0.02) # 1 x b x hidden
        event_concat = event_rep.view(bs,1,-1).contiguous() # b x 1 x event_len * 2 * hidden
        h_0, _ = self.lstm_0(event_concat, (h_c1,h_c2)) # b x 1 x hidden
        h_0 = h_0.view(1, bs, -1) # 1 x b x hidden

        return (h_0, h_0)
