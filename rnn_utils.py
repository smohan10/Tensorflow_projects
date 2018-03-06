import tensorflow as tf
import numpy as np


'''
Basic RNN cell class
Given Waa, Wax, ba, by, xt, a_prev, Wya, by
Dimensions
xt        : nx * m 
a_prev    : na * m 
Waa       : na * na
Wax       : na * nx
Wya       : ny * na
ba        : na * 1
by        : ny * 1
'''


class RNN_Cell(object):
    def __init__(self, m, nx, na, ny):
        
        self.parameters = {}
        self.nx = nx
        self.m = m
        self.na = na
        self.ny = ny
        
        self.initialize_params()
        
    def calculate_sigmoid(self, z):
        sigmoid = 1/(1 + np.exp(-z))
        return sigmoid
    
    def calculate_softmax(self, z):
        exp = np.exp(z)
        softmax = exp / np.sum(exp)
        return softmax
    
    def initialize_params(self):      
        
        
        Waa = np.random.randn(self.na, self.na) * np.sqrt(2/self.na)
        Wax = np.random.randn(self.na, self.nx) * np.sqrt(2/self.na)
        Wya = np.random.randn(self.ny, self.na) * np.sqrt(2/self.na)
        ba  = np.zeros(shape=(self.na, 1))
        by  = np.zeros(shape=(self.ny, 1))
        
        self.parameters["Waa"] = Waa
        self.parameters["Wax"] = Wax
        self.parameters["Wya"] = Wya
        self.parameters["ba"] = ba
        self.parameters["by"] = by
        
        
    def return_params(self):
        
        return self.parameters
    
    
    
    def gradient_clipping(self, gradients, maxValue):
        
        dWaa, dWax, dWya, dba, dby = gradients["dWaa"], gradients["dWax"], gradients["dWya"], gradients["dba"], gradients["dby"]

        for gradient in [dWaa, dWax, dWya, dba, dby]:
            np.clip(gradient, -maxValue, maxValue, out=gradient)

        gradients = {"dWaa": dWaa, "dWax" : dWax, "dWya" : dWya, "dba" : dba, "dby": dby}

        return gradients
        
    
    
    def generate_forward_pass(self, a_prev, xt, parameters):
        
        
        # tanh of Waa * a_prev + Wax * xt + ba 
        
        z = np.dot(parameters["Waa"], a_prev) + np.dot(parameters["Wax"], xt) + parameters["ba"]
        
        a_next = np.tanh(z)
        
        yt = self.calculate_softmax(np.dot(parameters["Wya"], a_next) + parameters["by"])
        
        cache = (a_prev, xt, parameters, a_next)
        
        return yt, cache
    

    
    
    
def rnn_forward_sequence_to_sequence(a_prev, x, one_rnn_cell, Tx):
    
    '''
    Input : X, a_prev, object of one RNN cell and Tx
    Return: y_pred, a_prev, caches
    '''
    
    parameters = one_rnn_cell.return_params()
    
    caches = []
    
    y_pred = np.zeros(shape=(x.shape[0], x.shape[1], Tx))
    
    for t in range(Tx):
        
        xt = x[:,:,t]
        
        y_pred[:,:,t], cache = one_rnn_cell.generate_forward_pass(a_prev, xt, parameters)
        
        a_prev = cache[-1]
        
        caches.append(cache)
        
        
    return y_pred, a_prev, caches
    
    
    
    
# ****************************************************************************************************************************
'''
Basic LSTM cell class
Given Wc, Wf, Wu, Wo, Wy, bc, bf, bu, bo, by, xt, a_prev
Dimensions
xt      : nx * m 
a_prev  : na * m 
concat  : (na + nx) * m
Wc      : na * (na + nx)
Wf      : na * (na + nx)
Wu      : na * (na + nx)
Wo      : na * (na + nx)
Wy      : ny * na
bc      : na * 1
bf      : na * 1
bu      : na * 1
bo      : na * 1
by      : ny * 1
'''    
    
class LSTM_Cell(object):
    def __init__(self, m, nx, na, ny):
        
        self.initialize_params(m, nx, na, ny)
        
    def calculate_sigmoid(self, z):
        sigmoid = 1/(1 + np.exp(-z))
        return sigmoid
    
    def calculate_softmax(self, z):
        exp = np.exp(z)
        softmax = exp / np.sum(exp)
        return softmax
    
    def initialize_params(self, m, nx, na, ny):      
        
        self.parameters = {}
        
        Wc = np.random.randn(na, na+nx) * np.sqrt(2/na)
        Wf = np.random.randn(na, na+nx) * np.sqrt(2/na)
        Wu = np.random.randn(na, na+nx) * np.sqrt(2/na)
        Wo = np.random.randn(na, na+nx) * np.sqrt(2/na)       
        Wy = np.random.randn(ny, na) * np.sqrt(2/na)
        bc  = np.zeros(shape=(na, 1))
        bf  = np.zeros(shape=(na, 1))
        bu  = np.zeros(shape=(na, 1))
        bo  = np.zeros(shape=(na, 1))
        by  = np.zeros(shape=(ny, 1))
        
        self.parameters["Wc"] = Wc
        self.parameters["Wf"] = Wf
        self.parameters["Wu"] = Wu
        self.parameters["Wo"] = Wo
        self.parameters["Wy"] = Wy
        self.parameters["bc"] = bc
        self.parameters["bf"] = bf
        self.parameters["bu"] = bu
        self.parameters["bo"] = bo
        self.parameters["by"] = by
        
        
    def return_params(self):
        
        return self.parameters
    
    
    def generate_forward_pass(self, a_prev, c_prev, xt, parameters):
        
        
        concat_in = np.zeros(shape=(a_prev.shape[0]+xt.shape[0], xt.shape[1]))
        
        concat_in[:na, :] = a_prev
        concat_in[na:, :] = xt
        
        zc = np.dot(parameters["Wc"], concat_in) +  parameters["bc"]
        candidate = np.tanh(zc)
        
        zf = np.dot(parameters["Wf"], concat_in) +  parameters["bf"]
        forget = self.calculate_sigmoid(zf)
        
        zu = np.dot(parameters["Wu"], concat_in) +  parameters["bu"]
        update = self.calculate_sigmoid(zu)
        
        zo = np.dot(parameters["Wo"], concat_in) +  parameters["bo"]
        output = self.calculate_sigmoid(zo)
        
        c_next = forget*c_prev + update*candidate
        
        a_next = output*np.tanh(c_next)
        
        yt = self.calculate_softmax(np.dot(parameters["Wy"], a_next) + parameters["by"])
        
        cache = (a_prev, xt, parameters, a_next, c_prev, c_next, candidate, forget, update, output)
        
        return yt, cache    

    
    
def lstm_forward_sequence_to_sequence(a_prev, c_prev, x, one_lstm_cell, Tx):
    
    '''
    Input : X, a_prev, object of one RNN cell and Tx
    Return: y_pred, a_prev, caches
    '''
    
    parameters = one_lstm_cell.return_params()
    
    caches = []
    
    y_pred = np.zeros(shape=(x.shape[0], x.shape[1], Tx))
    
    for t in range(Tx):
        
        xt = x[:,:,t]
        
        y_pred[:,:,t], cache = one_lstm_cell.generate_forward_pass(a_prev, c_prev, xt, parameters)
        
        a_prev = cache[3]
        
        c_prev = cache[5]
        
        caches.append(cache)
        
        
    return y_pred, a_prev, c_prev, caches




    


    
# ****************************************************************************************************************************        
if __name__ == '__main__':
    
    
    '''
    Testing one cell unit of RNN
    '''
    print("******* RNN ONE CELL ***********")
    
    np.random.seed(0)
    nx = 3
    m = 6
    na = 2
    ny = 3
    
    rnn_time_step = RNN_Cell(m, nx, na, ny) 
    
    parameters = rnn_time_step.return_params()
    
    xt     = np.random.randn(nx,m)
    a_prev = np.random.randn(na, m)      
 
    yt, cache = rnn_time_step.generate_forward_pass(a_prev, xt, parameters)
    
    a_next = cache[-1]
   
    print("Shape of xt     = %d * %d" %(xt.shape[0]    , xt.shape[1]))
    print("Shape of a_prev = %d * %d" %(a_prev.shape[0], a_prev.shape[1]))
    print("Shape of yt     = %d * %d" %(yt.shape[0]    , yt.shape[1]))
    print("Shape of a_next = %d * %d" %(a_next.shape[0], a_next.shape[1]))
    
    '''
    Test gradient clipping
    '''
    
    print("******* RNN GRADIENT CLIPPING ***********")
    np.random.seed(3)
    dWax = np.random.randn(5,3)*10
    dWaa = np.random.randn(5,5)*10
    dWya = np.random.randn(2,5)*10
    dba = np.random.randn(5,1)*10
    dby = np.random.randn(2,1)*10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "dba": dba, "dby": dby}
    gradients = rnn_time_step.gradient_clipping(gradients, 10)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])




    '''
    Testing the full forward pass of RNN for input length Tx and output length Ty (Tx = Ty in this case)
    '''
    print("******* RNN LONG SEQUENCE ***********")
    
    Tx     = 15
    x      = np.random.randn(nx,m, Tx)
    a_prev = np.random.randn(na,m)
    
    one_rnn_cell = RNN_Cell(m, nx, na, ny)
    
    y_pred, a, caches = rnn_forward_sequence_to_sequence(a_prev, x, one_rnn_cell, Tx)
    
    
    print("Shape of x      = %d * %d * %d" %(x.shape[0], x.shape[1], x.shape[2]))
    print("Shape of a_prev = %d * %d"      %(a_prev.shape[0], a_prev.shape[1]))
    print("Shape of y_pred = %d * %d * %d" %(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
    
    
    
    '''
    Testing one cell unit of LSTM
    '''
    print("******* LSTM ONE CELL ***********")
    
    np.random.seed(0)
    nx = 3
    m = 6
    na = 2
    ny = 3
    
    lstm_time_step = LSTM_Cell(m, nx, na, ny) 
    
    parameters = lstm_time_step.return_params()
    
    xt     = np.random.randn(nx,m)
    a_prev = np.random.randn(na,m)  
    c_prev = np.random.randn(na,m)
 
    yt, cache = lstm_time_step.generate_forward_pass(a_prev, c_prev, xt, parameters)
    
    a_next, c_next = cache[3],cache[5]
   
    print("Shape of xt     = %d * %d" %(xt.shape[0]    , xt.shape[1]))
    print("Shape of a_prev = %d * %d" %(a_prev.shape[0], a_prev.shape[1]))
    print("Shape of c_prev = %d * %d" %(c_prev.shape[0], c_prev.shape[1]))
    print("Shape of yt     = %d * %d" %(yt.shape[0]    , yt.shape[1]))
    print("Shape of a_next = %d * %d" %(a_next.shape[0], a_next.shape[1]))
    print("Shape of c_next = %d * %d" %(c_next.shape[0], c_next.shape[1]))
    
    
    
    '''
    Testing the full forward pass of LSTM for input length Tx and output length Ty (Tx = Ty in this case)
    '''
    print("******* LSTM LONG SEQUENCE ***********")
    Tx     = 15
    x      = np.random.randn(nx,m, Tx)
    a_prev = np.random.randn(na,m)
    c_prev = np.random.randn(na,m)
    
    one_lstm_cell = LSTM_Cell(m, nx, na, ny)
    
    y_pred, a, c, caches = lstm_forward_sequence_to_sequence(a_prev, c_prev, x, one_lstm_cell, Tx)
    
    
    print("Shape of x      = %d * %d * %d" %(x.shape[0], x.shape[1], x.shape[2]))
    print("Shape of a_prev = %d * %d"      %(a_prev.shape[0], a_prev.shape[1]))
    print("Shape of c_prev = %d * %d"      %(c_prev.shape[0], c_prev.shape[1]))
    print("Shape of a_next = %d * %d"      %(a_next.shape[0], a_next.shape[1]))
    print("Shape of c_next = %d * %d"      %(c_next.shape[0], c_next.shape[1]))
    print("Shape of y_pred = %d * %d * %d" %(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
    
    
# ****************************************************************************************************************************    

   