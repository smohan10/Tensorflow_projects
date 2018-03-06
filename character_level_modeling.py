"""
NAME: CHARACTER LEVEL MODELING
AUTHOR: SANDEEP MOHAN
DATE: 02/22/2018
"""

'''
This is an example of Sequence to Sequence Modeling
at the character level. 
The code reads the input data file, 
computes forward and back propogration 
and generates new data based on the model it has learnt.
'''


# Import the libraries
import tensorflow as tf
import numpy as np 
import random
import argparse
import os
import sys
import shutil
import logging
import datetime as dt
import time


# Define the class
class CharacterLevelModeling(object):
    
    # Initialization of the class
    def __init__(self, args):
        
        self.logger = self.create_logger_instance()
        
        self.train_file = args.train_data
        
        # Parse the training data
        self.read_training_file()
        
        
    def get_current_date_string(self):
        '''
        Method that returns a string for current date information
        Args     : None
        Returns  : Date and time string
        '''
        current_date_object = dt.date.today()
        current_date_str = str(current_date_object.year) + \
                                    str(current_date_object.month) \
                                        + str(current_date_object.day) + '_' + str(int(time.time()))
            
        return current_date_str
    
    
    def print_banner(self,message):
        ''' 
        Method that prints in style in log file
        Args     : Message to display
        Return   : None
        '''
        
        p = '\n'
        p+= '===================================\n'
        p+= message + '\n'
        p+= '===================================\n'
        self.logger.debug(p)
        
    
    def create_directory(self,path):
        """ 
        linux "mkdir -p" command but don't throw error if it already exists 
        """
        
        if os.path.isdir(path): return
        
        os.makedirs(path)
        
        
    def create_logger_instance(self):  
        '''
        Creating an instance for file level and console level logging
        '''
        
        logger = logging.getLogger('CHARACTER_LEVEL_MODELING')
        logger.setLevel(logging.DEBUG)
        
        logs_folder = 'logs'
        self.create_directory(logs_folder)

        logging_name = 'logs/character_modeling_rnn_' + self.get_current_date_string() +  '.log'
                    
        # create file handler which logs debug messages 
        fh = logging.FileHandler(logging_name)
        fh.setLevel(logging.DEBUG)
        
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)   
    
        return logger

    
    def read_training_file(self):
        """
        Method that reads the training data, 
        builds the map between characters and indices and
        vice-versa.         
        
        Args   : None
        Returns: None
        """
        
        self.print_banner("READ TRAINING FILE")
        
        # Prepare the complete path
        file_to_read = os.getcwd() + "\\" + self.train_file
    
        # Check if the file exists
        if not os.path.isfile(file_to_read):
            self.logger.error("FILE NOT FOUND...EXITING")
            sys.exit(1)

        # Open the file and read as characters
        with open(file_to_read, "r") as f:
            self.data = f.read().lower()     
            
            # Build a list of unique characters 
            self.chars = list(set(self.data))
            self.vocab_size = len(self.chars)

        # Open the file again, this time read the lines into a list as words itself
        with open(file_to_read, "r") as f:
            self.examples = f.readlines()            
            self.examples = [ex.lower().strip() for ex in self.examples]
            
        
        self.logger.debug("The number of characters in the input file is %d" %len(self.data))
        self.logger.debug("The number of unique characters in the input file is %d" % self.vocab_size)
        self.logger.debug("Number of examples is %d" % len(self.examples))
        
        # Hold the mapping between indices and characters

        self.char_to_idx = {}
        self.idx_to_char = {}
    
        self.char_to_idx = {  val : idx for idx, val in enumerate(sorted(self.chars)) }
        self.idx_to_char = {  idx : val for idx, val in enumerate(sorted(self.chars)) }
        
        
    def get_initial_loss(self, vocab_size, seq_length):
        """
        Method that computes and returns initial loss
        
        Args    : vocabulary size, sequence length(Number of sequences to produce)
        Returns : Initial loss 
        """
        self.print_banner("GET INITIAL LOSS")
        
        
        init_loss = -np.log(1.0/vocab_size)*seq_length
        self.logger.debug("Initial loss computed as: %f" % init_loss)
        
        return init_loss 
    
    
    def smooth(self, loss, cur_loss):
        """
        Method that smoothens the loss and returns the value
        
        Args    : Overall loss, current loss
        Returns : Smoothened loss
        """
        
        smooth_loss = loss * 0.999 + cur_loss * 0.001
        
        return smooth_loss


    def initialize_params(self, na, nx, ny):      
        """
        Initializes the weights and the biases 
        Args    : na (units for a), nx (feature dim of x), ny(feature dim of y)
        Returns : Dictionary of weights and biases
        """
        self.print_banner("INITIALIZE PARAMETERS")
        
        np.random.seed(1)
        
        parameters = {}

        Waa = np.random.randn(na, na) * 0.01
        Wax = np.random.randn(na, nx) * 0.01
        Wya = np.random.randn(ny, na) * 0.01
        ba  = np.zeros(shape=(na, 1))
        by  = np.zeros(shape=(ny, 1))

        parameters["Waa"] = Waa
        parameters["Wax"] = Wax
        parameters["Wya"] = Wya
        parameters["ba"] = ba
        parameters["by"] = by
        
        self.logger.debug("Computed the parameters")

        return parameters


    def calculate_softmax(self, z):
        """
        Computes the softmax of the input value
        Args    : Activation Z
        Returns : Softmax applied to the activation
        """
        
        exp = np.exp(z)
        
        softmax = exp / np.sum(exp)
                
        return softmax

    
    def gradient_clipping(self, gradients, maxValue):
        """
        Performs gradient clipping to a max value in case gradient shoots up unexpectedly
        Args    : Gradients Dictionary, Maximum value
        Returns : Updated Gradients Dictionary 
        """
        
        
        dWaa, dWax, dWya, dba, dby = gradients["dWaa"], gradients["dWax"], gradients["dWya"], gradients["dba"], gradients["dby"]

        # In place clipping
        for gradient in [dWaa, dWax, dWya, dba, dby]:
            np.clip(gradient, -maxValue, maxValue, out=gradient)

        gradients = {"dWaa": dWaa, "dWax" : dWax, "dWya" : dWya, "dba" : dba, "dby": dby}

        return gradients
    

    def sampling(self, parameters, seed):
        """
        Generates sampling indices from parameters and seed value
        Args    : Parameters dictionary, seed for random generator
        Returns : List of indices resulting from sampling
        """

        
        Waa, Wax, Wya, ba, by = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["ba"], parameters["by"]

        n_a = Waa.shape[0]
        a_prev = np.zeros(shape=(n_a, 1))
        x = np.zeros(shape=(self.vocab_size, 1))

        indices = []

        idx = -1
        counter = 0 

        new_line_char = self.char_to_idx["\n"]

        # Loop till end of the line character or till the counter value is 50 
        while idx != new_line_char and counter != 50:
            
            # forward propagation
            z = np.dot(Waa, a_prev) + np.dot(Wax, x) + ba
            a_next = np.tanh(z) 

            # Compute the output
            zy = np.dot(Wya, a_next) + by
            y = self.calculate_softmax(zy)
    

            # Given the softmax probabilities, sample a random index
            idx = np.random.choice(a = range(self.vocab_size), p = y.ravel(), replace=True)

            indices.append(idx)

            # Reset x and set the index value to be 1 (one hot encoded vector)
            x = np.zeros(shape=(self.vocab_size, 1))
            x[idx, 0] = 1

            # Set the previous a value to be the current a value
            a_prev = a_next

            # Update the counter and the seed
            counter += 1
            seed += 1


        return indices


    
    def run_forward_step(self, a_prev, xt, parameters):
        """
        Run one step of forward propagation
        Args     : a_prev, current x and parameters dictionary
        Returns  : Computed output yt and a_next
        """

        
        # Activation
        z = np.dot(parameters["Waa"], a_prev) + np.dot(parameters["Wax"], xt) + parameters["ba"]
        
        # Next a value
        a_next = np.tanh(z)
        
        # Compute the softmax to produce y at that time step
        yt = self.calculate_softmax(np.dot(parameters["Wya"], a_next) + parameters["by"])
       
        return yt, a_next


    def rnn_step_backward(self, dy, gradients, parameters, x, a, a_prev):
        """
        Computes one step of back propagation
        Args     : dy, gradients dictionary, parameters dictionary, x, a and a_prev
        Returns  : gradients dictionary
        """
    
        
        gradients['dWya'] += np.dot(dy, a.T)

        gradients['dby'] += dy

        da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h

        daraw = (1 - a * a) * da # backprop through tanh nonlinearity

        gradients['dba'] += daraw

        gradients['dWax'] += np.dot(daraw, x.T)

        gradients['dWaa'] += np.dot(daraw, a_prev.T)

        gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)

        return gradients
    
    

    def run_forward_pass(self, X, Y, a0, parameters, vocab_size = 27):
        '''
        Method that runs forward pass on a sequence of data
        Args     : X, Y, Initial a value, parameters dictionary, vocabulary size
        Returns  : loss,  cache tuple of predicted y, a and x
        '''
    
        
        # Empty dictionaries
        x, a, y_pred = {}, {}, {}

        # Initialize with a0
        a[-1] = np.copy(a0)

        # Set loss to be 0
        loss = 0

        # Loop for the length of X
        for t in range(len(X)):

            # For this time step
            x[t] = np.zeros(shape=(vocab_size,1))

            # One hot encoded vector for that time step of x
            if X[t] != None:
                x[t][X[t]] = 1

            # Run one cell
            y_pred[t], a[t] = self.run_forward_step(a[t-1], x[t], parameters)
            
            #print(y_pred[t], Y[t])
            # Compute the loss and accumulate
            loss -= np.log( y_pred[t][Y[t]] )

        cache = (y_pred, a, x)

        return loss ,cache



    def update_parameters(self, parameters, gradients, lr):
        """
        Method that updates the parameters after computing back prop
        Args     : Parameters dictionary, gradients dictionary, learning rate
        Returns  : Parameters dictionary
        """
        
        
        parameters['Wax'] += -lr * gradients['dWax']

        parameters['Waa'] += -lr * gradients['dWaa']

        parameters['Wya'] += -lr * gradients['dWya']

        parameters['ba']  += -lr * gradients['dba']

        parameters['by']  += -lr * gradients['dby']

        return parameters



    def run_back_prop(self, X, Y, parameters, cache):
        """
        Method that runs back prop on a sequence
        Args     : X, Y, parameters dictionary, cache tuple
        Returns  : gradients dictionary, a
        """
        
        
        # Initialize gradients as an empty dictionary
        gradients = {}

        # Retrieve from cache and parameters
        (y_hat, a, x) = cache

        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']

        # each one should be initialized to zeros of the same dimension as its corresponding parameter
        gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)

        gradients['dba'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)

        gradients['da_next'] = np.zeros_like(a[0])

        # Backpropagate through time

        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])

            dy[Y[t]] -= 1

            gradients = self.rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])


        return gradients, a



    def gradient_descent_optimization(self, X, Y, a_prev,params, vocab_size , learning_rate=0.01 ):
        '''
        Method that performs optimization 
        Args     : X, Y, previous a, parameters dictionary, vocabulary size, learning rate
        Returns  : Loss, gradients dictionary, final a value
        '''
        
        loss, cache = self.run_forward_pass(X, Y, a_prev, params, vocab_size)

        gradients, a = self.run_back_prop(X, Y, params, cache)

        gradients = self.gradient_clipping(gradients, 5.0)

        params = self.update_parameters(params, gradients,learning_rate)

        return loss, gradients, a[len(X)-1]


    def model(self,  num_iterations = 35000, vocab_size = 27, dino_names = 7, n_a = 50):
        '''
        Method that prepares (X,Y) data, run optimize, compute the loss and generate new samples
        Args    : Number of iterations, vocabulary size, generate new samples size and sequence n_a size
        Returns : parameters dictionary
        '''

        self.print_banner("BUILD THE MODEL AND GENERATE NEW SAMPLES")
        
        n_x , n_y = vocab_size, vocab_size
        
        a0 = np.zeros(shape=(n_a, 1))

        parameters = self.initialize_params(n_a, n_x, n_y)

        np.random.seed(0)
        
        # Get the initial loss
        loss = self.get_initial_loss(vocab_size, dino_names)

        # Shuffle the training data
        np.random.shuffle(self.examples)

        
        for j in range(num_iterations):
                
            # Get a starting index
            idx = j % len(self.examples)

            # Prepare X and Y
            X = [None] + [ self.char_to_idx[ch]  for ch in self.examples[idx]]
                        
            Y = X[1:] + [self.char_to_idx["\n"]]
            
            # Compute optimization method
            current_loss, gradients, a = self.gradient_descent_optimization(X, Y, a0, parameters, vocab_size, 0.01 )
            
            # Smoothen the loss
            loss = self.smooth(loss, current_loss)

            # For every 2000 iterations, print the loss and generate new samples 
            if j % 2000 == 0:

                self.logger.debug("Loss for iteration %d is %f" % (j, loss))

                seed = 0
                
                output_list = []

                for name in range(dino_names):

                    sample_idx = self.sampling(parameters, seed)
                    word = "".join(self.idx_to_char[ix] for ix in sample_idx)
                    output_list.append(word[0].upper() + word[1:-1])

                    seed += 1

                self.logger.debug("OUTPUT: %r" % output_list)

        return parameters


        
        
if __name__ == '__main__':

    '''

    read a text file 
    create an object out of the class
    
    '''

    parser = argparse.ArgumentParser(description="Character Level Modeling")
    parser.add_argument('--train_data', default="data/dinos.txt", help="Input the training file for modeling")
    args = parser.parse_args()

    char_level_model = CharacterLevelModeling(args)


    np.random.seed(2)
    _, n_a = 20, 100
    Wax, Waa, Wya = np.random.randn(n_a, char_level_model.vocab_size), np.random.randn(n_a, n_a), np.random.randn(char_level_model.vocab_size, n_a)
    ba, by = np.random.randn(n_a, 1), np.random.randn(char_level_model.vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    

    indices = char_level_model.sampling(parameters, 0)
    print("Sampling:")
    print("list of sampled indices:", indices)
    print("list of sampled characters:", [char_level_model.idx_to_char[i] for i in indices])


    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.random.randn(n_a, 1)
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    ba, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    X = [12,3,5,11,22,3]
    Y = [4,14,11,22,25, 26]

    loss, gradients, a_last = char_level_model.gradient_descent_optimization(X, Y, a_prev, parameters, vocab_size, learning_rate = 0.01)
    print("Loss =", loss)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    print("a_last[4] =", a_last[4])



    parameters = char_level_model.model( num_iterations = 35000, vocab_size = 27, dino_names = 7, n_a = 50)