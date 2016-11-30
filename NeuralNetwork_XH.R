################################################################
#--------------------------------------------------------------#
####   Functions to initialize and train a neural network   ####
####      Last updated by Xiao Huang on 11/29/2016          ####
#--------------------------------------------------------------#
################################################################



#################################################################
#----------------------------------------------------------------
Create.Network = function(sizes) {
  
  # Function to create initial network using standard Gaussian
  # Input: 
  # -- sizes: a vector containing number of neurons in each layer
  # Output: a list containing:
  # -- N.layers: number of layers
  # -- sizes: same as the input
  # -- biases: the biase vectors in each layer
  # -- weights: the weight matrices of each consecutive couple of layers
  
  N.layers = length(sizes)
  biases = list()
  weights = list()
  
  for (j in 2:N.layers) {
    n2 = sizes[j]
    n1 = sizes[j-1]
    bias = matrix(rnorm(n2),nrow=n2)
    biases[[j-1]] = bias
    weight = list(matrix(rnorm(n2*n1),nrow=n2,ncol=n1))
    weights = c(weights,weight)
  }
  
  network = list(N.layers=N.layers, sizes=sizes, weights=weights, biases=biases)
  return(network)
}
#--------------------------------------------------------------------
#####################################################################



#####################################################################
#--------------------------------------------------------------------
Sigmoid = function(z) {
  
  # The Sigmoid activation function
  a = 1/(1+exp(-z))
  return(a)
}

Sigmoid.Prime = function(z) {
  
  # The derivative of Sigmoid activation function
  a = Sigmoid(z)*(1-Sigmoid(z))
  return(a)
}
#--------------------------------------------------------------------
######################################################################


#####################################################################
#--------------------------------------------------------------------
Feed.Forward = function(network,x){
  
  # Function to perform the forward propagation
  # Inputs:
  # -- network: a list containing the network information
  # -- x: the input signal vector, must agree with the order of the first layer
  # Output: a list containing
  # -- output: the output signal in the last layer
  # -- activations: the output at all layers in the network, including the inputs
  # -- zs: inactivated outputs at each layer
  
  biases = network$biases
  weights = network$weights
  n = network$N.layers-1
  activations = list(x)
  zs = list(x)
  
  for (j in 1:n){
    w = weights[[j]]
    b = biases[[j]]
    z = apply(w,1,crossprod,x) + b
    x = Sigmoid(z)
    activations[[j+1]] = x
    zs[[j+1]] = z
  }
  
  output = x
  result = list(output=output,activations=activations,zs=zs)
  return(result)
  
}
#---------------------------------------------------------------------------
############################################################################




############################################################################
#---------------------------------------------------------------------------
Error.Compute = function(d,y){
  # The function to compute the output error of network
  # which is the derivative of the square cost function to the output
  # e = d - y
  # digit = rep(0,10)
  # digit[which.max(d)] = 1
  # e = digit-y
  # return(sum(abs(e)))
  e = d-y
  return(e)
}

Gradient.Initial = function(biases,weights){
  # The function to return lists of zeros matrices
  # that have the same sizes with biases and weights
  # This is used for the initialization of Back.Propagation
  grad.b = list()
  grad.w = list()
  n = length(biases)
  for (j in 1:n){
    bias = biases[[j]]
    nb = nrow(bias)
    b = matrix(rep(0,nb),nrow=nb)
    grad.b[[j]] = b
    weight = weights[[j]]
    nw.row = dim(weight)[1]
    nw.col = dim(weight)[2]
    w = matrix(rep(0,nw.row*nw.col),nrow=nw.row,ncol=nw.col)
    grad.w[[j]] = w
  }
  result = list(grad.b=grad.b,grad.w=grad.w)
  return(result)
}

Back.Propagation = function(network,x,y){
  # The function to perform the back propagation computation
  # Inputs: 
  # -- network: the current neural network
  # -- x: the input data
  # -- y: the output data
  # Outputs: a list containing
  # -- grad.b: the gradients with respect to the biases b
  # -- grad.w: the gradients with respect to the weights w
  
  # Initialization
  biases = network$biases
  weights = network$weights
  result = Gradient.Initial(biases,weights)
  grad.b = result$grad.b
  grad.w = result$grad.w
  
  # Forward feed
  result = Feed.Forward(network,x)
  activations = result$activations
  zs = result$activations
  d = result$output
  
  # Backward pass: the last layer
  L = network$N.layers
  delta = Error.Compute(d,y)*Sigmoid.Prime(zs[[L]]) # elementwise product for vector output
  delta = matrix(delta,nrow=length(delta)) # as a column vector
  a = matrix(activations[[L-1]],ncol=length(activations[[L-1]])) # as a row vector
  grad.b[[L-1]] = delta
  grad.w[[L-1]] = delta%*%a
  
  # Backward pass: previous layers
  for (l in 1:(L-2)) {
    z = zs[[L-l]]
    sp = Sigmoid.Prime(z)
    sp = matrix(sp,nrow=length(sp))
    delta = t(weights[[L-l]])%*%delta
    delta = delta*sp
    grad.b[[L-l-1]] = delta
    grad.w[[L-l-1]] = delta%*%t(activations[[L-l-1]])
  }
  
  result = list(grad.w=grad.w,grad.b=grad.b)
  return(result)
  
}
#====================================================================================
#####################################################################################



#####################################################################################
#====================================================================================
Network.Training = function(network,X,Y,eta){
  # Function to train the network using back propagation
  # Inputs:
  # -- network: the current neural network before training
  # -- X,Y: training data batch
  # -- eta: learning rate
  # Outputs:
  # -- network: the new network with updated weights and biases
  
  N = nrow(X) # X and Y must have the same number of row
  
  biases = network$biases
  weights = network$weights
  L = length(weights)
  result = Gradient.Initial(biases,weights)
  delta.b = result$grad.b
  delta.w = result$grad.w
  
  
  for (j in 1:N){
    
    x = X[j,]
    y = if(is.null(dim(Y))) Y[j] else Y[j,]
    result = Back.Propagation(network,x,y)
    for (l in 1:L){
      delta.b[[l]] = delta.b[[l]] + result$grad.b[[l]]
      delta.w[[l]] = delta.w[[l]] + result$grad.w[[l]]
    }
  }
    
    for (l in 1:L){
      weights[[l]] = weights[[l]] - (eta/N)*delta.w[[l]]
      biases[[l]] = biases[[l]] - (eta/N)*delta.b[[l]]/N
    }
    network$weights = weights
    network$biases = biases
  
  return(network)
  
}
#========================================================================
#########################################################################



#########################################################################
#========================================================================
Network.Build = function(sizes,Xs,Ys,N.epochs,N.batch,eta){
  # Function to build neural network with training data
  # Inputs:
  # -- sizes: same as function Create.Network
  # -- X,Y: training data
  # -- N.epochs: number of epochs provided to train the network
  # -- N.batch: the batch size
  # -- eta: the learning rate
  # Outputs:
  # -- the network learned
  
  N.size = nrow(Xs) # training sample size
  network = Create.Network(sizes) # Create initial network
  
  k = 0
  batch.IDs = c(1)
  while (k<(N.size-N.batch)){
    k = k + N.batch
    batch.IDs = c(batch.IDs,k)
  }
  
  for (j in 1:N.epochs){
    IDs = sample(N.size) # random shuffle
    print(paste("epoch",j))
    for (k in batch.IDs) {
      # Get training data batch
      ID = IDs[(k+1):(k+N.batch)]
      X = Xs[ID,] 
      Y = if(is.null(dim(Ys))) Ys[ID] else Ys[ID,]
      network = Network.Training(network,X,Y,eta) # network training
      print(paste("batch",k/N.batch))
    }
  }
  
  return(network)
  
}
#=====================================================================
######################################################################





