


training.data = read.csv("HDR.csv")

source("NeuralNetwork_XH.R")
digits = training.data[,1]
Xs = training.data[,2:785]
Xs = as.matrix(Xs)

digit.recog = function(digits){
  N = length(digits)
  Y = matrix(rep(0,10*N),nrow=N,ncol=10)
  for (j in 1:N){
    digit = digits[j]
    Y[j,digit+1] = 1
  }
  return(Y)
}

Ys = digit.recog(digits)
Ys = as.matrix(Ys)

N.size = 10000
Xs.1 = Xs[1:N.size,]
Ys.1 = Ys[1:N.size,]

sizes = c(784,30,10)
N.epochs = 1
N.batch = 100
eta = 10
network = Network.Build(sizes,Xs.1,Ys.1,N.epochs,N.batch,eta)
network$weights[[2]]

N.test = 100
n.test = sample(10001:20000,N.test)
# x = Xs[n.test,]
d = digits[n.test]

pred.digits = c()
for (j in 1:N.test){
  n = n.test[j]
  x = Xs[n,]
  z = Feed.Forward(network,x)$zs[[3]]
  p.d = which.max(Sigmoid(z)) - 1
  pred.digits = c(pred.digits,p.d)
}

sum(pred.digits==d)/N.test # prediction accuracy



