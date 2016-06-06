# coding=utf-8
import csv
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot,pause,axis, xlabel,ylabel,title,grid,legend
import numpy as np
import time
start_time = time.time()
LIST_ARG=1000
ITERATIONS=200

def getPseudoId(nom):
    for i in range(len(pseudo)):
        if(pseudo[i]==nom):
            return i

    return -1
def fromManyGetId(liste):
    for i in range(len(liste)):
        if(liste[i]==1):
            return i
    return -1
def ConvertToOneOfMany(d,nb_classes,bounds=(0,1)):
  d2 = ClassificationDataSet(d.indim, d.outdim, nb_classes=nb_classes)
  for n in range(d.getLength()):
    d2.addSample( d.getSample(n)[0], d.getSample(n)[1] )
  oldtarg=d.getField('target')
  newtarg=np.zeros([len(d),nb_classes],dtype='Int32')+bounds[0]
  for i in range(len(d)):
    newtarg[i,int(oldtarg[i])]=bounds[1]
  d2.setField('class',oldtarg)
  d2.setField('target',newtarg)
  return(d2)




file= open('Amazon_initial_50_30_10000 2.csv', 'rb');
tab=csv.reader(file);
liste=list(tab);

pseudo=set([])
for row in liste:
    pseudo.add(row[-1])
pseudo=sorted(list(pseudo))
print "number of author "+str(len(pseudo))
print "number of lines "+str(len(liste))
print "number of parameters "+str(len(liste[0])-1)
#print len(pseudo)
liste_finale=[]
for row in liste:
    elem=[]
    elem.append(getPseudoId(row[-1]))
    elem.append(map(int, row[:LIST_ARG])) # convertion en liste d'entiers
    liste_finale.append(elem)




#on a maintenant une liste d'elements de structure suivante [id_pseudo,[liste d'argument d'un commentaire de ce pseudo]]

alldata = ClassificationDataSet(LIST_ARG, 1, nb_classes=50)
for n in liste_finale:
    alldata.addSample(n[1],n[0])

#Randomly split the dataset into 75% training and 25% test data sets. Of course, we could also have created two different datasets to begin with.

#correcting a bug in the module
tstdata, trndata = alldata.splitWithProportion(0.25)


#
# For neural network classification, it is highly advisable to encode classes with one output neuron per class. Note that this operation duplicates the original targets and stores them in an (integer) field named ‘class’.
trndata=ConvertToOneOfMany(trndata,50)
tstdata=ConvertToOneOfMany(tstdata,50)

# Now build a feed-forward network with 5 hidden units. We use the shortcut buildNetwork() for this. The input and output layer size must match the dataset’s input and target dimension. You could add additional hidden layers by inserting more numbers giving the desired layer sizes.
# The output layer uses a softmax function because we are doing classification. There are more options to explore here, e.g. try changing the hidden layer transfer function to linear instead of (the default) sigmoid.
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], pseudo[fromManyGetId(trndata['target'][0])], trndata['class'][0]

#size of the hidden layer is the sum of the input, LIST_ARG and the output 50 (nombre d'auteurs)
fnn = buildNetwork( trndata.indim, LIST_ARG+50+1, trndata.outdim, outclass=SoftmaxLayer )
#Set up a trainer that basically takes the network and training dataset as input. For a list of trainers, see trainers. We are using a BackpropTrainer for this.


trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)





epoch=[]
train=[]
test=[]
figure(1)
ylabel('Error (%)')
xlabel('Iterations')
title('Neural Network Classification with '+str(LIST_ARG)+' arguments')
axis([0, ITERATIONS+1, 0, 100])
grid(True)
ion()
time_last=time.time()
for i in range(ITERATIONS):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(), trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
    dataset=tstdata), tstdata['class'])
    print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
    print("compute in "+str((time.time()-time_last)))
    print "-----------\r\n"
    time_last=time.time()
    epoch.append(trainer.totalepochs)
    train.append(trnresult)
    test.append(tstresult)

    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    ylabel('Error (%)')
    xlabel('Iterations')
    title('Neural Network Classification with ' + str(LIST_ARG) + ' arguments')
    axis([0, ITERATIONS, 0, 100])

    grid(True)
    plot(epoch,train,'o',label='training data')
    plot(epoch, test, '+',label='test data')
    legend(bbox_to_anchor=(0.6, 0.85, 0.402, .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
    ion()   # interactive graphics on
    draw()  # update the plot
    show()
    pause(0.0001)
ioff()
import time
print("--- %s seconds ---" % (time.time() - start_time))
show()