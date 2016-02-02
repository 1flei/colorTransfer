import random
from sklearn.utils.linear_assignment_ import linear_assignment
from PIL import Image
import numpy as np
import numpy.linalg as la
import scipy.interpolate as ip
import colorsys
from PIL import ImageFilter



def getDistMat(a,b):
    res = np.zeros((len(a),len(b)))
    for i in xrange(len(a)):
        for j in xrange(len(b)):
            res[i,j] = la.norm(a[i]-b[j],ord=2)**2
    return res

if __name__ == '__main__':
    refImg = Image.open('../res/colorTransfer/20160107200222.jpg')
    tgtImg = Image.open('../res/colorTransfer/20160107200218.jpg')
    refSize,tgtSize = refImg.size,tgtImg.size
    
    refPix = refImg.load()
    tgtPix = tgtImg.load()  
    
    refArray = np.array(refImg)
    tgtArray = np.array(tgtImg)
    print refArray.shape,tgtArray.shape
    
    refVec = np.reshape(refArray,(refArray.shape[0]*refArray.shape[1],3))
    tgtVec = np.reshape(tgtArray,(tgtArray.shape[0]*tgtArray.shape[1],3))
#     print refArray,tgtArray
    
    ver = np.array([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],[255,0,255],[255,255,0],[255,255,255]])
    
    numberOfSample = 600
    k = 1
    
    resArray = np.zeros((k,tgtVec.shape[0],tgtVec.shape[1]),'uint8')
    
    for i in xrange(k):
        print '%dth iterations'%(i+1)
        refSample = np.array(random.sample(refVec, numberOfSample))
        tgtSample = np.append(np.array(random.sample(tgtVec, numberOfSample-len(ver))),ver,axis=0)
        
#         print 'ref\n',refSample
#         print 'tgt\n',tgtSample
        distMat = getDistMat(tgtSample,refSample)
        assignment = np.array(linear_assignment(distMat))
#         print 'assignment\n',assignment
#         print np.append(tgtSample,refSample[assignment[:,1]],axis=1)
#         print 'aref\n',refSample[assignment[:,1]]
#         print 'atgt\n',tgtSample[assignment[:,0]]
        points = tgtSample
        values = refSample[assignment[:,1]]
#         print points
#         print values
        
        h0 = ip.LinearNDInterpolator(points=points,values=values[:,0])
        h1 = ip.LinearNDInterpolator(points=points,values=values[:,1])
        h2 = ip.LinearNDInterpolator(points=points,values=values[:,2])
        resArray[i,:,0] = h0(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])+0.499
        resArray[i,:,1] = h1(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])+0.499
        resArray[i,:,2] = h2(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])+0.499
            
    outpArray = np.zeros(tgtVec.shape,'uint8')
#     outpArray[:,0] = h0(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])
#     outpArray[:,1] = h1(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])
#     outpArray[:,2] = h2(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])
    np.median(resArray, axis=0, out=outpArray)
    print outpArray.shape
    print outpArray
    
    outpImg = Image.fromarray(outpArray.reshape(tgtArray.shape))
    outpImg.save('out.jpg')