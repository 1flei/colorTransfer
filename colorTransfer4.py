import random
from sklearn.utils.linear_assignment_ import linear_assignment
from PIL import Image
import numpy as np
import numpy.linalg as la
import scipy.interpolate as ip
from PIL import ImageFilter
import time
from scipy.ndimage import filters
from scipy.ndimage import measurements, morphology
from numpy.random import multivariate_normal

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def getDistMat(a,b):
    res = np.zeros((len(a),len(b)))
    for i in xrange(len(a)):
        for j in xrange(len(b)):
            res[i,j] = la.norm(a[i]-b[j],ord=2)**2
    return res
    
    
def reduceEdge(t,o,threshold=3,out=None):
#     baseKernel = np.ones((3,3,3))
#     print 'base\n',baseKernel
#     cannyKernel = baseKernel*np.array([[1,1,1],[1,-8,1],[1,1,1]])/8.
#     blurKernel = baseKernel*np.array([[1,1,1],[1,1,1],[1,1,1]])/9.
    cannyKernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])/8.
    blurKernel = np.ones((7,7))/49.
#     print 'canny\n',cannyKernel
#     print 'blur\n',blurKernel
    oe = np.zeros(o.shape)
    te = np.zeros(t.shape)
    oe[:,:,0] = filters.convolve(o[:,:,0], cannyKernel, mode='nearest')
    oe[:,:,1] = filters.convolve(o[:,:,1], cannyKernel, mode='nearest')
    oe[:,:,2] = filters.convolve(o[:,:,2], cannyKernel, mode='nearest')
#     oev = np.sum(oe*oe, axis=2)
    oev = la.norm(oe,axis=2,ord=2)
    te[:,:,0] = filters.convolve(t[:,:,0], cannyKernel, mode='nearest')
    te[:,:,1] = filters.convolve(t[:,:,1], cannyKernel, mode='nearest')
    te[:,:,2] = filters.convolve(t[:,:,2], cannyKernel, mode='nearest')
#     tev = np.sum(te*te, axis=2)
    tev = la.norm(te,axis=2,ord=2)
#     ee = np.logical_and(oev>threshold*threshold,tev<threshold*threshold)
    
    ee = oev>tev+threshold
#     print o[ee]
    out = o.copy()
    out[ee][:,0] = filters.convolve(o[:,:,0], blurKernel, mode='nearest')[ee]
    out[ee][:,1] = filters.convolve(o[:,:,1], blurKernel, mode='nearest')[ee]
    out[ee][:,2] = filters.convolve(o[:,:,2], blurKernel, mode='nearest')[ee]
#     out[:,:,0] = filters.convolve(o[:,:,0], blurKernel, mode='nearest')
#     out[:,:,1] = filters.convolve(o[:,:,1], blurKernel, mode='nearest')
#     out[:,:,2] = filters.convolve(o[:,:,2], blurKernel, mode='nearest')
    
    Image.fromarray(oev.astype('uint8')).save('oe.jpg')
    Image.fromarray(tev.astype('uint8')).save('te.jpg')
#     Image.fromarray(o[ee]).save('te.jpg')
    Image.fromarray(ee.astype('uint8')*255).save('ee.jpg')

#     np.savetxt('oe.txt',oev,fmt='%.2f', delimiter='\t')
#     np.savetxt('te.txt',tev,fmt='%.2f', delimiter='\t')
#     np.savetxt('ee.txt',oev-tev,fmt='%.2f', delimiter='\t')
    
#     print oe,te,ee
    
    return out


if __name__ == '__main__':
    random.seed(time.time())
    refImg = Image.open('../res/colorTransfer/paper_sea_day.png')
    tgtImg = Image.open('../res/colorTransfer/paper_sea_night.png')
    
    numberOfSamples = 600
    iterationTimes = 1
    alpha = 400.
    
    
    refSize,tgtSize = refImg.size,tgtImg.size
    
    refPix = refImg.load()
    tgtPix = tgtImg.load()  
    
    refArray = np.array(refImg)
    tgtArray = np.array(tgtImg)
    print refArray.shape,tgtArray.shape
    
    refVec0 = np.zeros((refArray.shape[0],refArray.shape[1],5))
    tgtVec0 = np.zeros((tgtArray.shape[0],tgtArray.shape[1],5))
    
    rx,ry = np.meshgrid(xrange(refArray.shape[1]),xrange(refArray.shape[0]))
    tx,ty = np.meshgrid(xrange(tgtArray.shape[1]),xrange(tgtArray.shape[0]))
#     print refArray.shape[0],refArray.shape[1]
#     print rx,ry
    
    
    refVec0[:,:,0] = rx*alpha/refArray.shape[1]
    refVec0[:,:,1] = ry*alpha/refArray.shape[0]
    refVec0[:,:,2:5] = refArray[:,:,0:3]
    tgtVec0[:,:,0] = tx*alpha/tgtArray.shape[1]
    tgtVec0[:,:,1] = ty*alpha/tgtArray.shape[0]
    tgtVec0[:,:,2:5] = tgtArray[:,:,0:3]
    
    refVec = refVec0.reshape(-1,5)
    tgtVec = tgtVec0.reshape(-1,5)

    print 'tgttest\n',refVec.reshape(refArray.shape[0],refArray.shape[1],5)

#     print 'tgtVec\n',tgtVec
    
    ver = cartesian([[0,alpha],[0,alpha],[0,255],[0,255],[0,255]])
    print len(ver)
    
    
    resArray = np.zeros((iterationTimes,tgtVec.shape[0],3),'uint8')
    
#     outpFile = open('output.txt','w')
    
    for i in xrange(iterationTimes):
        print '%dth iterations'%(i+1)
        refSample = np.array(random.sample(refVec, numberOfSamples))
        tgtSample = np.append(np.array(random.sample(tgtVec, numberOfSamples-len(ver))),ver,axis=0)
        
        print 'ref\n',refSample
        print 'tgt\n',tgtSample
        distMat = getDistMat(tgtSample,refSample)
#         print distMat
        
        assignment = np.array(linear_assignment(distMat))
#         print 'assignment\n',assignment
        print np.append(tgtSample,refSample[assignment[:,1]],axis=1)
        np.savetxt('output.txt',np.append(tgtSample,refSample[assignment[:,1]],axis=1),fmt='%d', delimiter='\t')
#         print 'aref\n',refSample[assignment[:,1]]
#         print 'atgt\n',tgtSample[assignment[:,0]]
        points = tgtSample
        values = refSample[assignment[:,1]]
#         print 'points\n',points
#         print 'values\n',values
        
        h0 = ip.LinearNDInterpolator(points=points,values=values[:,2])
        h1 = ip.LinearNDInterpolator(points=points,values=values[:,3])
        h2 = ip.LinearNDInterpolator(points=points,values=values[:,4])
        
        resArray[i,:,0] = h0(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2],tgtVec[:,3],tgtVec[:,4])+0.499
        resArray[i,:,1] = h1(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2],tgtVec[:,3],tgtVec[:,4])+0.499
        resArray[i,:,2] = h2(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2],tgtVec[:,3],tgtVec[:,4])+0.499
#         np.savetxt('outputRes%d.txt'%(i),resArray[i], fmt='%d', delimiter='\t')
            
    outpArray = np.zeros((tgtVec.shape[0],3),'uint8')
#     outpArray[:,0] = h0(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])
#     outpArray[:,1] = h1(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])
#     outpArray[:,2] = h2(tgtVec[:,0],tgtVec[:,1],tgtVec[:,2])
    np.median(resArray, axis=0, out=outpArray)
    print outpArray.shape
    print outpArray
    np.savetxt('output2.txt',outpArray, fmt='%d', delimiter='\t')
    
    outpImg = Image.fromarray(outpArray.reshape(tgtArray.shape[0],tgtArray.shape[1],3))
    outpImg.save('out.jpg')
    
#     outpBlured = reduceEdge(tgtArray, outpArray.reshape(tgtArray.shape))
#     outpBluredImg = Image.fromarray(outpBlured)
#     outpBluredImg.save('outBlured.jpg')
    