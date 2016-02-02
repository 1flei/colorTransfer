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

def smooth(h,x,alpha,verrange,gamma=1,k=5,eps=0.4):
    noisy = multivariate_normal([0,0,0,0,0],np.diag([alpha/255.*gamma,alpha/255.*gamma,gamma,gamma,gamma]),size=(k,x.shape[0]))
#     print 'noisy\n',noisy.shape,noisy
#     xn = np.array((inRange(x+ni,verrange[:,0],verrange[:,1]) for ni in noisy))

    xn = np.array([x+ni for ni in noisy])
        
    xn = np.fmax(np.fmin(xn,verrange[:,1]-eps),verrange[:,0]+eps)
    
    print 'xn\n',xn.shape,xn
#     print 'hxn\n',h(xn).shape,h(xn)
#     np.savetxt('xn.txt',xn[:,0], delimiter='\t')
#     np.savetxt('hxn.txt',np.average(h(xn), axis=0), delimiter='\t')
    return np.average(h(xn), axis=0)

def smoothFunc(h,alpha,verrange,gamma=1):
    return lambda x: smooth(h,x,alpha,verrange,gamma)

# def smoothPoint(points,gamma=1.,out=None):
#     for i,p in enumerate(points):
#         for i,p in enumerate(points):
#             
        

if __name__ == '__main__':
    random.seed(time.time())
    refImg = Image.open('../res/colorTransfer/reference1.jpg')
    tgtImg = Image.open('../res/colorTransfer/20160107200218.jpg')
    
    numberOfSamples = 300
    alpha = 300.
    gamma = 5.
    
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
    
    refVec0[:,:,0] = rx*alpha*2/(refArray.shape[0]+refArray.shape[1])
    refVec0[:,:,1] = ry*alpha*2/(refArray.shape[0]+refArray.shape[1])
    refVec0[:,:,2:5] = refArray[:,:,0:3]
    tgtVec0[:,:,0] = tx*alpha*2/(tgtArray.shape[0]+tgtArray.shape[1])
    tgtVec0[:,:,1] = ty*alpha*2/(tgtArray.shape[0]+tgtArray.shape[1])
    tgtVec0[:,:,2:5] = tgtArray[:,:,0:3]
    
    refVec = refVec0.reshape(-1,5)
    tgtVec = tgtVec0.reshape(-1,5)

#     print 'tgttest\n',refVec.reshape(refArray.shape[0],refArray.shape[1],5)

#     print 'tgtVec\n',tgtVec

    
    verRange = np.array([[0,alpha*2/(tgtArray.shape[0]+tgtArray.shape[1])*tgtArray.shape[1]],[0,alpha*2/(tgtArray.shape[0]+tgtArray.shape[1])*tgtArray.shape[0]],[0,255],[0,255],[0,255]])
    ver = cartesian(verRange)
    print 'ver[%d]:\n%s'%(len(ver),verRange)
    
#     outpFile = open('output.txt','w')
    
    print 'sampling'
    refSample = np.array(random.sample(refVec, numberOfSamples))
    tgtSample = np.append(np.array(random.sample(tgtVec, numberOfSamples-len(ver))),ver,axis=0)
    
    print 'assignment problem'
#     print 'ref\n',refSample
#     print 'tgt\n',tgtSample
    distMat = getDistMat(tgtSample,refSample)
#         print distMat
    
    assignment = np.array(linear_assignment(distMat))
#         print 'assignment\n',assignment
#     print np.append(tgtSample,refSample[assignment[:,1]],axis=1)
    np.savetxt('output.txt',np.append(tgtSample,refSample[assignment[:,1]],axis=1),fmt='%d', delimiter='\t')
#         print 'aref\n',refSample[assignment[:,1]]
#         print 'atgt\n',tgtSample[assignment[:,0]]
    points = tgtSample
    values = refSample[assignment[:,1]]
#         print 'points\n',points
#         print 'values\n',values

    print 'interpolating'
    
    h0 = ip.LinearNDInterpolator(points=points,values=values[:,2])
    h1 = ip.LinearNDInterpolator(points=points,values=values[:,3])
    h2 = ip.LinearNDInterpolator(points=points,values=values[:,4])
    sh0 = smoothFunc(h0,verrange=verRange,alpha=alpha,gamma=gamma)
    sh1 = smoothFunc(h1,verrange=verRange,alpha=alpha,gamma=gamma)
    sh2 = smoothFunc(h2,verrange=verRange,alpha=alpha,gamma=gamma)
        
            
    outpArray = np.zeros((tgtVec.shape[0],3),'uint8')
    outpArray[:,0] = sh0(tgtVec)+0.499
    outpArray[:,1] = sh1(tgtVec)+0.499
    outpArray[:,2] = sh2(tgtVec)+0.499
    outpArray2 = np.zeros((tgtVec.shape[0],3),'uint8')
    outpArray2[:,0] = h0(tgtVec)+0.499
    outpArray2[:,1] = h1(tgtVec)+0.499
    outpArray2[:,2] = h2(tgtVec)+0.499
    print outpArray.shape
    print outpArray
     
    outpImg = Image.fromarray(outpArray.reshape(tgtArray.shape[0],tgtArray.shape[1],3))
    outpImg.save('out.jpg')
    outpImg2 = Image.fromarray(outpArray2.reshape(tgtArray.shape[0],tgtArray.shape[1],3))
    outpImg2.save('out2.jpg')
     
    print 'done'
    
#     gammas = [0,1,2,4,8]
#     for g in gammas:
#         print 'g=%d'%(g)
#         s0 = smoothFunc(h0,verrange=verRange,alpha=alpha,gamma=g)
#         s1 = smoothFunc(h1,verrange=verRange,alpha=alpha,gamma=g)
#         s2 = smoothFunc(h2,verrange=verRange,alpha=alpha,gamma=g)
#         outpArray = np.zeros((tgtVec.shape[0],3),'uint8')
#         outpArray[:,0] = sh0(tgtVec)+0.499
#         outpArray[:,1] = sh1(tgtVec)+0.499
#         outpArray[:,2] = sh2(tgtVec)+0.499
# #         print outpArray.shape
# #         print outpArray
#         
#         outpImg = Image.fromarray(outpArray.reshape(tgtArray.shape[0],tgtArray.shape[1],3))
#         outpImg.save('out%d.jpg'%(g))