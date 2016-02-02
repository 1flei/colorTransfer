from PIL import Image
import numpy as np
import numpy.linalg as la
import colorsys

refImg = Image.open('../res/colorTransfer/105218yl5e8p8uzze944yu.jpg')
tgtImg = Image.open('../res/colorTransfer/photo13383414615_1.jpg')
refSize,tgtSize = refImg.size,tgtImg.size

refPix = refImg.load()
tgtPix = tgtImg.load()

BUCKETSIZE = 256


print refSize, tgtSize

def getHSVMap(pix0, pix1, HMap, SMap, VMap):
    HR,SR,VR,HT,ST,VT = np.zeros(BUCKETSIZE),np.zeros(BUCKETSIZE),np.zeros(BUCKETSIZE),np.zeros(BUCKETSIZE),np.zeros(BUCKETSIZE),np.zeros(BUCKETSIZE)
    for x in range(refSize[0]):
        for y in range(refSize[1]):
            r,g,b = refPix[x,y][0],refPix[x,y][1],refPix[x,y][2]
            h,s,v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            HR[int(h*(BUCKETSIZE-1))] += 1
            SR[int(s*(BUCKETSIZE-1))] += 1
            VR[int(v*(BUCKETSIZE-1))] += 1
            
    for x in range(tgtSize[0]):
        for y in range(tgtSize[1]):
            r,g,b = tgtPix[x,y][0],tgtPix[x,y][1],tgtPix[x,y][2]
            h,s,v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            HT[int(h*(BUCKETSIZE-1))] += 1
            ST[int(s*(BUCKETSIZE-1))] += 1
            VT[int(v*(BUCKETSIZE-1))] += 1

    #normalize
    HR = HR/la.norm(HR,ord=1)
    SR = SR/la.norm(SR,ord=1)
    VR = VR/la.norm(VR,ord=1)
    HT = HT/la.norm(HT,ord=1)
    ST = ST/la.norm(ST,ord=1)
    VT = VT/la.norm(VT,ord=1)

    def buildMap(refh, tgth, hmap, size=BUCKETSIZE):    
        refsum = 0
        tgtsum = 0
        j = 0
        for i in range(size):
            while refsum<tgtsum and j<size-1:
                refsum+=refh[j]
                j+=1
            hmap[i] = j
            tgtsum += tgth[i]

    buildMap(HR, HT, Hmap)
    buildMap(SR, ST, Smap)
    buildMap(VR, VT, Vmap)



Hmap = np.zeros(BUCKETSIZE)
Smap = np.zeros(BUCKETSIZE)
Vmap = np.zeros(BUCKETSIZE)
getHSVMap(refPix, tgtPix, Hmap, Smap, Vmap)
print 'H:\n',Hmap
print 'S:\n',Smap
print 'V:\n',Vmap


outpArray = np.zeros((tgtSize[1], tgtSize[0], 3), 'uint8')
for x in range(tgtSize[0]):
    for y in range(tgtSize[1]):
        r,g,b = tgtPix[x,y][0],tgtPix[x,y][1],tgtPix[x,y][2]
        h,s,v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
        ho = Hmap[int(h*(BUCKETSIZE-1))]
        so = Smap[int(s*(BUCKETSIZE-1))]
        vo = Vmap[int(v*(BUCKETSIZE-1))]
        #ro,go,bo = colorsys.hsv_to_rgb(ho/float(BUCKETSIZE), s, v)
        ro,go,bo = colorsys.hsv_to_rgb(ho/float(BUCKETSIZE), so/float(BUCKETSIZE), vo/float(BUCKETSIZE))
        outpArray[y,x,0],outpArray[y,x,1],outpArray[y,x,2] = 255*ro, 255*go, 255*bo
        if((x+255*y)%100000==0):
            print ho,so,vo
            print ho/float(BUCKETSIZE), so/float(BUCKETSIZE), vo/float(BUCKETSIZE)
            print 255*ro, 255*go, 255*bo

print outpArray
outpImg = Image.fromarray(outpArray)
outpImg.save('out.jpg')

