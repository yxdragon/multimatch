import cv2
import numpy as np
from numpy.linalg import inv
from matchfilter import Matcher
from gridmatch import match_multi
import matplotlib.pyplot as plt
from time import time

def add_one(pts):
    return np.hstack((pts, np.ones((len(pts),1))))

class Feature:
    def __init__(self, kps, feats, k=1, rect = None):
        self.kps, self.feats = kps, feats
        if not rect is None: self.set_rect(*rect)
        self.prj = None

    def set_rect(self, h, w):
        self.rect = np.array([(0,0),(w,0),(w,h),(0,h)])
        self.k = (w**2+h**2)**0.5

    def set_proj(self, para):
        para = np.hstack((para.A1,[0]*(8-para.size),[1]))
        self.prj = para.reshape((3,3)).astype(np.float32)

    def trans(self, offx=0, offy=0, k=1, prek=False):
        if self.prj is None: return None
        para = self.prj.copy()
        if prek:para[:,:2]/=self.k
        para[0] += para[2] * offx
        para[1] += para[2] * offy
        para[:2] *= k
        return para
        
    def project(self, pts, offx=0, offy=0, k=0):
        prj = self.trans(offx, offy, k or self.k)
        pts = np.dot(prj,add_one(pts).T)
        return pts/pts[2]

    def bound(self, offx=0, offy=0, k=0):
        return self.project(self.rect/self.k, offx, offy, k)

class FeatureSet:
    def __init__(self):
        self.num, self.feats, self.bins = [], [],[]
        self.relations, self.known = [], []

    def add_feat(self, num, feat):
        self.num.append(num)
        self.feats.append(feat)

    def add_known(self, num, feat):
        self.known.append((num, add_one(feat.kps), add_one(feat.kps)))

    def get_relations(self):
        self.relations = rst = []
        for i in range(len(self.feats)):
            for j in range(i+1, len(self.feats)):
                kps1, feats1 = self.feats[i].kps, self.feats[i].feats
                kps2, feats2 = self.feats[j].kps, self.feats[j].feats
                matcher = Matcher(8, 0.003)
                idx, msk, m = matcher.filter(kps1,feats1,kps2,feats2)
                print('check', i,j,len(msk), msk.sum())
                if max(abs(m[0]-1),abs(m[4]-1))>0.6:continue
                if msk.sum()/len(msk)<0.08: continue
                idx1, idx2 = idx[msk].T
                kps1, kps2 = add_one(kps1[idx1]), add_one(kps2[idx2])
                rst.append([(i, j), (kps1, kps2)])
        self.bins = [0] * len(self.num)
        for i,j in [i[0] for i in rst]:
            self.bins[i]+=1
            self.bins[j]+=1
        return rst
            
    def match_multi(self, dim=6):
        num = [i for i,j in zip(self.num, self.bins) if j>0]
        rst = match_multi(num, self.relations, self.known, dim)
        for i in num:
            self.feats[self.num.index(i)].set_proj(rst[num.index(i)])
        return rst

    def get_bound_prjs(self, diag):
        bound = []
        for i in self.feats:
            if not i.prj is None: bound.append(i.bound(k=1).T)
        bound = np.vstack(bound)
        minb, maxb = bound.min(axis=0), bound.max(axis=0)
        self.bound = minb[0],minb[1],maxb[0],maxb[1]
        l = ((maxb[0]-minb[0])**2+(maxb[1]-minb[1])**2)**0.5
        offx, offy, k = minb[0], minb[1], diag/l
        self.offx, self.offy, self.k = offx, offy, k
        size = np.array((maxb[0]-minb[0], maxb[1]-minb[1]))*k
        prjs = [i.trans(-offx, -offy, k, True) for i in self.feats]
        return (tuple(size.astype(np.uint16)), prjs)

    def info(self):
        print('Infomation:')
        for i,f,n in zip(self.num, self.feats, self.bins):
            print('num:%s times:%s'%(i, n))
        print('isolate:', [i for i in range(len(self.num)) if self.bins[i]==0])
              
def get_feat(img, diag, sigma=3):
    detector = cv2.xfeatures2d.SURF_create()
    l = diag/np.sqrt((np.array(img.shape)**2).sum())
    size = tuple((np.array(img.shape)*l).astype(np.int16))[:2]
    img2 = cv2.resize(img, size[::-1])
    img2 = cv2.GaussianBlur(img2, (sigma*4+1,)*2, sigma)
    kps, feats = detector.detectAndCompute(img2, None)
    kps = np.array([k.pt for k in kps])/diag
    return Feature(kps, feats, rect=img.shape[:2])

if __name__ == '__main__':
    imgs = ['imgs/DJI_0006.jpg','imgs/DJI_0005.jpg','imgs/DJI_0007.jpg',
            'imgs/DJI_0008.jpg','imgs/DJI_0009.jpg','imgs/DJI_0010.jpg',
            'imgs/DJI_0011.jpg','imgs/DJI_0012.jpg','imgs/DJI_0013.jpg']
    imgs = [cv2.imread(i)[:,:,[2,1,0]] for i in imgs]
    print('add feat')
    featset = FeatureSet()
    for i in range(len(imgs)):
        featset.add_feat(i, get_feat(imgs[i], 512, 3))
    print('OK')
    featset.add_known(3, featset.feats[3])
    #featset.add_known(5, featset.feats[5])
    print('build relation')
    featset.get_relations()
    print('OK')
    print(len(featset.relations))
    print('match multi')
    featset.match_multi(6)
    print('OK')
    featset.info()
    size, prjs = featset.get_bound_prjs(2048)

    buf = np.zeros(size[::-1]+(3,), dtype=np.uint8)
    for img, prj in zip(imgs, prjs):
        if prj is None: continue
        cur = cv2.warpPerspective(img, prj, size)
        buf[:] = np.where(cur>buf, cur, buf)

    plt.imshow(buf)
    plt.show()

