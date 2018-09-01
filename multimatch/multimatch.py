import cv2
import numpy as np
from numpy.linalg import inv
from matchfilter import Matcher
from gridmatch import match_multi
import matplotlib.pyplot as plt
from time import time
from glob import glob

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
    def __init__(self, nums, feats, relations, known=None):
        self.nums, self.known = nums, known
        self.relations, self.feats = relations, feats
        print('known', known)
        if known is None:
            i = self.nums.index(sorted(self.nums)[len(self.nums)//2])
            feat, num = self.feats[i], self.nums[i]
            self.known = [(num, add_one(feat.kps), add_one(feat.kps))]
        
    def match_multi(self, dim=6):
        rst = match_multi(self.nums, self.relations, self.known, dim)
        for i in range(len(self.nums)):self.feats[i].set_proj(rst[i])

    def get_bound_prjs(self, diag):
        bound = []
        for i in self.feats:
            if not i.prj is None: bound.append(i.bound(k=1).T)
        bound = np.vstack(bound)
        minb, maxb = bound.min(axis=0), bound.max(axis=0)
        self.bound = minb[0],minb[1],maxb[0],maxb[1]
        l = ((maxb[0]-minb[0])**2+(maxb[1]-minb[1])**2)**0.5
        offx, offy, k = minb[0], minb[1], diag/l
        size = np.array((maxb[0]-minb[0], maxb[1]-minb[1]))*k
        prjs = [i.trans(-offx, -offy, k, True) for i in self.feats]
        return (tuple(size.astype(np.uint16)), prjs)

class FeatureGraph:
    def __init__(self):
        self.nums, self.feats, self.bins = [], [], []

    def add_feat(self, num, feat):
        self.nums.append(num)
        self.feats.append(feat)
        
    def build_relation(self):
        self.relations = rst = []
        for i in range(len(self.nums)):
            for j in range(i+1, len(self.nums)):
                kps1, feats1 = self.feats[i].kps, self.feats[i].feats
                kps2, feats2 = self.feats[j].kps, self.feats[j].feats
                matcher = Matcher(8, 0.005)
                idx, msk, m = matcher.filter(kps1,feats1,kps2,feats2)
                #print('check', i,j,len(msk), msk.sum())
                if max(abs(m[0]-1),abs(m[4]-1))>0.6:continue
                if msk.sum()/len(msk)<0.08: continue
                idx1, idx2 = idx[msk].T
                kps1, kps2 = add_one(kps1[idx1]), add_one(kps2[idx2])
                rst.append([(self.nums[i], self.nums[j]), (kps1, kps2)])
        return rst

    def build_group(self):
        self.bins = [0] * len(self.nums)
        group = self.group = [set([i]) for i in self.nums]
        for (n1,n2),(f1,f2) in self.relations:
            self.bins[self.nums.index(n1)] += 1
            self.bins[self.nums.index(n2)] += 1
            i1 = [i for i in group if n1 in i][0]
            i2 = [i for i in group if n2 in i][0]
            if i1 == i2:continue
            group.remove(i1)
            group.remove(i2)
            group.append(i1.union(i2))

    def build_set(self):
        fs = []
        for pts in self.group:
            nums = list(pts)
            feats = [self.feats[self.nums.index(i)] for i in nums]
            relations = [i for i in self.relations if i[0][0] in pts]
            fs.append(FeatureSet(nums, feats, relations))
        return fs
            
    def info(self):
        print('Infomation:')
        for i,f,n in zip(self.nums, self.feats, self.bins):
            print('num:%s times:%s'%(i, n))
        print('isolate:', [i for i in range(len(self.nums)) if self.bins[i]==0])
        print('group', self.group)
    
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
    imgs = glob('imgs/*.jpg')[:]
    #print(imgs[:5])
    imgs = [cv2.imread(i)[:,:,[2,1,0]] for i in imgs]
    print('add feat')
    graph = FeatureGraph()
    for i in range(len(imgs)):
        graph.add_feat(i, get_feat(imgs[i], 512, 3))
    
    relation = graph.build_relation()
    graph.build_group()
    featsets = graph.build_set()
    graph.info()
    for featset in featsets:
        print('start')
        featset.match_multi(6)
        print('end')
        size, prjs = featset.get_bound_prjs(2048)
        buf = np.zeros(size[::-1]+(3,), dtype=np.uint8)
        for num, prj in zip(featset.nums, prjs):
            if prj is None: continue
            cur = cv2.warpPerspective(imgs[num], prj, size)
            buf[:] = np.where(cur>buf, cur, buf)

        plt.imshow(buf)
        plt.show()
    
    '''
    print('OK')
    featset.add_known(0, featset.feats[0])
    #featset.add_known(5, featset.feats[5])
    print('build relation')
    featset.get_relations()
    print('OK')
    featset.info()
    print('match multi')
    featset.match_multi(6)
    print('OK')
    
    size, prjs = featset.get_bound_prjs(2048)

    buf = np.zeros(size[::-1]+(3,), dtype=np.uint8)
    for img, prj in zip(imgs, prjs):
        if prj is None: continue
        cur = cv2.warpPerspective(img, prj, size)
        buf[:] = np.where(cur>buf, cur, buf)

    plt.imshow(buf)
    plt.show()
    '''
