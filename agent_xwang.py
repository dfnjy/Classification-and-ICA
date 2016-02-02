from __future__ import division  
import numpy as np
from sklearn.svm import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.neighbors import *
from sklearn.ensemble import RandomForestClassifier

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')

class Agent(object):
    def __init__(self, name):
        self.name = name
        self.my_products = []
        self.product_labels = [] #
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def choose_one_product(self, products):
        abstract()
        
    def add_to_my_products(self, product, label):
        self.my_products.append(product)
        self.product_labels.append(label) #good or bad

class MyAgent(Agent):
    """Chooses my product"""
    def __init__(self, name):
        self.name = name
        self.my_products = []
        self.product_labels = []
        self.CR=[2,2,2,2,2,2,2,2]
        self.CN=[2,2,2,2,2,2,2,2]
        self.CP=[1,1,1,1,1,1,1,1]
        self.X_train=[]
        self.tclass=0

    def add_to_my_products(self, product, label):
        self.my_products.append(product)
        self.X_train.append(product.features)
        self.product_labels.append(label) #good or bad
        #self.CN[self.tclass]+=1
        #if label:
        #    self.CR[self.tclass]+=1
        if label:
            for i in range(0,8):
                self.CN[i]+=1
                if self.CP[i]>0.5:
                    self.CR[i]+=1
        else:
            for i in range(0,8):
                self.CN[i]+=1
                if self.CP[i]<0.5:
                    self.CR[i]+=1

    def fit_classifiers(self):
        self.RFC=RandomForestClassifier(n_estimators=10)
        self.RFC.fit(self.X_train,self.product_labels)

        self.GNB=GaussianNB()
        self.GNB.fit(self.X_train,self.product_labels)

        self.DTC=DecisionTreeClassifier()
        self.DTC.fit(self.X_train,self.product_labels)

        self.KNC=KNeighborsClassifier(n_neighbors=2)
        self.KNC.fit(self.X_train,self.product_labels)

        self.BNB=BernoulliNB(alpha=0)
        self.BNB.fit(self.X_train,self.product_labels)

        self.LSVC=SVC(kernel='linear', probability=True)
        self.LSVC.fit(self.X_train,self.product_labels)

        self.RSVC=SVC(kernel='rbf', probability=True)
        self.RSVC.fit(self.X_train,self.product_labels)

        self.SSVC=SVC(kernel='sigmoid', probability=True)
        self.SSVC.fit(self.X_train,self.product_labels)


    def predict_probs(self, products):
        temp=[]
        tempp=self.CR[0]/self.CN[0]
        RFC=self.RFC.predict_proba(self.X_prod)
        temp=[x*tempp for x in RFC]
        #temp=self.RFC.predict_proba(self.X_prod)
        tprob=0
        tpoint=0
        tclass=0
        tlist=temp
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i

        tempp=self.CR[1]/self.CN[1]
        GNB=self.GNB.predict_proba(self.X_prod)
        temp=[x*tempp for x in GNB]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=1
                tlist=temp
        tempp=self.CR[2]/self.CN[2]
        DTC=self.DTC.predict_proba(self.X_prod)
        temp=[x*tempp for x in DTC]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=2
                tlist=temp
        tempp=self.CR[3]/self.CN[3]
        KNC=self.KNC.predict_proba(self.X_prod)
        temp=[x*tempp for x in KNC]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=3
                tlist=temp
        tempp=self.CR[4]/self.CN[4]
        BNB=self.BNB.predict_proba(self.X_prod)
        temp=[x*tempp for x in BNB]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=4
                tlist=temp
        tempp=self.CR[5]/self.CN[5]
        LSVC=self.LSVC.predict_proba(self.X_prod)
        temp=[x*tempp for x in LSVC]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=5
                tlist=temp
        tempp=self.CR[6]/self.CN[6]
        RSVC=self.RSVC.predict_proba(self.X_prod)
        temp=[x*tempp for x in RSVC]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=6
                tlist=temp
        tempp=self.CR[7]/self.CN[7]
        SSVC=self.SSVC.predict_proba(self.X_prod)
        temp=[x*tempp for x in SSVC]
        for i in range (0,len(temp)):
            if tprob<temp[i][1]:
                tprob=temp[i][1]
                tpoint=i
                tclass=7
                tlist=temp
        
        tprice=1000
        if tprob>0.3:
            tpara=0.3
        else:
            tpara=0
        for i in range(0,len(tlist)):
            if (tlist[i][1]>tprob-tpara and tprice>products[i].price):
                tpoint=i
                tprice=products[i].price

        self.CP[0]=RFC[tpoint][1]
        self.CP[1]=GNB[tpoint][1]
        self.CP[2]=DTC[tpoint][1]
        self.CP[3]=KNC[tpoint][1]
        self.CP[4]=BNB[tpoint][1]
        self.CP[5]=LSVC[tpoint][1]
        self.CP[6]=RSVC[tpoint][1]
        self.CP[7]=SSVC[tpoint][1]
        self.tpoint=tpoint
        self.tclass=tclass

    def get_features(self, products):
        self.X_prod = []
        for i in range (0,len(products)):
            self.X_prod.append(products[i].features)

    def choose_one_product(self, products):
        self.fit_classifiers()
        self.get_features(products)
        self.predict_probs(products)
        return self.tpoint
