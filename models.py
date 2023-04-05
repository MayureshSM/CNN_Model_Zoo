import matplotlib.pyplot as plt
from torchvision.models import resnet50, alexnet, densenet121,googlenet,inception_v3,mobilenet_v3_small
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt

f=open('Imagenet_class.txt')
classes=f.read().split('\n')

class Object_classification():
    def __init__(self):
        self.transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Resize(232),
                                                                    transforms.CenterCrop(224)
                                                                    ])
    def model(self,model,img):
        if model=='resnet50':
            return(self.resnet50(img))
        if model=='alexnet':
            return(self.alexnet(img))
        if model=='densenet121':
            return(self.densenet121(img))
        if model=='googlenet':
            return(self.googlenet(img))
        if model=='inception':
            return(self.inception(img))
        if model=='mobilenet':
            return(self.mobilenet(img))
            
    def _modelrun(self,img,model):
        model.eval()
        img=self.transform(img)
        res=model(img.unsqueeze(0))
        return(classes[np.argsort(np.array(res.detach()[0])).astype(int)[-1] ])
    
    
    
    def resnet50(self,img): 
        model=resnet50(pretrained=True, progress=True)
        return(self._modelrun(img,model))


    def alexnet(self,img):
        model = alexnet(pretrained=True)
        return(self._modelrun(img,model))
        
    
    def densenet121(self,img):
        model=densenet121(pretrained=True)   
        return(self._modelrun(img,model))
        
    
    def googlenet(self,img):
        model=googlenet(pretrained=True)   
        return(self._modelrun(img,model))        
                                
    def inception(self,image):
        model=inception_v3(pretrained=True)
        return(self._modelrun(img,model))
        
    def mobilenet(self,img):
        model=mobilenet_v3_small(pretrained=True)
        return(self._modelrun(img,model))
        

img=plt.imread("/storage/emulated/0/Documents/Repos/Images/images4.jpg")
A=Object_classification()
A.mobilenet(img)      
