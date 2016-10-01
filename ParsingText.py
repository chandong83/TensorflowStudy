#encoding:Utf-8
'''
next랑 one-hot vector부분을 채울것, unzip code 확인 필요, what the fuck..
뒤로 갈수록 os.path.isfile을 확인하는 부분에서 Hard IO를 반복해서 접근하기 때문에 느려질수 밖에 없다.
이 부분에 대해서 따로 i값에 대한 관리를 해주는 부분이 필요하다.

그리고 데이터에 대해서 검증을 하는 부분이 필요하다 1. html값으로 받는 경우 2. flicker 또는 해당 하는 값에 대한 에러 표시
정형화된 데이터 규격을 가지고 있을 가능성이 높으므로 간단한 pattern matching code를 구현하면 된다.

그리고 간단하게 logistic 방식의 Deep learning을 구현하고 테스트 하고 더 많은 양을 확인하는게 좋을 듯 싶다.
하지만 logistic 한 것에는 variable length problem을 어떻게 구현 해야하는지에 대한 고민이 필요하다.
wnid 고양이, 개 구별
'''


import sys, os, tarfile
import collections, pickle
import urllib2, requests
import numpy as np
import re
from matplotlib.pyplot import imshow
from PIL import Image
#총 class 갯수인 maxcount 값으로 n개의 클래스에 해당하는 데이터 셋을 만든다.

maxcount=1

imageandwnidurl="http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz"
#file 이름이 뭘로 저장되나요?? 다시 맵핑 시켜서 파일이름을 적어주나요?
wnidurl="http://image-net.org/archive/gloss.txt"
Datasets = collections.namedtuple('Datasets', ['train'])

def download_web_file(url, name='None'):
    '''name이 초기화 되지 않으면 url의 마지막 이름으로 저장된다.'''
    if name == 'None':
        test=re.match(r".*/(.*)", url)
        name=test.group(1)
    '''파일이 존재하는지 확인한다.'''
    if os.path.isfile(name):
        print "%s is already existed" %name
        return name

    print "Downloading %s" % name

    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')

    with open(name, 'wb') as f:
        dl = 0
        total_length = int(total_length)
        for data in response.iter_content():
            dl += len(data)
            f.write(data)
            done = int(100 * dl / total_length)
            sys.stdout.write("\r[%s%s] %s %%" % ('=' * done, ' ' * (100 - done), done))
            sys.stdout.flush()
        print 'Done'
    return name

#예외 처리가 필요하다. url이 존재하지 않을 경우? 내가 원하는 파일인지 아닌지 확인은 어떻게 할까? file이 존재한다면?

def tgzfile_extract(name):
    print 'extract the %s file' %name
    tar=tarfile.open(name, 'r:gz')
    tar.extractall()
    tar.close()

#이미지를 다운로드 한다. 파일 저장은 wnid에 _+i_.jpg 로 한다
def download_image_by_url(url, name):
    opener = urllib2.build_opener()
    urllib2.install_opener(opener)
    i=0


    modifiedname=name+'_'+str(i)+'.jpg'

    while(os.path.isfile(modifiedname)):
        i+=1
        modifiedname = name + '_' + str(i) + '.jpg'
    with open(modifiedname, 'wb') as f:
        try:
            f.write(urllib2.urlopen(url).read())
            f.close()
        except:
            print 'url is not exist'
    return modifiedname


#Dictionary에 wnid의 개수를 세고 그 갯수를 반환한다
def addToDic(dic, name, count):
    print name
    if dic.has_key(name):
        return count
    else:
        count += 1
        dic[name]= count
        return count

#Extract시 readError 문제를 해결해야함 파일의 형식 문제인가? 아님... 파일 받을 때 tar파일로 변형이 되나? 파일을 다시 받아 볼 필요가 있는 듯...
class DataSet:
    test="This is the test"

    def __init__(self, images, labels, size):
        self._images=images
        self._labels=labels
        self._size=size

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def size(self):
        return self._size

    def next_batch(self):
        print 'next batch'

    def sum(self, a, b):
        result=a+b
        print("%s님 %s+%s=%s입니다" %(self.name, a, b, result))

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_data_sets():
    if os.path.isfile('imagenet.pkl'):
        print 'test'
        with open('imagenet.pkl', 'rb') as input:
            train = pickle.load(input)

    else:
        '''이미지 URL과 wnid가 적혀있는 Url 파일을 다운 받는다.'''
        #download_web_file(imageandwnidurl)
        #wnidfile = download_web_file(wnidurl)
        #tgzfile_extract(wnidfile)

        k = 5  # 얻어올 갯수
        n = 0

        images=[]
        labels=[]
        size=[]

        count = 0
        dataSetSize=0
        dic = {'0': '0'}
        dic.clear()

        for line in file('fall11_urls.txt'):
            match = line.split('\t')

            name = re.match(r"(.*)_.*", match[0]).group(1)
            count = addToDic(dic, name, count)
            if (count == maxcount + 1):
                break;
            name=download_image_by_url(match[1], name)



            print 'download the imagfile'+name
            try:
               img = Image.open(name)
               #이미지의 모드를 확인하고 jpeg형태가 아니면 이미지를 삭제한다.
               #쓰레드를 이용해서 다운로드와 기타 행동에 대한 속도를 올릴 수 있다
               #하지만 이미지가 각 이미지 사이트에서 오류라고 보낸 경우에는 걸러지지 않는다.

               print img.mode

               data=np.asarray(img.getdata())

               images.append(data)
               labels.append(count)  # one-hot vector로 변경해야함
               size.append(img.size)
               dataSetSize += 1
               print 'count :', str(count) + '개'

            except:
                print 'it is not image file'


        train=DataSet(images, labels, size)
        save_object(train, 'imagenet.pkl')
    return Datasets(train=train)

datasets=read_data_sets()


#datasets의 train의 image, label, size 직접 접근 하는 코드
#print datasets.train.images[0]
#print datasets.train.labels[0]
#print datasets.train.size[0]

exit(0)


#tarfile.open("imagenet_fall11_urls.tgz").extract()

obj=""
k=5 #얻어올 갯수
n=0
DataSets=[]

for line in file('fall11_urls.txt'):
    #Checkig for the .o file
    match=line.split('\t')
    #print match[0]
    #print match[1]

    DataSets.append(match)
    print DataSets[n][0]
    print DataSets[n][1]
    if(k==n):
        break;
    n+=1

#image를 웹에서 받아와서 바로 메모리로 올리는 코드
response=urllib2.urlopen(DataSets[0][1])
img_bytes=response.read()
#print img_bytes
name=download_web_images(DataSets[0][1], DataSets[0][0])

img=Image.open(name)

#image 속의 raw data
#image 속의 raw data와 header를 분리하여  x,y와 wnid를 추출하여야 한다

print img.format, img.size, img.mode
#The format attribue identifies the source of an image.
#If the image was not read from a file, it is et to None.
#The size attribute is a 2-tuple containing widht and height(in pixels).
#The mode attribue defines the nubmer and names of the bands
#in the image, and also the pixel type and depth
#Common modes are L(luminance) for greyscale images,
#"RGB" for true colour images, adn "CMYK" for pre-press images.

img.show()

pixels=np.array(img.getdata())
print pixels

#wnid를 key를 1000개를 받아서 딕셔너리를 만든다.
#value가 되는 var값이 one-hot vector의 값이 된다.



# x, label, data, -> data의 경우 정상적인 데이터인지 확인하는 사전처리가 필요하다.

# 이 데이터들을 가지고 데이터 셋을 만들어야 한다.
# 그러려면 tensorflwo mnist의 데이터 셋에 대해서 공부를 해야한다.(class, one_hot vector)
