이미지 수정을 위해 필요한 모듈들 

``` python
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
``` 
이미지 메타정보 출력

``` python
earth = Image.open('earth.jpg')
print(earth.format)
print(earth.size)
print(earth.width)
print(earth.height)
print(earth.mode)
```

#1 이미지 사이즈 조절

#2 (0, 300)부터 (450, 600) 까지 자르기

#3 90도 돌리기

#4 좌우, 상하 뒤집기

#5 블러 필터

#6 contour(윤곽) 필터

#7 detail(선명함) 필터

#8 smooth(부드러움) 필터

#9 emboss(윤곽) 필터

``` python

#1
earth_resize = earth.resize((600,300))
earth_resize.save("resize_image.jpg","JPEG")

#2
earth_crop = earth.crop((0,300,450,600))
earth_crop.save("crop_image.jpg","JPEG")

#3
earth_rotat = earth.rotate(90)
earth_rotat.save("rotate_image.jpg","JPEG")

#4
earth_lr = earth.transpose(Image.FLIP_LEFT_RIGHT)
earth_tb = earth.transpose(Image.FLIP_TOP_BOTTOM)
earth_lr.save("earth_lr_image.jpg","JPEG")
earth_tb.save("earth_tb_image.jpg","JPEG")

#5
earth_blur = earth.filter(ImageFilter.GaussianBlur(10))
earth_blur.save("blur_image.jpg","JPEG")

#6
earth_contour = earth.filter(ImageFilter.CONTOUR)
earth_contour.save("contour_image.jpg","JPEG")

#7
earth_detail = earth.filter(ImageFilter.DETAIL) 
earth_detail.save("detail_image.jpg","JPEG")

#8
earth_smooth = earth.filter(ImageFilter.SMOOTH)
earth_smooth.save("smooth_image.jpg","JPEG")

#9
earth_emboss = earth.filter(ImageFilter.EMBOSS)
earth_emboss.save("emboss_image.jpg","JPEG")

```

운석 이미지와 지구 이미지를 적당한 크기로 조정한 후 알맞은 크기의 이미지에 사진을 붙여넣어  합쳐준다.

``` python

e = Image.open('earth.jpg')
m = Image.open('met.jpg')

mf = m.transpose(Image.FLIP_TOP_BOTTOM)
er = e.resize((800,600))
mr = mf.resize((200,150))
ec = er.crop((0,100,800,600))

new_image = Image.new('RGB',(ec.size[0], ec.size[1]+150), (250,250,250))
new_image.paste(ec,(0,150))
new_image.paste(mr,(100,0))
new_image.save("merged_image.jpg","JPEG")

```

이미지을 이진모드로 열고 바이트 데이터로 읽어줌

``` python

from PIL import Image

with open("earth.jpg", "rb") as Img:
    byte_data = bytearray(Img.read())

```

랜덤 모듈을 가져와 넘파이 배열로 변환된 이미지를 랜덤으로 섞어준 후 
시각화

``` python
import random

img = Image.open('earth.jpg')
img_np = np.array(img)

for i in img_np:
    random.shuffle(i)
    
plt.imshow(img_np)
plt.axis('off')
plt.show()

```