"""Суреттен ресурстарды жеке PNG-ке бөледі (python3 tools/slice_assets.py)."""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
import os
SRC=os.path.join(os.path.dirname(__file__),'source.png')
OUT=os.path.join(os.path.dirname(__file__),'..','img')
im=Image.open(SRC).convert('RGB')
# name: (x0,y0,x1,y1)
SPR={
'boy':(10,44,258,492),'girl':(258,70,490,495),
'pencil_long':(1058,50,1092,172),'pencil_short':(1106,98,1142,172),
'tree_big':(1188,42,1282,182),'tree_small':(1282,84,1358,152),
'book_thick':(1043,183,1218,306),'book_thin':(1222,218,1380,292),
'log_thick':(1392,158,1468,322),'log_thin':(1482,182,1512,322),
'rope_thick':(1048,314,1094,480),'rope_thin':(1146,318,1168,472),
'bear_big':(1192,316,1312,472),'bear_small':(1308,378,1372,470),
'box_big':(1378,344,1527,470),
'ruler_yellow':(868,64,926,406),'tape':(930,128,1022,232),'ruler_wood':(942,248,994,468),
'jacket':(498,48,684,226),'trousers':(512,228,658,486),'skirt':(674,186,832,296),'vest':(664,304,776,482),
'giraffe':(645,538,800,835),'elephant':(776,550,950,730),'dog':(990,562,1085,730),'cat':(835,730,922,842),
'btn_start':(1116,543,1417,614),'btn_next':(1116,622,1417,694),'btn_replay':(1116,700,1417,772),
'ico_sound':(1448,496,1522,570),'ico_home':(1448,644,1522,718),
'fb_ok':(20,902,208,992),'fb_bad':(216,902,394,992),
'star':(426,906,488,970),'star_gray':(636,903,696,966),
'trophy':(1276,884,1396,1012),'gift':(1412,894,1498,1012),
}
BG={'bg_class':(13,538,389,705),'bg_play':(399,538,623,697),'bg_school':(13,714,389,852),'bg_meadow':(399,704,623,852)}
def cut(box):
    c=im.crop(box); a=np.asarray(c).astype(int)
    near=(a.min(axis=2)>=232)
    lab,_=ndi.label(near)
    border=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    bg=np.isin(lab,list(border))
    fg=~bg
    # largest component (dilated to merge details)
    l2,n=ndi.label(ndi.binary_dilation(fg,iterations=3))
    if n>1:
        sizes=ndi.sum(fg,l2,range(1,n+1)); fg=fg&(l2==(1+int(np.argmax(sizes))))
    fg=ndi.binary_fill_holes(ndi.binary_closing(fg,iterations=1)) if False else fg
    alpha=Image.fromarray((fg*255).astype('uint8')).filter(ImageFilter.GaussianBlur(0.8))
    out=c.convert('RGBA'); out.putalpha(alpha)
    bb=out.getbbox(); out=out.crop(bb)
    s=2 if max(out.size)<400 else 1
    return out.resize((out.width*s,out.height*s),Image.LANCZOS) if s>1 else out
for k,b in SPR.items(): cut(b).save(f'{OUT}/{k}.png',optimize=True)
for k,b in BG.items():
    c=im.crop(b); c=c.resize((c.width*3,c.height*3),Image.LANCZOS); c.save(f'{OUT}/{k}.jpg',quality=88)
print('ok')
