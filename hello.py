import torch
import torchinfo ; print('torchinfo',torchinfo.__version__)
import siliconperception ; print('siliconperception',siliconperception.__version__)
from siliconperception.IE120NX import IE120NX

encoder = IE120NX()
torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,768,768))
print(dir(IE120NX))
