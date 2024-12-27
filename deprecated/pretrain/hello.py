import torch
import torchinfo ; print('torchinfo',torchinfo.__version__)
import siliconperception ; print('siliconperception',siliconperception.__version__)
from siliconperception.IE120NX import IE120NX

encoder = IE120NX.from_pretrained('siliconperception/IE120NX')

# push locally saved model to hub
#encoder = IE120NX.from_pretrained('./pretrain/ie120nx_0.0.3')
#encoder.push_to_hub("siliconperception/IE120NX")

torchinfo.summary(encoder,col_names=["input_size","output_size","num_params"],input_size=(1,3,768,768))
