import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from diffusers.models import AutoencoderKL

from DenoisingDiffusionProcess import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type= "stabilityai/sd-vae-ft-ema"#@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        self.model=AutoencoderKL.from_pretrained(model_type)
        
    def forward(self,input):
        return self.model(input).sample
    
    def encode(self,input,mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        """
            This is a simplified version of Latent Diffusion        
        """        
        
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        
        self.ae=AutoEncoder()
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,256,256)).shape[1]
        self.model=DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                             num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        #return self.output_T(self.model(*args,**kwargs))
        return self.output_T(self.ae.decode(self.model(*args,**kwargs)/self.latent_scale_factor))
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):   
        
        latents=self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        
        latents=self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('val_loss',loss)
        
        return loss
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None
    
    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
    
class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        pl.LightningModule.__init__(self)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        
        self.ae=AutoEncoder()
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,128,128)).shape[1]

        self.gender_emb = nn.Embedding(2, self.latent_dim) # 2 classes: male/female

        self.model=DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                        condition_channels=self.latent_dim*3,
                                                        num_timesteps=num_timesteps)
        
            
    @torch.no_grad()
    def forward(self,condition,*args,**kwargs):
        cond_img, gender = condition
        cond1, cond2 = torch.chunk(cond_img, 2, dim=1)

        latent1 = self.ae.encode(self.input_T(cond1)).detach() * self.latent_scale_factor
        latent2 = self.ae.encode(self.input_T(cond2)).detach() * self.latent_scale_factor
        
        gender_vec = self.gender_emb(gender.long())  # [B, latent_dim]
        gender_map = gender_vec.view(-1, self.latent_dim, 1, 1)
        gender_map = gender_map.expand(-1, -1, latent1.shape[2], latent1.shape[3])

        latents_condition = torch.cat([latent1, latent2, gender_map], dim=1)

        
        output_code=self.model(latents_condition,*args,**kwargs)/self.latent_scale_factor

        return self.output_T(self.ae.decode(output_code))
    
    def training_step(self, batch, batch_idx):   
        (cond_img, gender), output = batch
                
        with torch.no_grad():
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
            cond1, cond2 = torch.chunk(cond_img, 2, dim=1)

            latent1 = self.ae.encode(self.input_T(cond1)).detach() * self.latent_scale_factor
            latent2 = self.ae.encode(self.input_T(cond2)).detach() * self.latent_scale_factor
            
            gender_vec = self.gender_emb(gender.long()) # [B, latent_dim]
            gender_map = gender_vec.view(-1, self.latent_dim, 1, 1)
            gender_map = gender_map.expand(-1, -1, latent1.shape[2], latent1.shape[3])

            latents_condition = torch.cat([latent1, latent2, gender_map], dim=1)


        loss = self.model.p_loss(latents, latents_condition)
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        (cond_img, gender), output = batch

        
        with torch.no_grad():
            latents = self.ae.encode(self.input_T(output)).detach() * self.latent_scale_factor
            cond1, cond2 = torch.chunk(cond_img, 2, dim=1)

            latent1 = self.ae.encode(self.input_T(cond1)).detach() * self.latent_scale_factor
            latent2 = self.ae.encode(self.input_T(cond2)).detach() * self.latent_scale_factor

            gender_vec = self.gender_emb(gender.long())  # [B, latent_dim]
            gender_map = gender_vec.view(-1, self.latent_dim, 1, 1)
            gender_map = gender_map.expand(-1, -1, latent1.shape[2], latent1.shape[3])

            latents_condition = torch.cat([latent1, latent2, gender_map], dim=1)

        loss = self.model.p_loss(latents, latents_condition)
        self.log('val_loss',loss)
        
        return loss
    
    @torch.no_grad()
    def generate_from_parents(self, father_img, mother_img, gender, image_size=(128, 128)):
        """
        Generate a child image given father image, mother image, and desired gender.

        Args:
            father_img: Tensor [3, H, W] or PIL image → must be in [0,1] range
            mother_img: same
            gender: int scalar (0 or 1)
            image_size: (H, W)

        Returns:
            Generated image tensor in [0,1] range, shape [3, H, W]
        """
        self.eval()
        
        # Handle PIL images if needed
        if not isinstance(father_img, torch.Tensor):
            tfm = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
            father_img = tfm(father_img)
        if not isinstance(mother_img, torch.Tensor):
            tfm = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
            mother_img = tfm(mother_img)

        # Stack as batch
        cond = torch.cat([father_img.unsqueeze(0), mother_img.unsqueeze(0)], dim=0)  # [2, 3, H, W]
        cond = cond.unsqueeze(0).view(1, 6, *father_img.shape[1:])  # [1, 6, H, W]
        gender_tensor = torch.tensor([gender]).to(cond.device)

        # Run forward
        output = self.forward((cond.to(self.device), gender_tensor.to(self.device)))
        return output[0]  # remove batch dimension