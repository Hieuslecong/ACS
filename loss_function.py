from train import *

def Power_function_loss(my_preds, gt_preds):

  # q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗(alpha/(1-alpha))^gama

  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]

  eps = 0.00001
  alpha = sum(my_preds == 0.0)/(batch_size * IMG_SIZE * IMG_SIZE)
  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps
  q_alpha=beta*((alpha/(1-alpha))**gama)
  loss = -q_alpha*gt_preds*torch.nan_to_num(np.log(my_preds))
  loss -= (1-gt_preds)*torch.nan_to_num(np.log(1-my_preds))
  loss=loss.sum()
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss
def  Logarithmic_function_loss(my_preds, gt_preds):
  # loss = q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗ln(alpha/(1-alpha))
  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]

  eps = 0.00001
  alpha = len(my_preds == 0.0)/(batch_size * IMG_SIZE * IMG_SIZE)
  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps

  q_alpha=beta*(np.log(alpha/(1-alpha)))

  loss = -q_alpha*gt_preds*np.log(my_preds)
  loss -= (1-gt_preds)*np.log(1-my_preds)
  loss=loss.sum()
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
def  Exponential_function_loss(my_preds, gt_preds):
  # loss = q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗ln(alpha/(1-alpha))
  
#q(α) = β ∗ a^gama*(2α−1)
  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[2]
  my_preds1=my_preds.view(batch_size,IMG_SIZE,IMG_SIZE)
#   img_grid = torchvision.utils.make_grid(my_preds1)
#   matplotlib_imshow(img_grid, one_channel=True)
#   plt.show()
  img=my_preds1.detach().cpu().numpy()
  #print(img.shape,np.unique(img))

  #my_preds1=my_preds.view(batch_size,IMG_SIZE,IMG_SIZE)
#   print(my_preds1.shape)
#   my_preds1=(my_preds1.permute(1, 2, 0))
#   my_preds1=my_preds.cpu().numpy()
#   plt.imshow(my_preds1,cmap="gray")
#   plt.show()
  #gt_preds=gt_preds.view(batch_size,IMG_SIZE,IMG_SIZE)
  #print(torch.unique(my_preds))
#   gt_preds=gt_preds.detach().cpu().numpy()
#   my_preds=my_preds.detach().cpu().numpy()
  #plt.imshow(my_preds1)
  #print(my_preds.shape)
  #print(np.unique(my_preds1))

  #my_preds=my_preds+0.0000001
#   my_preds1 = torch.sigmoid(my_preds)
#   my_preds1 = (my_preds1 > 0.5).float()
  alpha = (len(np.argwhere(my_preds.detach().cpu().numpy() == 0.0)))/(batch_size * IMG_SIZE * IMG_SIZE)
#   print((gt_preds== 0.0).sum(),(gt_preds == 1.0).sum())
#   print('img')
#   print((my_preds == 0.0).sum(),(my_preds == 1.0).sum())
  #print(alpha,alpha/(1-alpha))
  #print(my_preds.sum(),IMG_SIZE * IMG_SIZE,alpha)
  beta=0.75
  gama=1
  a=10
  b=1
  eps = 0.00001
  #my_preds=my_preds.astype(np.uint8)
  #gt_preds=gt_preds.astype(np.uint8)
  
  #my_preds[my_preds == 0.0] = eps
  #my_preds[my_preds == 1.0] = 1.0 - eps
  #
  #print(alpha)
  #alpha = len(my_preds == 0.0)/(32 * 256 * 256)
  q_alpha=beta*a**(gama*(2*alpha-1))
  #print(q_alpha)
  #print(q_alpha)
  #print(np.log(my_preds))
  #print('aaaa')
  pos_weight =  torch.tensor(q_alpha*torch.ones([IMG_SIZE])).to(device='cuda:0')  # All weights are equal to 1
  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  #print(my_preds.type(),gt_preds.type())
  loss=criterion(my_preds, gt_preds)
  #print(loss)
  #print((my_preds == 0.0).sum(),len((my_preds == 0.0)))
#   loss=gt_preds*torch.log(my_preds+0.000001)
		
#   loss = -q_alpha*loss
#   loss -= (1-gt_preds)*torch.log(1-my_preds+0.000001)
#   #print(loss.sum(),torch.sum(loss))
#   loss=torch.nan_to_num(loss)
#   loss=loss.sum()
  
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss
def  Holist_function_loss(my_preds, gt_preds):
  # loss = q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗ln(alpha/(1-alpha))
  
#q(α) = β ∗ a^gama*(2α−1)
  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]

  eps = 0.00001
  
  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps

  #q_alpha=beta*a**(gama*(2*alpha-1))
  loss1= -alpha*gt_preds*np.log(my_preds)
  loss1-= (1-alpha)*(1-gt_preds)*np.log(1-my_preds)
  loss1=loss1.sum()
  loss2= (sum(gt_preds*my_preds) +lamda)/(sum(gt_preds)+sum(my_preds)-sum(gt_preds*my_preds)+lamda)
  loss = a*loss1 + b*(1-loss2)
  # loss = -q_alpha*gt_preds*np.log(my_preds)
  # loss -= (1-gt_preds)*np.log(1-my_preds)
  #loss=loss.sum()
  
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss
