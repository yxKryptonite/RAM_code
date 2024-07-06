import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IMG_SIZE = 448

class Demo:

    def __init__(self, imgs, ft, img_size):
        self.ft = ft # N+1, C, H, W
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():

                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                    # x, y = 401, 117
                    print("click on:", x, y)

                    src_ft = self.ft[0].unsqueeze(0)
                    src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel)  # 1, C

                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:]) # N, C, H, W
                    trg_vec = trg_ft.view(self.num_imgs - 1, num_channel, -1) # N, C, HW

                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    src_vec = F.normalize(src_vec) # 1, C
                    trg_vec = F.normalize(trg_vec) # N, C, HW
                    cos_map = torch.matmul(src_vec, trg_vec).view(self.num_imgs - 1, self.img_size, self.img_size).cpu().numpy() # N, H, W

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='r', s=scatter_size)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                        print("best correspondence:", max_yx[1], max_yx[0])
                        axes[i].clear()

                        heatmap = cos_map[i-1]
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=scatter_size)
                        axes[i].set_title('target image')

                    del cos_map
                    del heatmap
                    gc.collect()
                    plt.show()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


class XYViewer:
    def __init__(self, img_path):
        self.img = Image.open(img_path).convert('RGB')

    def plot(self):
        fig, ax = plt.subplots()
        ax.imshow(self.img)
        ax.axis('off')

        plt.show()

    
def visualize_max_xy(save_root, src_pos, max_xy, src_img_PIL, tgt_img_PIL, heatmap=None, filename=None):
    '''draw a point and heatmap (optional)'''
    fig, axes = plt.subplots(1, 2)
    src_img_PIL = src_img_PIL.resize((IMG_SIZE, IMG_SIZE))
    max_xy = (max_xy[0] * IMG_SIZE / tgt_img_PIL.size[0], max_xy[1] * IMG_SIZE / tgt_img_PIL.size[1])
    tgt_img_PIL = tgt_img_PIL.resize((IMG_SIZE, IMG_SIZE))
    axes[0].imshow(src_img_PIL)
    axes[0].scatter(src_pos[0], src_pos[1], c='r', marker='x')
    axes[0].set_title('source image')
    axes[0].axis('off')
    axes[1].imshow(tgt_img_PIL)
    axes[1].scatter(max_xy[0], max_xy[1], c='r', marker='x')
    if heatmap is not None:
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        axes[1].imshow(255 * heatmap, alpha=0.45, cmap='viridis')
    axes[1].set_title('target image')
    axes[1].axis('off')
    
    if filename:
        plt.savefig(f'{save_root}/{filename}.png')
    else:
        plt.savefig(f'{save_root}/max_xy.png')

def visualize_max_xy_list(save_root, src_pos_list, max_xy_list, src_img_PIL, tgt_img_PIL, filename=None):
    '''draw a list of (inlier) points'''
    fig, axes = plt.subplots(1, 2)
    src_img_PIL = src_img_PIL.resize((IMG_SIZE, IMG_SIZE))
    axes[0].imshow(src_img_PIL)
    axes[0].plot(src_pos_list[:, 0], src_pos_list[:, 1], c='r')
    axes[0].scatter(src_pos_list[:, 0], src_pos_list[:, 1], c='g', marker='.')
    axes[0].set_title('source image')
    axes[0].axis('off')
    axes[1].imshow(tgt_img_PIL)
    axes[1].plot(max_xy_list[:, 0], max_xy_list[:, 1], c='r')
    axes[1].scatter(max_xy_list[:, 0], max_xy_list[:, 1], c='g', marker='.')
    axes[1].set_title('target image')
    axes[1].axis('off')
    if filename:
        plt.savefig(f'{save_root}/{filename}.png')
    else:
        plt.savefig(f'{save_root}/max_xy_list.png')

def visualize_max_xy_linear(save_root, src_pos, src_dir, tgt_pos, tgt_dir, src_img_PIL, tgt_img_PIL, filename=None):
    '''draw an arrow'''
    fig, axes = plt.subplots(1, 2)
    src_img_PIL = src_img_PIL.resize((IMG_SIZE, IMG_SIZE))
    axes[0].imshow(src_img_PIL)
    axes[0].arrow(src_pos[0], src_pos[1], src_dir[0]*100, src_dir[1]*100, color='r', linewidth=2.5, head_width=12)
    axes[0].set_title('source image')
    axes[0].axis('off')
    axes[1].imshow(tgt_img_PIL)
    axes[1].arrow(tgt_pos[0], tgt_pos[1], tgt_dir[0]*100, tgt_dir[1]*100, color='r', linewidth=2.5, head_width=12)
    axes[1].set_title('target image')
    axes[1].axis('off')
    if filename:
        plt.savefig(f'{save_root}/{filename}.png')
    else:
        plt.savefig(f'{save_root}/max_xy_linear.png')
        

def visualize_ft_pca(feature_map):
    '''
    ft: 1, C, H, W
    use PCA to decompose C into 3 for visualization
    '''
    feature_map = feature_map.cpu().detach().numpy()

    # Ensure the feature map is of shape [1, C, H, W]
    assert feature_map.ndim == 4 and feature_map.shape[0] == 1, "Input tensor should have shape [1, C, H, W]"
    
    # Remove the batch dimension
    feature_map = feature_map[0]
    
    # Get the shape of the feature map
    C, H, W = feature_map.shape
    
    # Reshape the feature map to [C, H*W]
    reshaped_feature_map = feature_map.reshape(C, -1).T  # Now shape is [H*W, C]
    
    # Standardize the feature map
    scaler = StandardScaler()
    standardized_feature_map = scaler.fit_transform(reshaped_feature_map)
    
    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(standardized_feature_map)
    
    # Reshape PCA result back to [H, W, 3]
    pca_image = pca_result.reshape(H, W, 3)
    
    # Normalize the image to the range [0, 1]
    pca_image_min = pca_image.min()
    pca_image_max = pca_image.max()
    pca_image = (pca_image - pca_image_min) / (pca_image_max - pca_image_min)
    
    return pca_image