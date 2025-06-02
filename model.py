# import torch

# class BiomarkerNet(torch.nn.Module):
#     def __init__(self, mode='fusion', reconstruction=False):
#         super().__init__()
#         assert mode in ['oct', 'fusion']
#         self.mode = mode
#         self.reconstruction = reconstruction
#         # Encoder
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             torch.nn.BatchNorm2d(128),
#             torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             torch.nn.BatchNorm2d(256),
#             torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(2),
#         )
#         self.img_proj = torch.nn.Linear(256 * 14 * 14, 128)
#         if mode == 'fusion':
#             self.clin_proj = torch.nn.Linear(2, 128)
#             self.img_weight = torch.nn.Parameter(torch.tensor(0.5))
#             self.clin_weight = torch.nn.Parameter(torch.tensor(0.5))
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(128, 512),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(512, 6)
#         )
#         if reconstruction:
#             self.decoder = torch.nn.Sequential(
#                 torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#                 torch.nn.BatchNorm2d(128),
#                 torch.nn.LeakyReLU(),
#                 torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#                 torch.nn.BatchNorm2d(64),
#                 torch.nn.LeakyReLU(),
#                 torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#                 torch.nn.BatchNorm2d(32),
#                 torch.nn.LeakyReLU(),
#                 torch.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
#                 torch.nn.Tanh()
#             )

#     def forward(self, x_img, x_clin=None):
#         x_feat = self.features(x_img)
#         x_img_flat = x_feat.view(x_feat.size(0), -1)
#         x_emb = self.img_proj(x_img_flat)
#         if self.mode == 'oct':
#             fused = x_emb
#         elif self.mode == 'fusion':
#             x_clin_emb = self.clin_proj(x_clin)
#             fused = self.img_weight * x_emb + self.clin_weight * x_clin_emb
#         pred = self.classifier(fused)
#         if self.reconstruction:
#             recon_img = self.decoder(x_feat)
#             return pred, recon_img
#         else:
#             return pred

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


import torch

class BiomarkerNet(torch.nn.Module):
    def __init__(self, mode='fusion', reconstruction=False, img_embed_dim=128):
        """
        Args:
            mode: 'oct' (image only) or 'fusion' (image + clinical)
            reconstruction: If True, also outputs reconstructed image
            img_embed_dim: embedding size for image branch (default 128)
        """
        super().__init__()
        assert mode in ['oct', 'fusion']
        self.mode = mode
        self.reconstruction = reconstruction

        # Image Encoder
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.img_proj = torch.nn.Linear(256 * 14 * 14, img_embed_dim)

        if self.mode == 'fusion':
            # Clinical branch: project to img_embed_dim for easy fusion
            self.clin_proj = torch.nn.Linear(2, img_embed_dim)
            self.img_weight = torch.nn.Parameter(torch.tensor(0.5))
            self.clin_weight = torch.nn.Parameter(torch.tensor(0.5))

        # Classifier: input is img_embed_dim
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(img_embed_dim, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 6)
        )

        if self.reconstruction:
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(32),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
                torch.nn.Tanh()
            )

    def forward(self, x_img, x_clin=None):
        x_feat = self.features(x_img)
        x_img_flat = x_feat.view(x_feat.size(0), -1)
        x_emb = self.img_proj(x_img_flat)

        if self.mode == 'oct':
            fused = x_emb
        elif self.mode == 'fusion':
            assert x_clin is not None, "Clinical data must be provided in fusion mode."
            x_clin_emb = self.clin_proj(x_clin)
            fused = self.img_weight * x_emb + self.clin_weight * x_clin_emb

        pred = self.classifier(fused)
        if self.reconstruction:
            recon_img = self.decoder(x_feat)
            return pred, recon_img
        else:
            return pred
