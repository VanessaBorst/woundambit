# WoundAmbit: Pretrained Weights

The framework by default expects pretrained weights to be located in `./data/pretrained`. 

The weights, while automatically downloaded on first use in the case of U-Net, FUSegNet and HiFormer, for most models have to be 
downloaded manually. The following links can be used for this purpose:

- [FCBFormer](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth): ImageNet-1K (according to GitHub)
- [HarDNet-DFUS](https://huggingface.co/kytimmylai/DFUS-HarDNet/resolve/main/DFUC/kingnet53.pth): ImageNet-1K (according to GitHub issue #1)
- [InternImage](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth): ImageNet-1K (according to GitHub)
- [MISSFormer MiT-B1](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth): ImageNet-1K[^1]
- [SegFormer MiT-B3](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/embed.aspx?UniqueId=bd981ee6-491e-4dab-abd0-984343179cc6): ImageNet-1K (according to GitHub[^2])
- [SegNeXt](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/files/?p=%2Fmscan_l.pth&dl=1): ImageNet (according to GitHub)
- [TransNeXt](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true): ImageNet-1K (according to GitHub)
- [ConvNeXt-S backbone](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth): ImageNet-1K[^3]

[^1]: Except for the head, the checkpoint is identical to the one provided as "pretrained on ImageNet-1K" in 
the [official SegFormer repository](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
(after conversion to the MMSegmentation style with the dedicated [script](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/tools/model_converters/mit2mmseg.py)).

[^2]: We use the [weights from the official SegFormer repo](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/download.aspx?UniqueId=bd981ee6-491e-4dab-abd0-984343179cc6&Translate=false). 
The MMSegmentation-styled version can be found [here](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth).

[^3]: We use the pretrained weights linked in the [MMSegmentation Configs ReadMe](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/convnext), 
not the ones from the original ConvNeXt repo.