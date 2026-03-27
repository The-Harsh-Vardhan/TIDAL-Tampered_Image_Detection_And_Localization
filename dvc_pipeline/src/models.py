"""Model builder: UNet + ResNet-34 with 9ch input."""
import torch, segmentation_models_pytorch as smp

def build_model(encoder="resnet34", encoder_weights="imagenet", in_channels=9, num_classes=1, freeze_strategy="body_frozen_bn_unfrozen"):
    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=num_classes)
    if in_channels==9 and encoder_weights:
        with torch.no_grad():
            pre3 = smp.Unet(encoder_name=encoder,encoder_weights=encoder_weights,in_channels=3,classes=num_classes).encoder.conv1.weight.data.clone()
            model.encoder.conv1.weight.data = pre3.repeat(1,3,1,1)/3.0
    if freeze_strategy=="body_frozen_bn_unfrozen":
        for n,p in model.encoder.named_parameters(): p.requires_grad = "bn" in n or "conv1" in n
        for p in model.decoder.parameters(): p.requires_grad=True
        for p in model.segmentation_head.parameters(): p.requires_grad=True
    return model

def get_model_info(model):
    t=sum(p.numel() for p in model.parameters()); tr=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params":t,"trainable_params":tr,"frozen_params":t-tr,"trainable_pct":round(100*tr/t,1)}
